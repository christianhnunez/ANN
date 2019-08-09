from ROOT import TFile, TTree
import root_numpy as rnp
import json
from HistoLib import HistoTool, PhysicsProcess, Cut, Make1DPlots, DrawPlotsWithCutsForEachSample, mBB_Binning_long
from rootpy.io import root_open
from scipy.stats import pearsonr, binned_statistic
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import matplotlib
matplotlib.use("PS")
import matplotlib.pyplot as plt
from keras.models import model_from_json
from ANN import predictANN
from utils import getPearsonDist, publishPearson

#################################################
# 		            SETUP					    #
#################################################
lamb=0

#weights = "L23/L23-hpc1.h5"
#json_string = "L23/L23-hpc1.json"

json_string = "ANN_lambda0_clpretrain2_adpretrain2_epoch120_minibatch256_mBBbins10_architecture.json"
weights = "ANN_lambda0_clpretrain2_adpretrain2_epoch120_minibatch256_mBBbins10_model_weights.h5"

#################################################
# 		     SIDEBAND DATA MANAGEMENT           #
#################################################
var_bb  = ["mBB", "pTBB", "eventWeight"]
var_ANN = ["mJJ","pTJJ","asymJJ","pT_balance","dEtaBBJJ", "dEtaJJ",
           "dPhiBBJJ","angleAsymBB","nJets20pt","mindRJ1_Ex",
           "mindRJ2_Ex","NTrk500PVJ2"]
var_all = var_bb+var_ANN
print("Number of variables used: ", len(var_all))
print(var_all)


# create physics processes
SAMPLES = {}
SAMPLES["sideband"] = PhysicsProcess("DataSidebands", "1for2cen_loose/tree_20161718.root")
SAMPLES["embed"] = PhysicsProcess("Embed", "1for2cen_loose/tree_embedData_ade.root")

# load variables:
for s in SAMPLES: 
    for v in var_all:
        SAMPLES[s].AddEventVarFromTree(v)

# Adding cuts to SAMPLES[s], a PhysicsProcess instance
for s in SAMPLES:
    SAMPLES[s].AddCut("pTBB>70", SAMPLES[s].var["pTBB"]>70)
    SAMPLES[s].AddCut("pTBB>80", SAMPLES[s].var["pTBB"]>80)
    SAMPLES[s].AddCut("pTBB>90", SAMPLES[s].var["pTBB"]>90)
    SAMPLES[s].AddCut("pTBB>100", SAMPLES[s].var["pTBB"]>100)
    SAMPLES[s].AddCut("pTBB>110", SAMPLES[s].var["pTBB"]>110)
    SAMPLES[s].AddCut("pTBB>120", SAMPLES[s].var["pTBB"]>120)
    SAMPLES[s].AddCut("pTBB>130", SAMPLES[s].var["pTBB"]>130)
    SAMPLES[s].AddCut("pTBB>140", SAMPLES[s].var["pTBB"]>140)

pTBBcuts = ["pTBB>70", "pTBB>80", "pTBB>90", "pTBB>100", 
            "pTBB>110", "pTBB>120", "pTBB>130", "pTBB>140"]

## apply pTbb cuts (2 central use 120 GeV, to be checked)
pTCut           = 120
mass_range_low  = 70
mass_range_high = 170
for s in SAMPLES:
    SAMPLES[s].var["eventWeight"] = (SAMPLES[s].var["pTBB"]>pTCut)           * SAMPLES[s].var["eventWeight"]
    SAMPLES[s].var["eventWeight"] = (SAMPLES[s].var["mBB"]<mass_range_high) * SAMPLES[s].var["eventWeight"]
    SAMPLES[s].var["eventWeight"] = (SAMPLES[s].var["mBB"]>mass_range_low)  * SAMPLES[s].var["eventWeight"]

    print (SAMPLES[s].var["eventWeight"])
    print (SAMPLES[s].var["mBB"])

## stack sample variables
MVAInputs = {}
for s in ["sideband", "embed"]:
    mva_arr = None
    # first variable is the label
    idx = SAMPLES[s].var["eventWeight"]>0

    if s == "sideband":
        mva_arr = np.array([0]*len(SAMPLES[s].var["eventWeight"]))
    if s == "embed":
        mva_arr = np.array([0]*len(SAMPLES[s].var["eventWeight"]))

    print (idx)
    print ("idx")
    print (mva_arr)


    mva_arr = mva_arr[idx]

    mva_arr = np.vstack( (mva_arr, SAMPLES[s].var["eventWeight"][idx])) ## second variable is the evtweight
    mva_arr = np.vstack( (mva_arr, SAMPLES[s].var["mBB"][idx]) ) ## third variable is mbb

    for var in var_ANN:
        mva_arr = np.vstack( (mva_arr, SAMPLES[s].var[var][idx]) )
    MVAInputs[s] = mva_arr


############################
### DERIVED VARIABLES (MATT)
###

# Creating new variables:
var_NEW = ["mJJ","pTJJ","asymJJ","pT_balance","dEtaBBJJ/dEtaJJ",
           "dPhiBBJJ","angleAsymBB","nJets20pt","min(mindRJ1_Ex,10)",
           "min(mindRJ2_Ex,10)","NTrk500PVJ2"]
mm = ["label", "eventWeight", "mBB"] + var_NEW
print("using new ANN variables")
for i in range(0, len(mm)):
    print(i, " : ", mm[i])

# Taking MVAInputs and derive:
orig = MVAInputs
MVAInputs = {}
for s in ["sideband", "embed"]:
    arr = np.vstack((orig[s][0, :], 
                     orig[s][1, :], 
                     orig[s][2, :], 
                     orig[s][3, :],
                     orig[s][4, :],
                     orig[s][5, :],
                     orig[s][6, :],
                     (orig[s][7, :]/orig[s][8, :]),
                     orig[s][9, :],
                     orig[s][10, :],
                     orig[s][11, :],
                     np.minimum(orig[s][12, :], np.array([10]*(orig[s].shape[1]))),
                     np.minimum(orig[s][13, :], np.array([10]*(orig[s].shape[1]))),
                     orig[s][14, :]))
    MVAInputs[s] = arr

# ### 
# ### END DERIVED VARIABLES
# #########################

sideband_set = MVAInputs["sideband"].T
print("sideband_set (uncut): ", sideband_set.shape)
sig_data = sideband_set[sideband_set[:, 0]==1]
bkg_data = sideband_set[sideband_set[:, 0]==0]
print("sig set (uncut): ", sig_data.shape)
print("bkg set (uncut): ", bkg_data.shape)

embed_set = MVAInputs["embed"].T
print("embed_set (uncut): ", embed_set.shape)
sig_data_em = embed_set[embed_set[:, 0]==1]
bkg_data_em = embed_set[embed_set[:, 0]==0]
print("sig set (uncut): ", sig_data_em.shape)
print("bkg set (uncut): ", bkg_data_em.shape)
# Multliply embedded eventWeights by -1
embed_set[:, 1] = embed_set[:, 1] * -1

# Now, we combine embed and sidebands:
real_data = np.vstack((sideband_set, embed_set))
#Finally, cut real_data for sidebands
lower = real_data[real_data[:, 2] < 100]
upper = real_data[real_data[:, 2] > 140]
real_data = np.vstack((lower, upper))

#################################################
# 	          TEST ON SIDEBANDS                 #
#################################################

# model reconstruction from JSON:
with open(json_string, 'r') as json_file:
    model = model_from_json(json_file.read())
model.load_weights(weights)

# Prepare sideband for ann
mass_range_low = 70
mass_range_high = 170
ann_dataset      = {}
mass_bin_size    = 10
mass_bins        = range(mass_range_low, mass_range_high+mass_bin_size, mass_bin_size  )
mass_cat_test    = []
for i in range(len(mass_bins)-1):
    mass_cat_test.append( np.logical_and( real_data[:, 2]>mass_bins[i], 
                                          real_data[:, 2]<mass_bins[i+1] ) )
mass_cat_test    = np.array(mass_cat_test).T

ann_dataset["X_test"]        = real_data[:, 3:]
ann_dataset["Y_test"]        = np.hstack( (mass_cat_test,  real_data[:, [0]]))
ann_dataset["weights_test"]  = real_data[:, 1]

# Predict on sideband:
def predictOnSideband(model, dataset):
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']
    
    pred_test = model.predict( X_test )[:,0]
    roc_test   = roc_curve(Y_test[:, -1],  pred_test)
    auc_test   = round(auc(roc_test[1], roc_test[0], reorder=True), 4)
    prc_test = precision_recall_curve( Y_test[:, -1], pred_test)
    auc_prc_test = round(auc(prc_test[1], prc_test[0], reorder=True), 4)

    results = {"pred_test":     pred_test,             
               "roc_test":      roc_test,      
               "auc_test":      auc_test,     
               "prc_test":      prc_test,  
               "auc_prc_test":  auc_prc_test}

    return  results

ann_results = predictOnSideband(model, ann_dataset)
test_mass_score_corr   =  round(pearsonr(ann_results["pred_test"][real_data[:,0]==0],  real_data[:, 2][real_data[:,0]==0])[0], 3)

for key in ann_results:
	print(key)
print("\n\nsideband predict complete\n\n")

#################################################
# 	          TEST ON SIDEBANDS                 #
#################################################

# Background only
means_result = binned_statistic(real_data[:, 2][real_data[:,0]==0], 
                                [ann_results["pred_test"][real_data[:,0]==0], 
                                 ann_results["pred_test"][real_data[:,0]==0]**2], 
                                bins=10, range=[mass_range_low, mass_range_high], statistic='mean')
means, means2 = means_result.statistic
standard_deviations = np.sqrt(means2 - means**2)
bin_edges = means_result.bin_edges
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
plt.errorbar(x=bin_centers, y=means, yerr=standard_deviations, linestyle='none', marker='o', markersize=4, alpha=0.7, label="bkg", solid_capstyle='projecting', capsize=4)
plt.ylabel("Score")
plt.xlabel("Mbb (GeV)")
plt.grid()
plt.legend(fancybox=True)
plt.savefig("SIDEBAND_profile_ann_test_mass_lambda{!s}.png".format(str(lamb)))
plt.close()

## plot the ROC curve
plt.figure()
plt.plot(  ann_results["roc_test"][1],  ann_results["roc_test"][0], 
           label="roc test set, AUC="+str(ann_results["auc_test"])+ " rho(mass,score)="+str(test_mass_score_corr))
plt.figtext(0.2, 0.2, "lambda={!s}".format(str(lamb)))
plt.legend()
plt.xlabel("Signal Eff")
plt.ylabel("Background Eff")
plt.savefig("SIDEBAND_roc_ann_lambda{!s}.png".format(str(lamb)))
plt.close()

## plot the PRC curve
plt.figure()
plt.plot(  ann_results["prc_test"][1],  ann_results["prc_test"][0], 
           label="prc test set, AUC="+str(ann_results["auc_prc_test"])+ " rho(mass,score)="+str(test_mass_score_corr))
plt.legend()
plt.xlabel("recall")
plt.ylabel("precision")
plt.savefig("SIDEBAND_prc_ann_lambda{!s}.png".format(str(lamb)))
plt.close()

## medians
means_result = binned_statistic(real_data[:, 2][real_data[:,0]==0], 
                                ann_results["pred_test"][real_data[:,0]==0], 
                                bins=10, range=[mass_range_low, mass_range_high], statistic='median')
med = means_result.statistic
bin_edges = means_result.bin_edges
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
bin_number = means_result.binnumber
plt.plot(bin_centers, med, linestyle='none', marker='o', markersize=10, alpha=0.7, label="median bkg")
plt.title("ANN on Test, lamb = " + str(lamb))
lowest = np.amin(med)
plt.legend()
plt.grid()
plt.ylabel("Score")
plt.xlabel("Mbb (GeV)")
plt.savefig("median_ann_SIDEBAND_lambda{!s}.png".format(str(lamb)))
plt.close()

getPearsonDist(model, ann_dataset, ann_results, real_data, lamb=10.0, parts=10, ANN=True, sideband=True)
percentile_list = [90,95,99,99.9,99.99994]
publishPearson(test_mass_score_corr, (real_data[real_data[:,0]==0]).shape[0], percentile_list=percentile_list, ANN=True, sideband=True)









