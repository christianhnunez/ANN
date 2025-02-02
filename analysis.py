from ANN import TrainANN, predictANN
from sidebander import getBDTANNsidebands
import ROOT
from ROOT import TFile, TTree
import root_numpy as rnp
from keras.layers.core import Dense, Dropout, Activation
from HistoLib import HistoTool, PhysicsProcess, Cut, Make1DPlots, DrawPlotsWithCutsForEachSample, mBB_Binning_long
from rootpy.io import root_open
import AtlasStyle as Atlas
from scipy.stats import pearsonr, binned_statistic
import numpy as np
from array import array
from copy import deepcopy
from bdt import buildBDT, buildBDT_new, predictBDT, predictBDTonSideband
from utils import overlayed_fig5, getPearsonDist, publishPearson, save_ANN_dataset, save_BDT_predictions, save_MVA_array, save_BDT_sideband_predictions, save_ANN_predictions, scatterMassScore
import matplotlib
matplotlib.use("PS")
import matplotlib.pyplot as plt
import logging 
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING) 
logging.getLogger('matplotlib').setLevel(logging.WARNING) 

# Filename for data storage
filename = "analysis_results.root"
f = TFile(filename, "recreate")
f.Close()

## declare event variables of interests
var_bb  = ["mBB", "pTBB", "eventWeight"]
# var_ANN = ["mJJ", "pTJJ", "cosTheta_boost",  "mindRJ1_Ex", "max_J1J2",
#           "eta_J_star", "deltaMJJ",
#           "pT_balance"]

var_ANN = ["mJJ","pTJJ","asymJJ","pT_balance","dEtaBBJJ", "dEtaJJ",
           "dPhiBBJJ","angleAsymBB","nJets20pt","mindRJ1_Ex",
           "mindRJ2_Ex","NTrk500PVJ2"]

var_all = var_bb+var_ANN

# TEST SIDEBANDS?
testBDTonSidebands = False

## create physics process
SAMPLES = {}
SAMPLES["sherpa"] = PhysicsProcess("Sherpa Multi-b", "1for2cen_loose/tree_mcmjfullreweight_d.root ")
SAMPLES["vbf"]    = PhysicsProcess("VBF H->bb",      "1for2cen_loose/tree_vbf_ade.root")

## load all varibles
for s in SAMPLES:
    for v in var_all:
        SAMPLES[s].AddEventVarFromTree(v)

## make pTbb cuts
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
Make1DPlots(SAMPLES, "mBB", "mBB", "eventWeight", pTBBcuts, mBB_Binning_long)


## initiate drawtool 
DrawTool = HistoTool()
DrawTool.sqrtS = "13"


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
for s in ["sherpa", "vbf"]:
    mva_arr = None
    # first variable is the label
    idx = SAMPLES[s].var["eventWeight"]>0

    if s == "sherpa":
        mva_arr = np.array([0]*len(SAMPLES[s].var["eventWeight"]))
    if s == "vbf":
        mva_arr = np.array([1]*len(SAMPLES[s].var["eventWeight"]))

    print (idx)
    print ("idx")
    print (mva_arr)


    mva_arr = mva_arr[idx]

    mva_arr = np.vstack( (mva_arr, SAMPLES[s].var["eventWeight"][idx])) ## second variable is the evtweight
    mva_arr = np.vstack( (mva_arr, SAMPLES[s].var["mBB"][idx]) ) ## third variable is mbb

    for var in var_ANN:
        
        #normed  = SAMPLES[s].var[var][idx]
        #normed  = (normed-np.mean(normed))/(np.std(normed))
        #mva_arr = np.vstack( (mva_arr, normed) ) ## then input features
        mva_arr = np.vstack( (mva_arr, SAMPLES[s].var[var][idx]) )
    MVAInputs[s] = mva_arr


############################
### DERIVED VARIABLES (NEW VARS)
###

# Creating new variables:
var_NEW = ["mJJ","pTJJ","asymJJ","pT_balance","dEtaBBJJ_div_dEtaJJ",
           "dPhiBBJJ","angleAsymBB","nJets20pt","min_mindRJ1_Ex_10",
           "min_mindRJ2_Ex_10","NTrk500PVJ2"]
mm = ["label", "eventWeight", "mBB"] + var_NEW
print("using new ANN variables")
for i in range(0, len(mm)):
    print(i, " : ", mm[i])

# Taking MVAInputs and derive:
orig = MVAInputs
MVAInputs = {}
for s in ["sherpa", "vbf"]:
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

## prepare MVA data sets
train_frac      = 0.8
n_bkg           = MVAInputs["sherpa"].shape[1]
n_sig           = MVAInputs["vbf"].shape[1]

# first concatenate sig and bkg and take transpose
# the resulting shapes are Nevent x Nvariable
MVA_train_array = np.concatenate( (MVAInputs["sherpa"][:, 0:int(n_bkg*train_frac)], MVAInputs["vbf"][:, 0:int(n_sig*train_frac)] ), axis =1).T
MVA_test_array  = np.concatenate( (MVAInputs["sherpa"][:, int(n_bkg*train_frac):], MVAInputs["vbf"][:, int(n_sig*train_frac):] ), axis =1).T

# random permute the training and testing set orders
np.random.seed(10)
train_index_perm = np.random.permutation( np.array(range(MVA_train_array.shape[0])) )
test_index_perm  = np.random.permutation( np.array(range(MVA_test_array.shape[0])) )
MVA_train_array  = MVA_train_array[train_index_perm,:]
MVA_test_array   = MVA_test_array[test_index_perm,:]

# print("\n\n FOR TESTING: Taking only 5 percent of train and test. \n\n")
# MVA_train_array = MVA_train_array[:int(MVA_train_array.shape[0]*0.05)]
# MVA_test_array = MVA_test_array[:int(MVA_test_array.shape[0]*0.05)]
# print("MVA_train_array length = " + str(MVA_train_array.shape[0]))
# print("MVA_test_array length = " + str(MVA_test_array.shape[0]))

# Down/up-sampling
#MVA_train_array = downsample_bkg(MVA_train_array, k=1)
#MVA_train_array = upsample_sig(MVA_train_array)


# Save the MVA arrays for train and test
branch_names = ["label", "eventWeight", "mBB"] + var_NEW
save_MVA_array(filename, "MVA_train_array", MVA_train_array, branch_names)
save_MVA_array(filename, "MVA_test_array", MVA_test_array, branch_names)

#########
## BDT ##
#########

## prepare dataset for bdt

bdt_dataset = {}
bdt_dataset["X_train"]       = MVA_train_array[:, 3:]
bdt_dataset["X_test"]        = MVA_test_array[:, 3:]
bdt_dataset["Y_train"]       = MVA_train_array[:, 0]
bdt_dataset["Y_test"]        = MVA_test_array[:, 0]
bdt_dataset["weights_train"] = MVA_train_array[:, 1]
bdt_dataset["weights_test"]  = MVA_test_array[:, 1]

## train and predict bdt
bdt_model      =   buildBDT(bdt_dataset)
bdt_results    =   predictBDT(bdt_model, bdt_dataset)
train_mass_score_corr  =  round(pearsonr(bdt_results["pred_train"][MVA_train_array[:,0]==0], MVA_train_array[:, 2][MVA_train_array[:,0]==0])[0], 5)
test_mass_score_corr   =  round(pearsonr(bdt_results["pred_test"][MVA_test_array[:,0]==0],  MVA_test_array[:, 2][MVA_test_array[:,0]==0])[0], 5)

# get Pearson distribution:
getPearsonDist(bdt_model, bdt_dataset, bdt_results, MVA_test_array,parts=10, ANN=False)

# publishPearson stats:
percentile_list = [90,95,99,99.9,99.99994]
publishPearson(test_mass_score_corr, (MVA_test_array[MVA_test_array[:,0]==0]).shape[0], percentile_list=percentile_list, sideband=False)

# scatter mass vs. score
scatterMassScore(MVA_test_array, bdt_results)

## plot the ROC curve
plt.figure()
plt.plot(  bdt_results["roc_train"][1],  bdt_results["roc_train"][0], 
           label="roc training set, AUC="+str(bdt_results["auc_train"])+ " rho(mass,score)="+str(train_mass_score_corr))
plt.plot(  bdt_results["roc_test"][1],  bdt_results["roc_test"][0], 
           label="roc test set, AUC="+str(bdt_results["auc_test"])+ " rho(mass,score)="+str(test_mass_score_corr))
plt.legend()
plt.xlabel("Signal Eff")
plt.ylabel("Background Eff")
plt.savefig("roc_bdt.png")
plt.close()

## plot the PRC curve
plt.figure()
plt.plot(  bdt_results["prc_train"][1],  bdt_results["prc_train"][0], 
           label="prc training set, AUC="+str(bdt_results["auc_prc_train"])+ " rho(mass,score)="+str(train_mass_score_corr))
plt.plot(  bdt_results["prc_test"][1],  bdt_results["prc_test"][0], 
           label="prc test set, AUC="+str(bdt_results["auc_prc_test"])+ " rho(mass,score)="+str(test_mass_score_corr))
plt.legend()
plt.xlabel("recall")
plt.ylabel("precision")
plt.savefig("prc_bdt.png")
plt.close()

save_BDT_predictions(filename, bdt_results, train=True)
save_BDT_predictions(filename, bdt_results, train=False)
overlayed_fig5(MVA_train_array, MVA_test_array, bdt_results, ANN=False)

## plot Mbb profile plot
# ===
# TRAIN
# ===

# Background only
means_result = binned_statistic(MVA_train_array[:, 2][MVA_train_array[:,0]==0], 
                                [bdt_results["pred_train"][MVA_train_array[:,0]==0], 
                                 bdt_results["pred_train"][MVA_train_array[:,0]==0]**2], 
                                bins=10, range=[mass_range_low, mass_range_high], statistic='mean')
means, means2 = means_result.statistic
standard_deviations = np.sqrt(means2 - means**2)
bin_edges = means_result.bin_edges
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
plt.errorbar(x=bin_centers, y=means, yerr=standard_deviations, linestyle='none', marker='o', markersize=4, alpha=0.7, label="bkg", solid_capstyle='projecting', capsize=4)

# Signal only
means_result = binned_statistic(MVA_train_array[:, 2][MVA_train_array[:,0]==1], 
                                [bdt_results["pred_train"][MVA_train_array[:,0]==1], 
                                 bdt_results["pred_train"][MVA_train_array[:,0]==1]**2], 
                                bins=10, range=[mass_range_low, mass_range_high], statistic='mean')
means, means2 = means_result.statistic
standard_deviations = np.sqrt(means2 - means**2)
bin_edges = means_result.bin_edges
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
plt.errorbar(x=bin_centers, y=means, yerr=standard_deviations, linestyle='none', marker='o', markersize=4, alpha=0.7, label="sig", solid_capstyle='projecting', capsize=4)

# Show
plt.ylabel("Score")
plt.xlabel("Mbb (GeV)")
plt.grid()
plt.legend(fancybox=True)
plt.savefig("profile_bdt_train_mass.png")
plt.close()


# === 
# TEST
# ===
# BKG only
means_result = binned_statistic(MVA_test_array[:, 2][MVA_test_array[:,0]==0], 
                                [bdt_results["pred_test"][MVA_test_array[:,0]==0], 
                                 bdt_results["pred_test"][MVA_test_array[:,0]==0]**2], 
                                bins=10, range=[mass_range_low, mass_range_high], statistic='mean')
means, means2 = means_result.statistic
standard_deviations = np.sqrt(means2 - means**2)
bin_edges = means_result.bin_edges
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
plt.errorbar(x=bin_centers, y=means, yerr=standard_deviations, linestyle='none', marker='o', markersize=4, alpha=0.7, label="bkg", solid_capstyle='projecting', capsize=4)

# Signal only
means_result = binned_statistic(MVA_test_array[:, 2][MVA_test_array[:,0]==1], 
                                [bdt_results["pred_test"][MVA_test_array[:,0]==1], 
                                 bdt_results["pred_test"][MVA_test_array[:,0]==1]**2], 
                                bins=10, range=[mass_range_low, mass_range_high], statistic='mean')
means, means2 = means_result.statistic
standard_deviations = np.sqrt(means2 - means**2)
bin_edges = means_result.bin_edges
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
plt.errorbar(x=bin_centers, y=means, yerr=standard_deviations, linestyle='none', marker='o', markersize=4, alpha=0.7, label="sig", solid_capstyle='projecting', capsize=4)

# Show
plt.ylabel("Score")
plt.xlabel("Mbb (GeV)")
plt.grid()
plt.legend(fancybox=True)
plt.savefig("profile_bdt_test_mass.png")
plt.close()


# Medians
means_result = binned_statistic(MVA_test_array[:, 2][MVA_test_array[:,0]==0], 
                                bdt_results["pred_test"][MVA_test_array[:,0]==0], 
                                bins=10, range=[mass_range_low, mass_range_high], statistic='median')
med = means_result.statistic
bin_edges = means_result.bin_edges
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
bin_number = means_result.binnumber
plt.plot(bin_centers, med, linestyle='none', marker='o', markersize=10, alpha=0.7, label="median bkg")
plt.title("BDT on Test")
lowest = np.amin(med)
print(med)
plt.legend()
plt.grid()
plt.savefig("median_bdt_test.png")
plt.close()

#################################################
#             TEST BDT ON SIDEBANDS             #
#################################################
if testBDTonSidebands:
    bdt_sideband, ann_sideband, real_data = getBDTANNsidebands(var_all)
    bdt_sideband_results = predictBDTonSideband(bdt_model, bdt_sideband)
    test_mass_score_corr   =  round(pearsonr(bdt_sideband_results["pred_test"][real_data[:,0]==0],  real_data[:, 2][real_data[:,0]==0])[0], 4)

    # Background only
    means_result = binned_statistic(real_data[:, 2][real_data[:,0]==0], 
                                    [bdt_sideband_results["pred_test"][real_data[:,0]==0], 
                                     bdt_sideband_results["pred_test"][real_data[:,0]==0]**2], 
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
    plt.savefig("SIDEBAND_profile_BDT.png")
    plt.close()

    ## plot the ROC curve
    plt.figure()
    plt.plot(  bdt_sideband_results["roc_test"][1],  bdt_sideband_results["roc_test"][0], 
               label="roc test set, AUC="+str(bdt_sideband_results["auc_test"])+ " rho(mass,score)="+str(test_mass_score_corr))
    plt.legend()
    plt.xlabel("Signal Eff")
    plt.ylabel("Background Eff")
    plt.savefig("SIDEBAND_roc_BDT.png")
    plt.close()

    ## plot the PRC curve
    plt.figure()
    plt.plot(  bdt_sideband_results["prc_test"][1],  bdt_sideband_results["prc_test"][0], 
               label="prc test set, AUC="+str(bdt_sideband_results["auc_prc_test"])+ " rho(mass,score)="+str(test_mass_score_corr))
    plt.legend()
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.savefig("SIDEBAND_prc_BDT.png")
    plt.close()

    # medians
    means_result = binned_statistic(real_data[:, 2][real_data[:,0]==0], 
                                bdt_sideband_results["pred_test"][real_data[:,0]==0], 
                                bins=10, range=[mass_range_low, mass_range_high], statistic='median')
    med = means_result.statistic
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    bin_number = means_result.binnumber
    plt.plot(bin_centers, med, linestyle='none', marker='o', markersize=10, alpha=0.7, label="median bkg")
    plt.title("BDT on Sideband")
    lowest = np.amin(med)
    print(med)
    plt.legend()
    plt.grid()
    plt.savefig("median_bdt_SIDEBAND.png")
    plt.close()

    save_BDT_sideband_predictions(filename, bdt_sideband_results, train=False)
    getPearsonDist(bdt_model, bdt_sideband, bdt_sideband_results, real_data, parts=10, ANN=False, sideband=True)

###########
### ANN ###
###########

## prepare dataset for ann
ann_dataset      = {}
mass_bin_size    = 10
mass_bins        = range(mass_range_low, mass_range_high+mass_bin_size, mass_bin_size  )
mass_cat_train   = []
mass_cat_test    = []
for i in range(len(mass_bins)-1):
    mass_cat_train.append( np.logical_and( MVA_train_array[:, 2]>mass_bins[i], 
                                           MVA_train_array[:, 2]<mass_bins[i+1] ) )

    mass_cat_test.append( np.logical_and( MVA_test_array[:, 2]>mass_bins[i], 
                                          MVA_test_array[:, 2]<mass_bins[i+1] ) )
mass_cat_train   = np.array(mass_cat_train).T
mass_cat_test    = np.array(mass_cat_test).T

ann_dataset["X_train"]       = MVA_train_array[:, 3:]
ann_dataset["X_test"]        = MVA_test_array[:, 3:]
ann_dataset["Y_train"]       = np.hstack( (mass_cat_train, MVA_train_array[:, [0]]))
ann_dataset["Y_test"]        = np.hstack( (mass_cat_test,  MVA_test_array[:, [0]]))
ann_dataset["weights_train"] = MVA_train_array[:, 1]
ann_dataset["weights_test"]  = MVA_test_array[:, 1]

save_ANN_dataset(filename, "ann_dataset_train", ann_dataset, train=True)
save_ANN_dataset(filename, "ann_dataset_test", ann_dataset, train=False)

# For the megaROC curve, which combines the results of the ROC curves of the all lambdas tested
# Format example for lambda=10: megaROC['lamb10'] = miniROC
# where miniROC has keys "lamb" (for check), "ann_results", "rho_train", "rho_test"
megaROC = {}
#for lamb in [0, 2.0, 10.0]:
for lamb in [10.0]:

    # set gamma:
    gam = 1.0

    model, hist = TrainANN( ann_dataset, lamb=lamb, gam=gam, clpretrain = 2, adpretrain = 2, 
                            epoch=120,  batch_size = 256 , nMBBbins = 10, lr=1e-4)

    plt.figure()
    ax1 = plt.subplot(311)  
    plt.plot(  range(len(hist["ann"])), hist["ann"], label="training loss ann")
    plt.legend()
    ax2 = plt.subplot(312)  
    plt.plot(  range(len(hist["ann_cl"])), hist["ann_cl"], label="training loss cl only")
    plt.legend()
    ax3 = plt.subplot(313)  
    plt.plot(  range(len(hist["ann_ad"])), hist["ann_ad"], label="training loss ad only")
    plt.legend()
    plt.xlabel("epoch")
    plt.savefig("training_ann_hist_lambda_{!s}.png".format(str(lamb)))
    plt.close()

    ann_results = predictANN(model, ann_dataset)
    train_mass_score_corr  =  round(pearsonr(ann_results["pred_train"][MVA_train_array[:,0]==0], MVA_train_array[:, 2][MVA_train_array[:,0]==0])[0], 5)
    test_mass_score_corr   =  round(pearsonr(ann_results["pred_test"][MVA_test_array[:,0]==0],  MVA_test_array[:, 2][MVA_test_array[:,0]==0])[0], 5)
    
    # Write Pearson dist to file
    getPearsonDist(model, ann_dataset, ann_results, MVA_test_array, lamb, parts=10, ANN=True)

    # publishPearson stats:
    publishPearson(test_mass_score_corr, (MVA_test_array[MVA_test_array[:,0]==0]).shape[0], percentile_list=percentile_list, lamb=lamb, sideband=False)

    # scatter mass vs. score
    scatterMassScore(MVA_test_array, ann_results)


    ## plot the fig5 curve:
    overlayed_fig5(MVA_train_array, MVA_test_array, ann_results, ANN=True, lamb=lamb)

    ## plot the ROC curve
    plt.figure()
    plt.plot(  ann_results["roc_train"][1],  ann_results["roc_train"][0], 
               label="roc training set, AUC="+str(ann_results["auc_train"])+ " rho(mass,score)="+str(train_mass_score_corr))
    plt.plot(  ann_results["roc_test"][1],  ann_results["roc_test"][0], 
               label="roc test set, AUC="+str(ann_results["auc_test"])+ " rho(mass,score)="+str(test_mass_score_corr))
    plt.figtext(0.2, 0.2, "lambda={!s}".format(str(lamb)))
    plt.legend()
    plt.xlabel("Signal Eff")
    plt.ylabel("Background Eff")
    plt.savefig("roc_ann_lambda{!s}.png".format(str(lamb)))
    plt.close()

    ## plot the PRC curve
    plt.figure()
    plt.plot(  ann_results["prc_train"][1],  ann_results["prc_train"][0], 
               label="prc training set, AUC="+str(ann_results["auc_prc_train"])+ " rho(mass,score)="+str(train_mass_score_corr))
    plt.plot(  ann_results["prc_test"][1],  ann_results["prc_test"][0], 
               label="prc test set, AUC="+str(ann_results["auc_prc_test"])+ " rho(mass,score)="+str(test_mass_score_corr))
    plt.legend()
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.savefig("prc_ann_lambda{!s}.png".format(str(lamb)))
    plt.close()

    # ================================================
    # Plotting Mbb profile plot, separated sig/bkg
    # ================================================
    # TRAIN ->
    # Background only
    means_result = binned_statistic(MVA_train_array[:, 2][MVA_train_array[:,0]==0], 
                                    [ann_results["pred_train"][MVA_train_array[:,0]==0], 
                                     ann_results["pred_train"][MVA_train_array[:,0]==0]**2], 
                                    bins=10, range=[mass_range_low, mass_range_high], statistic='mean')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    plt.errorbar(x=bin_centers, y=means, yerr=standard_deviations, linestyle='none', marker='o', markersize=4, alpha=0.7, label="bkg", solid_capstyle='projecting', capsize=4)

    # Signal only
    means_result = binned_statistic(MVA_train_array[:, 2][MVA_train_array[:,0]==1], 
                                    [ann_results["pred_train"][MVA_train_array[:,0]==1], 
                                     ann_results["pred_train"][MVA_train_array[:,0]==1]**2], 
                                    bins=10, range=[mass_range_low, mass_range_high], statistic='mean')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    plt.errorbar(x=bin_centers, y=means, yerr=standard_deviations, linestyle='none', marker='o', markersize=4, alpha=0.7, label="sig", solid_capstyle='projecting', capsize=4)

    # Show
    plt.ylabel("Score")
    plt.xlabel("Mbb (GeV)")
    plt.grid()
    plt.legend(fancybox=True)
    plt.savefig("profile_ann_train_mass_lambda{!s}.png".format(str(lamb)))
    plt.close()


    # TEST -->
    # Background only
    means_result = binned_statistic(MVA_test_array[:, 2][MVA_test_array[:,0]==0], 
                                    [ann_results["pred_test"][MVA_test_array[:,0]==0], 
                                     ann_results["pred_test"][MVA_test_array[:,0]==0]**2], 
                                    bins=10, range=[mass_range_low, mass_range_high], statistic='mean')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    plt.errorbar(x=bin_centers, y=means, yerr=standard_deviations, linestyle='none', marker='o', markersize=4, alpha=0.7, label="bkg", solid_capstyle='projecting', capsize=4)

    # Signal only
    means_result = binned_statistic(MVA_test_array[:, 2][MVA_test_array[:,0]==1], 
                                    [ann_results["pred_test"][MVA_test_array[:,0]==1], 
                                     ann_results["pred_test"][MVA_test_array[:,0]==1]**2], 
                                    bins=10, range=[mass_range_low, mass_range_high], statistic='mean')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    plt.errorbar(x=bin_centers, y=means, yerr=standard_deviations, linestyle='none', marker='o', markersize=4, alpha=0.7, label="sig", solid_capstyle='projecting', capsize=4)

    # Show
    plt.ylabel("Score")
    plt.xlabel("Mbb (GeV)")
    plt.grid()
    plt.legend(fancybox=True)
    plt.savefig("profile_ann_test_mass_lambda{!s}.png".format(str(lamb)))
    plt.close()

    # Medians:
    means_result = binned_statistic(MVA_test_array[:, 2][MVA_test_array[:,0]==0], 
                                    ann_results["pred_test"][MVA_test_array[:,0]==0], 
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
    plt.savefig("median_ann_test_mass_lambda{!s}.png".format(str(lamb)))
    plt.close()

    # ========== #
    # Compiling data for the megaROC curve, combining the results of the ROC curves of the all lambdas tested
    miniROC = {}
    miniROC["lamb"] = lamb
    miniROC["ann_results"] = ann_results
    miniROC["rho_train"] = train_mass_score_corr
    miniROC["rho_test"] = test_mass_score_corr

    # Add mini to mega:
    miniName = "lamb" + str(lamb)
    megaROC[miniName] = miniROC

    save_ANN_predictions(filename, "ann_results_pred_train"+"_lamb"+str(lamb), ann_results, train=True)
    save_ANN_predictions(filename, "ann_results_pred_test"+"_lamb"+str(lamb), ann_results, train=False)

# Finally, create the megaROC curve. Test with just the test data.
plt.figure()
count = 0
for key in megaROC.keys():
    cols = ['r', 'b', 'g', 'pink', 'teal']
    # Retrieve data from mega->mini
    ann_results = megaROC[key]["ann_results"]
    test_mass_score_corr = megaROC[key]["rho_test"]
    lamb = megaROC[key]["lamb"]
    # Create plot
    legendName = "lamb = " + key.split("lamb")[1]
    plt.plot(  ann_results["roc_test"][1],  ann_results["roc_test"][0], 
               label= legendName + " | roc test set, AUC="+str(ann_results["auc_test"])+ " rho(mass,score)="+str(test_mass_score_corr), alpha=0.7, color=cols[count])
    count = count + 1
plt.legend(fancybox=True)
plt.xlabel("Signal Efficiency")
plt.ylabel("Background Efficiency")
plt.savefig("roc_ann_HPsearch.png")
plt.close()










