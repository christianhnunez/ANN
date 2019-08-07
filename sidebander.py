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


def getBDTANNsidebands(var_all):
	var_ANN = var_all[3:]
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
	# Finally, cut real_data for sidebands
	lower = real_data[real_data[:, 2] < 100]
	upper = real_data[real_data[:, 2] > 140]
	real_data = np.vstack((lower, upper))

	#################################################
	# 	          GET BDT AND ANN SETS              #
	#################################################

	# Prepare sideband for ann
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

	## prepare dataset for bdt
	bdt_dataset = {}
	bdt_dataset["X_test"]        = real_data[:, 3:]
	bdt_dataset["Y_test"]        = real_data[:, 0]
	bdt_dataset["weights_test"]  = real_data[:, 1]


	return bdt_dataset, ann_dataset, real_data


