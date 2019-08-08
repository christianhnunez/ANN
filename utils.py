# utility functions for root file recovery written by chnunez -- chnunez@stanford.edu 
from ROOT import TFile, TTree
import root_numpy as rnp
from HistoLib import HistoTool, PhysicsProcess, Cut, Make1DPlots, DrawPlotsWithCutsForEachSample, mBB_Binning_long
from rootpy.io import root_open
from scipy.stats import pearsonr, binned_statistic, norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------------------------
# analysis_results.root file general content
# ---------------------------------------------------------------------------------------------
# filename = "/eos/user/c/chnunez/SWAN_projects/ANN/analysis_results.root"
# f = root.TFile(filename)
# MVA_trees = ["MVA_train_array", "MVA_test_array"]
# bdt_trees = ["bdt_results_train", "bdt_results_test"]
# ann_dataset_trees = ["ann_dataset_train", "ann_dataset_test"]
# ann_results_trees = ["ann_results_pred_train_lamb0", "ann_results_pred_test_lamb0", 
#                      "ann_results_pred_train_lamb2.0", "ann_results_pred_test_lamb2.0",
#                      "ann_results_pred_train_lamb10.0", "ann_results_pred_test_lamb10.0"]
#
# Chosen features + extras (label, eventWeight, mBB)
# MVA_branches = ["label", "eventWeight", "mBB", "mJJ", "pTJJ", "cosTheta_boost", "mindRJ1_Ex",
#                 "max_J1J2", "eta_J_star", "QGTagger_NTracksJ2", "deltaMJJ", "pT_balance"]
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# Helper functions for recovery function
# ---------------------------------------------------------------------------------------------
def getMVAarray(f, treename, branches):
    MVA_temp = rnp.tree2array(f.Get(treename), branches=branches)
    MVA_array = []
    for i in range(len(MVA_temp)):
        event = []
        for j in range(len(MVA_temp[i])):
            event.append(MVA_temp[i][j])
        MVA_array.append(event)
    MVA_array = np.array(MVA_array)
    return MVA_array

def getBDTdict(f):
    bdt_results = {}
    bdt_results["pred_train"] = rnp.tree2array(f.Get("bdt_results_train"), "pred_train")
    bdt_results["pred_test"] = rnp.tree2array(f.Get("bdt_results_test"), "pred_test")
    return bdt_results

def getANNdatasetdict(f):
    ann_dataset = {}
    branches = []
    for i in range(0, 11):
        branches.append("bin"+str(i))
    
    for treename in ["ann_dataset_train", "ann_dataset_test"]:
        ann_array = []
        ann_temp = rnp.tree2array(f.Get(treename), branches=branches)
        for i in range(len(ann_temp)):
            event = []
            for j in range(len(ann_temp[i])):
                event.append(ann_temp[i][j])
            ann_array.append(event)
        if(treename == "ann_dataset_train"):
            ann_dataset["Y_train"] = np.array(ann_array)
        else:
            ann_dataset["Y_test"] = np.array(ann_array)
    return ann_dataset

#Lamb list must be doubles  0.0, 2.0, 10.0, etc.
def getANNresultslib(f, lamblist):
    ANN_masterdict = {}
    for lamb in lamblist:
        ann_results = {}
        if lamb == 0.0: lamb = int(lamb)       
        ann_results["pred_train"] = rnp.tree2array(f.Get("ann_results_pred_train_lamb"+str(lamb)), "pred_train")        
        ann_results["pred_test"] = rnp.tree2array(f.Get("ann_results_pred_test_lamb"+str(lamb)), "pred_test")
        ANN_masterdict["lamb"+str(lamb)] = ann_results
        print("key " + "lamb"+str(lamb) + " added to ANN master dictionary")
    return ANN_masterdict

# ---------------------------------------------------------------------------------------------
# General recovery function
# ---------------------------------------------------------------------------------------------
def recover_from_ROOT_file(filename, MVA_branches, lamblist):
    f = TFile(filename)
    MVA_train_array = getMVAarray(f, "MVA_train_array", MVA_branches)
    MVA_test_array = getMVAarray(f, "MVA_test_array", MVA_branches)
    BDT_results = getBDTdict(f)
    ann_dataset = getANNdatasetdict(f)
    ANN_masterdict = getANNresultslib(f, lamblist)
    return MVA_train_array, MVA_test_array, BDT_results, ann_dataset, ANN_masterdict

# ---------------------------------------------------------------------------------------------
# General run recreation function (recreates all datasets and results for plotting of results)
# ---------------------------------------------------------------------------------------------
def recreate_run(filename, MVA_branches, lamblist):
    MVA_train_array, MVA_test_array, bdt_results, ann_dataset_Y, ANN_masterdict = recover_from_ROOT_file(filename, MVA_branches, lamblist)
    
    # ----
    # BDT
    # ----
    # Recreate bdt_dataset
    bdt_dataset = {}
    bdt_dataset["X_train"]       = MVA_train_array[:, 3:]
    bdt_dataset["X_test"]        = MVA_test_array[:, 3:]
    bdt_dataset["Y_train"]       = MVA_train_array[:, 0]
    bdt_dataset["Y_test"]        = MVA_test_array[:, 0]
    bdt_dataset["weights_train"] = MVA_train_array[:, 1]
    bdt_dataset["weights_test"]  = MVA_test_array[:, 1]
    
    
    # ----
    # ANN
    # ----
    # Recreate ann_dataset
    ann_dataset = {}
    ann_dataset["X_train"]       = MVA_train_array[:, 3:]
    ann_dataset["X_test"]        = MVA_test_array[:, 3:]
    ann_dataset["Y_train"]       = ann_dataset_Y["Y_train"]
    ann_dataset["Y_test"]        = ann_dataset_Y["Y_test"]
    ann_dataset["weights_train"] = MVA_train_array[:, 1]
    ann_dataset["weights_test"]  = MVA_test_array[:, 1]
    
    
    return MVA_train_array, MVA_test_array, bdt_dataset, bdt_results, ann_dataset, ANN_masterdict

# ---------------------------------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------------------------------
def overlayed_fig5(MVA_train_array, MVA_test_array, results, ANN=True, lamb=-1):
    train_sig = results["pred_train"][MVA_train_array[:,0] == 1]
    train_bkg = results["pred_train"][MVA_train_array[:,0] == 0]
    test_sig  = results["pred_test"][MVA_test_array[:,0] == 1]
    test_bkg  = results["pred_test"][MVA_test_array[:,0] == 0]
    
    plt.hist(train_sig, density=True, color="orange", histtype="step", label="train_sig", bins=50)
    plt.hist(train_bkg, density=True, color="lightblue", histtype="step", label="train_bkg", bins=50)
    plt.hist(test_sig, density=True, color="red", histtype="step", label="test_sig", bins=50)
    plt.hist(test_bkg, density=True, color="navy", histtype="step", label="test_bkg", bins=50)

    plt.legend(fancybox=True)
    plt.xlabel("Classifier Score") if ANN else plt.xlabel("BDT Score")
    plt.grid()
    plt.minorticks_on()
    if ANN: 
        plt.xlim(0, 1)
        if lamb >= 0: plt.title("lamb = " + str(lamb))
        plt.savefig("fig5_ann_lambda_{!s}.png".format(str(lamb)))
    else:
        plt.title("BDT")
        plt.savefig("fig5_bdt.png")
    plt.close()


def save_ANN_dataset(filename, treename, ann_dataset, train):
    name = ""
    if train:
        name = "Y_train"
    else:
        name = "Y_test"

    f = TFile(filename, "update")
    t = TTree(treename, "ann_dataset")

    # Fill variables
    fill_vars = []
    for i in range(ann_dataset[name].shape[1]):
        fill_vars.append(np.zeros(1, dtype=np.float64))

    # Create all branches
    for j in range(ann_dataset[name].shape[1]):
        t.Branch("bin"+str(j), fill_vars[j], "bin"+str(j)+"/D")

    # Fill the tree
    # Outer loop: over all events (inputs)
    for i in range(ann_dataset[name].shape[0]):
        # Inner loop: over all input vars
        for j in range(len(fill_vars)):
            fill_vars[j][0] = (ann_dataset[name])[i,j]
        t.Fill()
    
    # Write and close file
    f.Write()
    f.Close()

# Save pred_train and pred_test
def save_BDT_predictions(filename, bdt_results, train=False, newbdt=False):
    name = ""
    if train: 
        name = "train"
    else:
        name = "test"

    f = TFile(filename, "update")

    bdt_version = ""
    if newbdt: 
        bdt_version = "new"

    t = TTree(bdt_version+"bdt_results_"+name, "BDT predictions")

    # Fill variables
    pred_f = np.zeros(1, dtype=np.float64)

    # Create all branches
    t.Branch("pred_"+name, pred_f, "pred_"+name+"/D")

    # Fill the tree
    for i in range(len(bdt_results["pred_"+name])):
        pred_f[0] = bdt_results["pred_"+name][i]
        t.Fill()
        
    # Write and close file
    f.Write()
    f.Close()

def save_MVA_array(filename, treename, MVA_array, bnames):
    f = TFile(filename, "update")
    t = TTree(treename, "MVA_array")

    # Fill variables in order of MVA_test_array inputs (see bnames for list)
    fill_vars = []
    for i in range(MVA_array.shape[1]):
        fill_vars.append(np.zeros(1, dtype=np.float64))

    # Create all branches
    for i in range(MVA_array.shape[1]):
        t.Branch(bnames[i], fill_vars[i], bnames[i]+"/D")

    # Fill the tree
    # Outer loop: over all events (inputs)
    for i in range(MVA_array.shape[0]):
        # Inner loop: over all input vars (label, eventWeight, etc.)
        for j in range(len(fill_vars)):
            fill_vars[j][0] = np.array(MVA_array[i, j])
        t.Fill()
    
    # Write and close file
    f.Write()
    f.Close()

def save_BDT_sideband_predictions(filename, bdt_results, train):
        name = ""
        if train: 
            name = "train"
        else:
            name = "test"

        f = TFile(filename, "update")
        t = TTree("bdt_sideband_results_"+name, "BDT predictions")

        # Fill variables
        pred_f = np.zeros(1, dtype=np.float64)

        # Create all branches
        t.Branch("pred_"+name, pred_f, "pred_"+name+"/D")

        # Fill the tree
        for i in range(len(bdt_results["pred_"+name])):
            pred_f[0] = bdt_results["pred_"+name][i]
            t.Fill()
            
        # Write and close file
        f.Write()
        f.Close()

def save_ANN_predictions(filename, treename, ann_results, train):
    name = ""
    if train:
        name = "pred_train"
    else:
        name = "pred_test"

    f = TFile(filename, "update")
    t = TTree(treename, "ann_dataset")

    # Fill variables
    print("\n\n\n ann_results[name] SHAPE: ", ann_results[name].shape)
    pred_f = np.zeros(1, dtype=np.float64)

    # Create branches
    t.Branch(name, pred_f, name+"/D")

    # Fill the tree
    # Outer loop: over all events (inputs)
    for i in range(ann_results[name].shape[0]):
        pred_f[0] = ann_results[name][i]
        t.Fill()
    
    # Write and close file
    f.Write()
    f.Close()

# BACKGROUND DOWNSAMPLING
# function: downsample_bkg()
# args:
#   MVA_train_array is (n_training_examples, len(features+label+eventweight+mbb)) 2D matrix
#   k is the final ratio of bkg:sig (k:1).
def downsample_bkg(MVA_train_array, k=1):
    # Set a k-value
    # where the resulting train array will have a k:1 bkg:sig ratio
    # Try downsampling:
    i_bkg = np.where(MVA_train_array[:, 0] == 0)[0]
    i_sig = np.where(MVA_train_array[:, 0] == 1)[0]

    # Print sample counts
    num_bkg = len(i_bkg)
    num_sig = len(i_sig)
    print("bkg samples: ", num_bkg)
    print("sig samples: ", num_sig)
    print("bkg/sig = ", num_bkg/num_sig)

    # Randomly sample from bkg len(i_sig) times without replacement
    i_bkg_downsampled = np.random.choice(i_bkg, size=k*num_sig, replace=False)
    print("bkg_downsampled/sig = ", len(i_bkg_downsampled)/num_sig)

    # Join our new train set together
    new_train_array = np.vstack((MVA_train_array[i_bkg_downsampled, :], MVA_train_array[i_sig,:]))

    # Reshuffle
    train_index_perm = np.random.permutation( np.array(range(new_train_array.shape[0])) )
    new_train_array  = new_train_array[train_index_perm,:]
    return new_train_array

def upsample_sig(MVA_train_array):
    i_bkg = np.where(MVA_train_array[:, 0] == 0)[0]
    i_sig = np.where(MVA_train_array[:, 0] == 1)[0]

    # Print sample counts
    num_bkg = len(i_bkg)
    num_sig = len(i_sig)
    print("bkg samples: ", num_bkg)
    print("sig samples: ", num_sig)
    print("bkg/sig = ", num_bkg/num_sig)

    # Randomly sample from sig len(i_bkg) times WITH replacement
    i_sig_upsampled = np.random.choice(i_sig, size=num_bkg, replace=True)
    print("sig_upsampled/bkg = ", len(i_sig_upsampled)/num_bkg)

    # Join our new train set together
    new_train_array = np.vstack((MVA_train_array[i_bkg, :], MVA_train_array[i_sig_upsampled,:]))

    # Reshuffle
    train_index_perm = np.random.permutation( np.array(range(new_train_array.shape[0])) )
    new_train_array  = new_train_array[train_index_perm,:]
    return new_train_array

####################
# Pearson's r work #
####################

def fisherTransform(r):
    return np.arctanh(r)

# Assumptions: All sample pairs are iid and follow bivariate normal dist.
# r = same correlation, n = sample size, alpha = test size
def getPearsonCI(r, n, alpha):
    # For a confidence interval of 95%, alpha = 0.05
    # 100(1-alpha)%CI : rho \in [tanh(arctanh(r) - z_{alpha/2}SE,
    #                            tanh(arctanh(r) + z_{alpha/2}SE)]
    # Now: z_{alpha/2} = invNorm(alpha/2)
    z = np.abs(norm.ppf(alpha/2))

    # Standard error in the Fisher space
    SE = 1/np.sqrt(n - 3)

    # Compute upper and lower bound:
    lower = np.tanh(np.arctanh(r) - z*SE)
    upper = np.tanh(np.arctanh(r) + z*SE)

    return (lower, upper)


# Want 95% and 99% confidence intervals: percentile_list = [95, 99]
def publishPearson(r, n, percentile_list, ANN=True, sideband=False):
    # Write to file
    if ANN==True:
        if sideband:
            file = open("pearsonCONF_ANN_SIDEBAND_lambda{!s}.txt".format(str(lamb)), "w+")
        else:
            file = open("pearsonCONF_ANN_lambda{!s}.txt".format(str(lamb)), "w+")
    else:
        if sideband:
            file = open("pearsonCONF_BDT_SIDEBAND.txt", "w+")
        else:
            file = open("pearsonCONF_BDT.txt", "w+")

    file.write("|=== Confidence Intervals for Pearson's r ===|\n")
    file.write("Sample pearson r: " + str(r))
    file.write(", n = " + str(n) + "\n")
    file.write("Confidence intervals:\n")

    for percentile in percentile_list:
        alpha = (1-percentile)/100
        CI = getPearsonCI(r, n, alpha)
        file.write(str(percentile) + "% CI: " + str(CI) + "\n")

def scatterMassScore(MVA_array, results, lamb=-9999):
    mbb = MVA_array[:, 2]
    score = results["pred_test"]
    plt.scatter(mbb, score)
    plt.ylabel("score")
    plt.xlabel("mbb")
    plt.grid()
    if lamb != -9999:
        plt.title("ANN on test, lambda = " + str(lamb) + ", n = " + str(MVA_array.shape[0]) + ")")
        plt.savefig("scatter_ANN_mass_score_lambda{!s}.png".format(str(lamb)))
    else:
        plt.title("BDT on test, (n = " + str(MVA_array.shape[0]) + ")")
        plt.savefig("scatter_BDT_mass_score.png")
    plt.close()

def getPearsonDist(model, dataset, results, MVA_test_array, lamb=-9999, parts=10, ANN=True, sideband=False):
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']
    overall_r   =  round(pearsonr(results["pred_test"][MVA_test_array[:,0]==0],  MVA_test_array[:, 2][MVA_test_array[:,0]==0])[0], 5)

    # n=parts measurements and get spread for test:
    X_val = np.array_split(X_test, parts)
    Y_val = np.array_split(Y_test, parts)
    MVA_val = np.array(np.array_split(MVA_test_array, parts))

    print(MVA_val.shape)
    print(MVA_val[0].shape)

    pearson_list = []
    for i in range(0, parts):
        if ANN==True:
            pred_val = model.predict(X_val[i])[:,0]
        else:
            pred_val = model.decision_function(X_val[i])
        test_mass_score_corr   =  round(pearsonr(pred_val[MVA_val[i][:,0]==0],  MVA_val[i][:, 2][MVA_val[i][:,0]==0])[0], 5)
        pearson_list.append(test_mass_score_corr)
    pearson_avg = np.average(pearson_list)
    pearson_std = np.std(pearson_list)
    
    # Write to file
    if ANN==True:
        if sideband:
            file = open("pearsonlist_ANN_SIDEBAND_lambda{!s}.txt".format(str(lamb)), "w+")
        else:
            file = open("pearsonlist_ANN_lambda{!s}.txt".format(str(lamb)), "w+")
    else:
        if sideband:
            file = open("pearsonlist_BDT_SIDEBAND.txt", "w+")
        else:
            file = open("pearsonlist_BDT.txt", "w+")
    file.write("=== "+str(parts)+"-fold Pearson's r ===\n" )
    file.write("Overall Pearson's r = " + str(overall_r) + "\n")
    file.write("Pearson's r list for the folds: " + str(pearson_list) + "\n")
    file.write("Pearson's r average = " + str(pearson_avg) + "\n")
    file.write("Pearson's r std_dev = " + str(pearson_std) + "\n")
    file.write("Fold size = " + str(X_val[i].shape))
    file.close()

    # Distribution of pearson's r values
    label = "Pearson's r (" + str(parts) + "-fold x-val): " + str(pearson_avg) + " +/- " + str(pearson_std)
    plt.hist(np.array(pearson_list), bins=10, alpha=0.7, label=label)
    if ANN:
        plt.title("lamb = " + str(lamb) + " Pearson's r distribution on x-val")
        plt.savefig("pearsonr_ann_lambda{!s}.png".format(str(lamb)))
    else:
        plt.title("BDT Pearson's r distribution on x-val")
        plt.savefig("pearsonr_BDT.png")
    plt.close()

    