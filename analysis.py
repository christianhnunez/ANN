import ROOT
import root_numpy as rnp
from keras.layers.core import Dense, Dropout, Activation
from HistoLib import HistoTool, PhysicsProcess, Cut, Make1DPlots, DrawPlotsWithCutsForEachSample, mBB_Binning_long
from rootpy.io import root_open
import AtlasStyle as Atlas
import numpy as np
from array import array
from copy import deepcopy


## declare event variables of interests
var_bb  = ["mBB", "pTBB", "eventWeight"]
var_ANN = ["mJJ", "pTJJ", "cosTheta_boost", "mindRJ1_Ex", "mindRJ2_Ex", "max_J1J2",
           "eta_J_star",  "QGTagger_NTracksJ1", "QGTagger_NTracksJ2", "deltaMJJ",
           "pT_balance"]
var_all = var_bb+var_ANN


## create physics process
SAMPLES = {}
SAMPLES["sherpa"] = PhysicsProcess("Sherpa Multi-b", "/eos/atlas/user/m/maklein/vbfhbb_mvainputs/cen2/tree_2central_sherpamj_de.root")
SAMPLES["vbf"]    = PhysicsProcess("VBF H->bb",      "/eos/atlas/user/m/maklein/vbfhbb_mvainputs/cen2/tree_2central_vbf_de.root")
#SAMPLES["data"]   = PhysicsProcess("data",           "/eos/atlas/user/m/maklein/vbfhbb_mvainputs/cen2/tree_2central_201718.root")

## load all varibles
for s in SAMPLES:
    for v in var_all:
        SAMPLES[s].AddEventVarFromTree(v)

## make data event weight all 1
#SAMPLES["data"].var["eventWeight"] = np.array([1]*len(SAMPLES["data"].var["eventWeight"]))

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
pTCut = 120
for s in SAMPLES:
    SAMPLES[s].var["eventWeight"] = (SAMPLES[s].var["pTBB"]>pTCut) * SAMPLES[s].var["eventWeight"]

## blind data
'''
SAMPLES["data"].var["eventWeight"] = np.logical_or( SAMPLES["data"].var["mBB"]<100, SAMPLES["data"].var["mBB"]>140) * SAMPLES[s].var["eventWeight"]
'''


## example of make histograms
'''
outfile       = ROOT.TFile("output.root", "recreate")
outfile.cd()
DrawPlotsWithCutsForEachSample(DrawTool, SAMPLES,
                               "mBB", "Mbb_pTbbCheck", xLabel="M(bb)", yLabel="Event", 
                               cuts=pTBBcuts, norm=False)
'''

## stack sample variables
MVAInputs = {}
for s in ["sherpa", "vbf"]:
    mva_arr = None
    # first variable is the label
    if s == "sherpa":
        mva_arr = np.array([0]*len(SAMPLES[s].var["eventWeight"]))
    if s == "vbf":
        mva_arr = np.array([1]*len(SAMPLES[s].var["eventWeight"]))

    mva_arr = np.vstack( (mva_arr, SAMPLES[s].var["eventWeight"])) ## second variable is the evtweight
    mva_arr = np.vstack( (mva_arr, SAMPLES[s].var["mBB"]) ) ## third variable is mbb

    for var in var_ANN:
        mva_arr = np.vstack( (mva_arr, SAMPLES[s].var[var]) ) ## then input features
    MVAInputs[s] = mva_arr

## prepare MVA data sets
train_frac      = 0.5
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

MVA_dataset = {}
MVA_dataset["X_train"]       = MVA_train_array[:, 3:]
MVA_dataset["X_test"]        = MVA_test_array[:, 3:]
MVA_dataset["Y_train"]       = MVA_train_array[:, 0]
MVA_dataset["Y_test"]        = MVA_test_array[:, 0]
MVA_dataset["weights_train"] = MVA_train_array[:, 1]

