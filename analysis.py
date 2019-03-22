import ROOT
import root_numpy as rnp
from keras.layers.core import Dense, Dropout, Activation
from HistoLib import HistoTool, PhysicsProcess, Cut, Make1DPlots, DrawPlotsWithCutsForEachSample, mBB_Binning_long
from rootpy.io import root_open
import AtlasStyle as Atlas
from math import sqrt, log
import numpy as np
from array import array
from copy import deepcopy


## declare event variables of interests
var_bb  = ["mBB", "pTBB", "eventWeight"]
var_ANN = ["mJJ", "pTJJ", "cosTheta_boost", "mindRJ1_Ex", "mindRJ2_Ex", "max_J1J2",
           "eta_J_star",  "QGTagger_NTracksJ1", "QGTagger_NTracksJ2", "deltaMJJ",
           "pT_balance", ]
var_all = var_bb+var_ANN


## create physics process
SAMPLES = {}
SAMPLES["sherpa"] = PhysicsProcess("Sherpa Multi-b", "/eos/atlas/user/m/maklein/vbfhbb_mvainputs/cen2/tree_2central_sherpamj_de.root")
SAMPLES["vbf"]    = PhysicsProcess("VBF H->bb",      "/eos/atlas/user/m/maklein/vbfhbb_mvainputs/cen2/tree_2central_vbf_de.root")
SAMPLES["data"]   = PhysicsProcess("data",           "/eos/atlas/user/m/maklein/vbfhbb_mvainputs/cen2/tree_2central_201718.root")

## load all varibles
for s in SAMPLES:
    for v in var_all:
        SAMPLES[s].AddEventVarFromTree(v)

## make data event weight all 1
SAMPLES["data"].var["eventWeight"] = np.array([1]*len(SAMPLES["data"].var["eventWeight"]))

## make pTbb cuts
for s in SAMPLES:
    SAMPLES[s].AddCut("pTBB>70", SAMPLES[s].var["pTBB"]>70)
    SAMPLES[s].AddCut("pTBB>80", SAMPLES[s].var["pTBB"]>80)
    SAMPLES[s].AddCut("pTBB>90", SAMPLES[s].var["pTBB"]>90)
    SAMPLES[s].AddCut("pTBB>100", SAMPLES[s].var["pTBB"]>100)
    SAMPLES[s].AddCut("pTBB>110", SAMPLES[s].var["pTBB"]>110)
    SAMPLES[s].AddCut("pTBB>120", SAMPLES[s].var["pTBB"]>120)


pTBBcuts = ["pTBB>70", "pTBB>80", "pTBB>90", "pTBB>100", "pTBB>110", "pTBB>120"]
Make1DPlots(SAMPLES, "mBB", "mBB", "eventWeight", pTBBcuts, mBB_Binning_long)


## initiate drawtool 
DrawTool = HistoTool()
DrawTool.sqrtS = "13"

## make histograms
outfile       = ROOT.TFile("output.root", "recreate")
outfile.cd()
DrawPlotsWithCutsForEachSample(DrawTool, SAMPLES,
                               "mBB", "Mbb_pTbbCheck", xLabel="M(bb)", yLabel="Event", 
                               cuts=pTBBcuts, norm=False)
