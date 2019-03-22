import ROOT
import root_numpy as rnp
from rootpy.io import root_open
import AtlasStyle as Atlas
from math import sqrt, log
import numpy as np
from array import array
from copy import deepcopy


mBB_Binning_long = array('d',range(50, 300, 10))

def Make1DPlots(samples, histname, var, weight, cuts, binning):
    for s in samples:
        samples[s].Add1DHist(histname, binning, NoCut)
        samples[s].FillHist (histname, samples[s].var[var], samples[s].var[weight])

        for cut in cuts:
            samples[s].Add1DHist( histname, binning, samples[s].cut[cut])
            samples[s].FillHist ( histname, samples[s].var[var], samples[s].var[weight], samples[s].cut[cut])


def DrawPlotsWithCutsForEachSample(DrawTool, 
                                   samples, var, canvname, 
                                   xLabel, yLabel, 
                                   cuts, norm=False):

    for s in samples:
        plots = []
        labels = []

        plots.append(samples[s].histograms[var])
        labels.append("pre-selection")

        #DrawTool.DrawHists(var+"_"+s+"_"+canvname+"presel",  [xLabel, yLabel], [samples[s].histograms[var]], ["pre-selection"])

        samples[s].histograms[var].Write()

        for cut in cuts:
            if norm:
                samples[s].Norm1DHist(var+"_"+cut)
                plots.append(samples[s].histograms[var+"_"+cut+"_Normed"])
            else:
                plots.append(samples[s].histograms[var+"_"+cut])
                samples[s].histograms[var+"_"+cut].Write()
            labels.append(s+"_"+cut)

        DrawTool.DrawHists(var+"_"+s+"_"+canvname,  [xLabel, yLabel], plots, ["pre-selection"]+cuts)



class Cut:
    def __init__(self, name, array):
        self.name = name
        self.array = array

    def __add__(self, cut2):
        return Cut( self.name+"_"+cut2.name, np.logical_and(self.array, cut2.array) )

NoCut = Cut("NoCut", np.array([1]))

def NormHists(hists):
    newlist = []
    for i in range(len(hists)):
        if hists[i] != None and hists[i].Integral() !=0 :
            newhist = deepcopy(hists[i])
            newhist.Scale(1/newhist.Integral())
            newlist.append(newhist)
        else:
            newlist.append(None)
    return newlist


class HistoTool:

    def __init__(self, drawratio = "Data / MC", studytype = "Internal", lumi = "20.3", sqrtS = "8", doAtlasLabel = True, doLabel = True, doRescale = True, doLogX = False, doLogY = False, doLogZ = False):
        self.doLabel = doLabel
        self.doAtlasLabel = doAtlasLabel
        self.AtlasLabelPos = 0.2
        self.studytype = studytype
        self.lumi = lumi
        self.sqrtS = sqrtS
        self.doRescale = doRescale
        self.doLogX = doLogX
        self.doLogY = doLogY
        self.doLogZ = doLogZ
        self.texts = []
        self.legend = [0.7, 0.75, 0.93, 0.93]
        self.colorlist = [2, 4, 8, 28, 51, 93, 30, 38, 41, 42, 46]
        #self.colorlist = colorind
        #self.origcolorlist = colorind
        self.colorlistseq = range(500)
        self.drawOption = ""
        self.FillStyle = 3008
        self.CompareData = False
        self.MCscale = 1.0
        self.DrawScale = 1.0
        self.DrawRatio = drawratio
        self.doPrintPlots = False
        self.OutPlotDir = ""
        self.shift = 0.13
        self.doDiff = False
        self.draw2Dtext = False

    def SetDrawOption(self, option):
        self.drawOption = option

    def SetMCscale(self, scale):
        self.MCscale = scale

    def SetDrawScale(self, scale):
        self.DrawScale = scale

    def SetFillStyle(self, option):
        self.FillStyle = option

    def SetCompData(self, do):
        self.CompareData = do

    def ClearDrawOption(self):
        self.drawOption = ""

    def SetColors(self, colorlist):
        self.colorlist = colorlist

    def ReSetColors(self):
        self.colorlist = self.origcolorlist

    def SetLumi(self, lumi):
        self.lumi = lumi

    def SetSqrtS(self, SqrtS):
        self.sqrtS = SqrtS

    def AddTexts(self, x, y, color=1, size =0.08, text=""):
        self.texts.append([x, y, color, size, text])

    def ClearTexts(self):
        self.texts = []

    def DoLogX(self):
        self.doLogX = True

    def UndoLogX(self):
        self.doLogX = False

    def DoRescale(self):
        self.doRescale = True

    def UndoRescale(self):
        self.doRescale = False

    def DoLogY(self):
        self.doLogY = True

    def UndoLogY(self):
        self.doLogY = False

    def DoLogZ(self):
        self.doLogZ = True

    def UndoLogZ(self):
        self.doLogZ = False
    
    def DoPrintPlots(self):
        self.doPrintPlots = True

    def UndoPrintPlots(self):
        self.doPrintPlots = False

    def SetOutPlotDir(self, Dir):
        self.OutPlotDir = Dir

    def getROC( self, signal, background, label, cut_start=None, cut_end=None):
        
        ROCList = []
        for ivar in range(len(signal)):
            s_sort = np.sort( signal[ivar] )
            b_sort = np.sort( background[ivar] )

            print (s_sort, b_sort)

            for i in range(s_sort.shape[0]):
                if s_sort[i] == float("Inf"):
                    s_sort[i] = 100000
                if s_sort[i] == float("-Inf"):
                    s_sort[i] = -1000000

            for i in range(b_sort.shape[0]):
                if b_sort[i] == float("Inf"):
                    b_sort[i] = 100000
                if b_sort[i] == float("-Inf"):
                    b_sort[i] = -1000000

            c_start=np.min( (s_sort[0], b_sort[0]) )
            c_end=  np.max( (s_sort[len(s_sort)-1], b_sort[len(b_sort)-1]) )

            if c_start==-float('inf'):
                c_start = -2*c_end

            print (label[ivar], "min(", s_sort[0],  b_sort[0],  ")=", c_start)
            print (label[ivar], "max(", s_sort[-1], b_sort[-1], ")=", c_end)

            s_eff=[]
            b_rej=[]

            n_points = 1000
            c_delta = (1.0*c_end - 1.0*c_start) / (1.0*n_points)
            for i in range(1000):
                cut = c_start + i*1.0*c_delta
                s_eff.append( 1.0*np.count_nonzero( s_sort > cut ) / (1.0*len(s_sort))  )
                b_count = np.count_nonzero( b_sort > cut )
                b_rej.append(  (1.0*len(b_sort)) / (1.0 if b_count==0 else (1.0*b_count))  )

            ROC = ROOT.TGraph(n_points, array('d', s_eff), array('d', b_rej))
            ROC.SetName("ROC_%i" % (ivar))

            ROCList.append(ROC)

        canvas = ROOT.TCanvas("ROC_Overlay", "ROC_Overlay", 800, 600)
        canvas.cd()
        mg = ROOT.TMultiGraph()
        legend = ROOT.TLegend(0.5, 0.5, 0.75, 0.75)
        for i in range(len(ROCList)):
            ROC = ROCList[i]
            ROC.SetLineWidth(2)
            ROC.SetLineColor(self.colorlist[i])
            ROC.SetMarkerColor(self.colorlist[i])
            mg.Add(ROC)
            legend.AddEntry(ROC, label[i], "lp")

        mg.Draw("AL")
        mg.GetXaxis().SetTitle("signal efficiency")
        mg.GetYaxis().SetTitle("background rejection")
        legend.Draw()
        canvas.Write()
        canvas.Close()


    def DrawHists(self, title, axisname=[], inplots =[], inlabel=[], instacks=[], instacklabel = [], sys = []):
        maxval =1
        minval =1000
        secminval = 10000
        legend = ROOT.TLegend(self.legend[0], self.legend[1], self.legend[2], self.legend[3])
        legend.SetFillColor(0)
        doExist = True


        ## finding the right axises space         
        for i in range(len(inplots)):
            if inplots[i] == None:
                continue
            doExist = False

            if ("TH2D" in inplots[i].ClassName()):
                continue
                
            if ("TH" not in inplots[i].ClassName()):
                legend.AddEntry(inplots[i], inlabel[i], "LPS")
                continue

            inplots[i] = deepcopy(inplots[i])
            legend.AddEntry(inplots[i], inlabel[i], "LPS")
            thismax = inplots[i].GetMaximum()
            thismin = inplots[i].GetMinimum()

            if maxval< thismax:
                maxval = thismax
            if (minval >= thismin):
                minval = thismin

            inplots[i].GetYaxis().SetTitleOffset( inplots[i].GetYaxis().GetTitleOffset()*1.1)

        for i in range(len(instacks)):
            ## scale the MC
            if instacks[i] == None:
                continue

            doExist = False

            instacks[i] = deepcopy(instacks[i])
            if ((i != len(instacks)-1) and self.CompareData):
                instacks[i].Scale(self.DrawScale)
                legend.AddEntry(instacks[i], instacklabel[i], 'f')

            if ((i == len(instacks)-1) and self.CompareData):
                legend.AddEntry(instacks[i], instacklabel[i])

            instacks[i].GetYaxis().SetTitleOffset( instacks[i].GetYaxis().GetTitleOffset()*1.1)

            thismax = instacks[i].GetMaximum()
            thismin = instacks[i].GetMinimum()

            if maxval< thismax:
                maxval = thismax
            if (minval >= thismin and thismin != 0):
                minval = thismin

        if doExist:
            return 

        if minval <= 1.0:
            minval = 1.0

        ###### draw histogram
        Canv = ROOT.TCanvas('Canv_' + title, 'Canv_' + title, 0, 0, 800, 600)
        if(self.CompareData):
            Pad1 = ROOT.TPad('Pad1', 'Pad1', 0.0, 0.25, 1.0, 0.99, 0)
            Pad2 = ROOT.TPad('Pad2', 'Pad2', 0.0, 0.00, 1.0, 0.32, 0)
            Pad2.SetBottomMargin(0.4)
            Pad1.Draw()
            Pad2.Draw();
            Pad1.cd()

        ncolor =0

        for i in range(len(instacks)):
            if instacks[i]==None:
                ncolor+=1            
                continue
            instacks[i].SetMarkerColor(self.colorlist[ncolor])
            instacks[i].SetFillStyle(self.FillStyle)
            instacks[i].SetLineColor(self.colorlist[ncolor])
            instacks[i].SetLineColor(1)
            instacks[i].SetLineWidth(1)
            instacks[i].GetXaxis().SetTitle(axisname[0])
            instacks[i].GetYaxis().SetTitle(axisname[1])

            if (self.doRescale and not(self.doLogY)):
                instacks[i].GetYaxis().SetRangeUser(0, maxval*4.5/3.)
            if (self.doRescale and self.doLogY):
                instacks[i].GetYaxis().SetRangeUser(minval/100., maxval*10.)
            ncolor+=1            

        for i in range(len(inplots)):
            if self.CompareData:
                XaxisTitle = inplots[i].GetXaxis().GetTitle()
                labelsize = inplots[i].GetXaxis().GetLabelSize()
                inplots[i].SetTitle("")
                inplots[i].GetXaxis().SetLabelSize(0)
                inplots[i].GetYaxis().SetTitle(axisname[1])

                if  i == len(inplots)-1:
                    if ("TH" not in inplots[i].ClassName()):
                        inplots[i].Draw("")
                    else:
                        inplots[i].Draw("e")
                    inplots[i].SetLineColor(1)
                    inplots[i].SetMarkerColor(1)
                    #inplots[i].SetFillColor(1)
                    continue

                if  i == len(inplots)-2:
                    if ("TH" not in inplots[i].ClassName()):
                        inplots[i].Draw("same")
                    else:
                        inplots[i].Draw("e same")
                    inplots[i].SetMarkerStyle(20)
                    inplots[i].SetMarkerColor(2)
                    inplots[i].SetLineColor(2)

                    Pad2.cd()
                    relsize = Pad2.GetAbsHNDC()/Pad1.GetAbsHNDC()
                    size = Atlas.tsize/relsize

                    Ratio = None

                    if self.doDiff:
                        Fit = None
                        if ("TH" not in inplots[i].ClassName()):
                            Ratio = deepcopy (inplots[len(inplots)-1])
                            PreBinInt = Ratio.Integral()
                            Fit = deepcopy(inplots[i])

                        else:
                            Fit = deepcopy(inplots[len(inplots)-1])
                            Ratio = deepcopy (inplots[i])
                            PreBinInt = Ratio.Integral()



                        data = deepcopy(Ratio)
                        Ratio.Add(Fit, -1)
                        Ratio.Rebin(5)
                        data.Rebin(5)
                        Ratio.Divide(data)
                        try:
                            Fit.Rebin(5)
                        except AttributeError:
                            None
                    else:

                        Ratio = deepcopy (inplots[len(inplots)-1])
                        Ratio.Divide( inplots[i])

                    Ratio.SetTitle("")
                    Ratio.GetXaxis().SetLabelSize(size)
                    Ratio.GetYaxis().SetLabelSize(size)
                    Ratio.GetXaxis().SetTitleSize(size)
                    Ratio.GetYaxis().SetTitleSize(size) 
                    Ratio.GetXaxis().SetTitleOffset( Ratio.GetXaxis().GetTitleOffset()*relsize*2.9)
                    Ratio.GetXaxis().SetLabelOffset(0.03)
                    Ratio.GetYaxis().SetTitleOffset( Ratio.GetYaxis().GetTitleOffset()*relsize)
                    Ratio.GetYaxis().SetTitle(self.DrawRatio)
                    Ratio.GetXaxis().SetTitle(axisname[0])
                    Ratio.GetYaxis().SetNdivisions(4)
                    if self.doDiff:
                        Ratio.GetYaxis().SetRangeUser(-0.2, 0.2)
                    else:
                        Ratio.GetYaxis().SetRangeUser(0.5, 1.5)
                    
                    Ratio.SetMarkerColor(1)
                    Ratio.SetLineColor(1)
                    Ratio.GetYaxis().SetNdivisions(5, ROOT.kFALSE)
                    if sys ==[]:
                        Ratio.Draw('e')
                    else:
                        Ratio.Draw('e')
                        sys[0].SetFillStyle(3004)
                        sys[0].SetFillColor(1)
                        sys[0].SetMarkerStyle(10)
                        sys[0].SetMarkerSize(0)

                        if sys[0].GetBinError(1)!= 0:
                            sys[0].Draw("e2 same")
                        if len(sys)>1:
                            sys[1].SetFillStyle(3004)
                            sys[1].SetFillColor(2)
                            sys[1].SetMarkerStyle(10)
                            sys[1].SetMarkerSize(0)
                            sys[1].SetLineColor(2)
                            sys[1].SetLineWidth(2)
                            sys[1].Draw("e2 same")

                    #if Ratio != None:
                        #line.Draw("same")
                    Pad1.cd()

                    continue
            
            if inplots[i] ==None:
                continue
            inplots[i].SetMarkerColor(self.colorlist[ncolor])
            #inplots[i].SetFillColor(self.colorlist[ncolor])
            inplots[i].SetLineColor(self.colorlist[ncolor])
            inplots[i].GetXaxis().SetTitle(axisname[0])
            inplots[i].GetYaxis().SetTitle(axisname[1])

            if (self.doRescale and not(self.doLogY)):
                inplots[i].GetYaxis().SetRangeUser(0, maxval*4.5/3.)
            if (self.doRescale and self.doLogY):
                inplots[i].GetYaxis().SetRangeUser(minval/100., maxval*10.)
            ncolor+=1            

        count = 0

        for i in range(len(inplots)):
            if inplots[i] ==None:
                continue
            inplots[i].SetTitle("")
            if count != 0:
                self.drawOption = "same"
            if inplots[i].ClassName() != "TH2D":
                inplots[i].Draw(self.drawOption)
            else:
                if self.draw2Dtext:
                    inplots[i].Draw("colz text") 
                else:
                    inplots[i].Draw("colz") 

            count +=1

        count = 0

        for i in range(len(instacks)):
            if instacks[i] == None:
                continue

            #instacks[i].GetXaxis().SetTitle("")
            if "TH2" not in (instacks[i].ClassName()):
                #instacks[i].GetYaxis().SetMaxDigits(3)
                if count == 0:
                    instacks[i].SetTitle("")
                    if self.CompareData:
                        instacks[i].GetXaxis().SetLabelSize(0)
                        instacks[i].GetXaxis().SetTitle("")
                    instacks[i].Draw("hist")
                    count +=1
                    continue
                else:
                    if self.CompareData:
                        XaxisTitle = instacks[i].GetXaxis().GetTitle()
                        labelsize = instacks[i].GetXaxis().GetLabelSize()
                        instacks[i].SetTitle("")
                        instacks[i].GetXaxis().SetLabelSize(0)

                        if  i == len(instacks)-1:
                            instacks[i].Draw("e same")
                            instacks[i].SetLineColor(1)
                            instacks[i].SetLineWidth(2)
                            instacks[i].SetMarkerColor(1)
                            instacks[i].SetFillColor(0)

                            #### pay attention ##
                            DataIntError = ROOT.Double(0)
                            DataInt = instacks[i].IntegralAndError(0, instacks[i].GetNbinsX()+1, DataIntError)
                            MCIntError  = ROOT.Double(0)
                            MCInt = instacks[i-1].IntegralAndError(0, instacks[i-1].GetNbinsX()+1, MCIntError)
                            if MCInt ==0:
                                MCInt = 1
                                print (" warning: no mc")
                            if DataInt ==0:
                                DataInt = 1
                                print (" warning: no data")
                            rat = DataInt/MCInt
                            rerr = sqrt((DataIntError/DataInt)**2+(MCIntError/MCInt)**2)*rat
                            #Atlas.myText(0.7, 0.45, 1, 0.04, "Data / MC = " + str(round(rat, 3)) +"#pm" + str(round(rerr,3)))
                            count +=1
                            continue

                        if  i == len(instacks)-2:
                            instacks[i].SetMarkerStyle(10)
                            instacks[i].SetFillStyle(3004)
                            instacks[i].SetMarkerSize(0.00001)
                            instacks[i].SetFillColor(1)
                            instacks[i].SetLineWidth(0)
                            instacks[i].Draw("e2 same")

                            Pad2.cd()
                            relsize = Pad2.GetAbsHNDC()/Pad1.GetAbsHNDC()
                            size = Atlas.tsize/relsize

                            Ratio = deepcopy (instacks[len(instacks)-1])
                            line = 0
                            if Ratio == None:
                                continue
                            if Ratio != None:
                                line = ROOT.TLine(Ratio.GetXaxis().GetBinLowEdge(1), 1, Ratio.GetXaxis().GetBinUpEdge(Ratio.GetNbinsX()), 1)
                            Ratio.Divide(instacks[i])

                            Ratio.SetTitle("")
                            Ratio.GetXaxis().SetLabelSize(size)
                            Ratio.GetYaxis().SetLabelSize(size)
                            Ratio.GetXaxis().SetTitleSize(size)
                            Ratio.GetYaxis().SetTitleSize(size) 
                            Ratio.GetXaxis().SetTitleOffset( Ratio.GetXaxis().GetTitleOffset()*relsize*2.9)
                            Ratio.GetXaxis().SetLabelOffset(0.03)
                            Ratio.GetYaxis().SetTitleOffset( Ratio.GetYaxis().GetTitleOffset()*relsize)
                            Ratio.GetYaxis().SetTitle(self.DrawRatio)
                            #Ratio.GetXaxis().SetTitle(XaxisTitle)
                            Ratio.GetXaxis().SetTitle(axisname[0])
                            Ratio.GetYaxis().SetRangeUser(0.5, 1.5)
                            Ratio.GetYaxis().SetNdivisions(4)
                            Ratio.SetMarkerColor(1)
                            Ratio.SetLineColor(1)
                            Ratio.SetLineWidth(2)
                            Ratio.GetYaxis().SetNdivisions(5, ROOT.kFALSE)
                            if sys ==[]:
                                Ratio.Draw('e')

                            elif len(sys)>1:

                                Ratio.Draw('e')

                                sys[1].SetFillStyle(3001)
                                sys[1].SetFillColor(30)
                                sys[1].SetMarkerStyle(10)
                                sys[1].SetMarkerSize(0)
                                sys[1].SetLineColor(2)
                                sys[1].SetLineWidth(2)
                                sys[1].Draw("e2 same")

                                sys[0].SetFillStyle(3004)
                                sys[0].SetFillColor(1)
                                sys[0].SetMarkerStyle(10)
                                sys[0].SetLineWidth(2)
                                sys[0].SetMarkerSize(0)
                                if sys[0].GetBinError(1)!= 0:
                                    sys[0].Draw("e2 same")
                                Ratio.Draw('e same')

                            else:
                                Ratio.Draw('e')
                                sys[0].SetFillStyle(3004)
                                sys[0].SetFillColor(1)
                                sys[0].SetMarkerStyle(10)
                                sys[0].SetLineWidth(2)
                                sys[0].SetMarkerSize(0)
                                sys[0].Draw("e2 same")


                            #if Ratio != None:
                            #    line.Draw("same")
                            Pad1.cd()
                            count +=1
                            continue
                        
                        instacks[i].Draw("hist same")


                    else:
                        if  i == len(instacks)-1:
                            instacks[i].SetMarkerStyle(10)
                            instacks[i].SetFillStyle(3020)
                            instacks[i].Draw("e2 same")
                            count +=1
                            continue
                        instacks[i].Draw("hist same")

            ##### draw 2d
            else:
                instacks[i].SetTitle("")
                instacks[i].Draw('cont') 
                count +=1

        if( (instacks != [] and instacks[-1]!=None and ("TH2" not in (instacks[-1].ClassName())) ) or (inplots !=[] and inplots[-1]!=None and ("TH2" not in (inplots[-1].ClassName()))) ):
            legend.Draw("same")

        for text in self.texts:
            Atlas.myText(text[0], text[1] ,text[2], text[3], text[4])

        if self.doAtlasLabel:
            Atlas.ATLASLabel(self.AtlasLabelPos, 0.88, self.shift, self.studytype,color=1)
        if self.doLabel and self.lumi != "0":
            Atlas.myText(self.AtlasLabelPos, 0.81 ,color=1, size=0.04,text="#sqrt{s}="+self.sqrtS + " TeV, " + self.lumi + " fb^{-1}") 
        if self.doLabel and self.lumi == "0":
            Atlas.myText(self.AtlasLabelPos, 0.81 ,color=1, size=0.04,text="#sqrt{s}="+self.sqrtS + " TeV") 

        if self.doLogY:
            Canv.SetLogy()
        if self.doLogX:
            Canv.SetLogx()
        if self.doLogZ:
            Canv.SetLogz()

        Canv.Write()
        if (self.doPrintPlots):
            Canv.SaveAs(self.OutPlotDir+Canv.GetName()+".png")
        Canv.Close()

        return inplots


class PhysicsProcess:
    def __init__(self, name, filename, sysName = ""):
        self.name = name
        self.filename = filename
        self.var = {}
        self.cut ={}
        self.histograms ={}
        self.file = None
        self.tree = None
        self.isSys = False
        if filename != None:
            self.file = root_open(filename) 
            if sysName =="":
                self.tree = self.file.Nominal
            else:
                self.tree = self.file.Get(sysName)
                self.isSys = True

    def Add1DHist(self, histname, bin, cut = NoCut):
        if cut == NoCut:
            self.histograms[histname] = ROOT.TH1D(self.name+"_"+histname , self.name+"_"+histname, len(bin)-1, bin)
        else:
            histname = histname +"_"+cut.name
            self.histograms[histname] =ROOT.TH1D(self.name+"_"+histname, self.name+"_"+histname, len(bin)-1, bin)

    def Add2DHist(self, histname, binx, biny, cut = NoCut):
        self.histograms[histname] = ROOT.TH2D(self.name+"_"+histname +"_" +cut.name, self.name+"_"+histname+"_"+cut.name, len(binx)-1, binx, len(biny), biny[0], biny[-1])

    def Norm1DHist(self, histname):
        self.histograms[histname+"_Normed"] = NormHists([self.histograms[histname]])[0]

    def FillHist(self, histname, array, weight, cut = NoCut):
        if cut == NoCut:
            rnp.fill_hist( self.histograms[histname], array, weight)
        else:
            rnp.fill_hist( self.histograms[histname+"_"+cut.name], array, weight*cut.array)

    def AddCut(self, name, array):
        self.cut[name] = Cut(name, array)

    def AddNoCut(self):
        self.cut["NoCut"] = NoCut

    def AddEventVarFromTree(self, varname, test=False):
        stop = None
        if test:
            stop = 300000

        newarray = rnp.tree2array(self.tree, varname, stop=stop)

        if varname == "pT_ballance":
            varname = "pT_balance"

        VarInMeV = ["mBB", "pTBB", "pTJJ", "mJJ", "mJ1B1", "mJ1B2", "deltaMJJ", "HT_MVA", "HT_soft", "pTB1", "pTB2", "pTJ1", "pTJ2", "pT_tot"]

        if varname in VarInMeV:
            self.var[varname] = newarray/1000.
        else:
            self.var[varname] = newarray

