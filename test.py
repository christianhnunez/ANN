from ROOT import TFile
import root_numpy as rnp
from keras.layers.core import Dense, Dropout, Activation

def loadDataFromeRootFile(f_name, tree_name, varname):


    t_file = TFile(f_name, "r")
    t_tree = t_file.Get(tree_name)

    stop    = 3000 ## n events to be loaded                                                                                                                                       
    #var_array = rnp.tree2array(t_tree, varname, stop=stop) ## if loading partial data use stop option                                                                            
    var_array = rnp.tree2array(t_tree, varname) ## if loading whole tree get rid off stop option                                                                                  

    return var_array

arr = loadDataFromeRootFile("/eos/atlas/user/m/maklein/vbfhbb_mvainputs/cen2/tree_2central_sherpamj_de.root", "Nominal", "mJJ")


print ( arr )
print ( Dense(16) )
