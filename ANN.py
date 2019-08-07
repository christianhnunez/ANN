# Testing version control
import numpy as np
import root_numpy as rnp
from copy import deepcopy
import os
import scipy.stats as stats
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import theano
import theano.tensor as T
theano.config.optimizer='None'
theano.config.exception_verbosity='high'

import matplotlib
matplotlib.use("PS")
import matplotlib.pyplot as plt

from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten, Masking, Lambda
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2

def saveModel(fileNameBase, model):
    ###################################################
    ###  Function to save a trained keras model
    ###  Architecture saved in json, weights saved in h5
    ###  input:
    ###         fileNameBase:  str
    ###         model: keras model object
    ###################################################

    json_string = model.to_json()
    print ('base ', fileNameBase)
    open(fileNameBase+"_architecture.json", 'w').write(json_string)
    model.save_weights(fileNameBase + '_model_weights.h5', overwrite=True)


def loadModel(fileNameBase):
    ###################################################
    ###  Function to load a keras model from json and h5
    ###  input:
    ###         fileNameBase:  str
    ###         model:         keras model object
    ###################################################

    model = model_from_json(open( fileNameBase+'_architecture.json').read())
    model.load_weights(fileNameBase + '_model_weights.h5')
    return  model


def ANNStruct(inputshape, nMBBbins=16):
    ############################################################################
    ###  Function to build ANN network structure
    ###  input:
    ###         inputshape: array[int], 
    ###                     the input layer shape, should be the dimension of features
    ###  output:
    ###         ANN_Model:  keras model object (not trained, just a structure)
    ############################################################################


    ## hyper parmaeters of the network
    n_dense_classification = 256
    n_dense_adversary = 128

    input_layer = Input(inputshape)

    ## first we have the classifier part 
    ## one can change the structure to add more rectifier layers and hidden layers
    ## the output in a binary classification problem should be sigmoid
    dense_cl_batch  = BatchNormalization(name="batch_cl")(input_layer)
    dense_cl_hidden1 = Dense(n_dense_classification, activation="sigmoid", 
                            name="dense_cl_hidden1", init="he_uniform", kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3))(dense_cl_batch)
    # dense_cl_hidden2 = Dense(64, activation="sigmoid", 
    #                         name="dense_cl_hidden2", init="he_uniform", kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3))(dense_cl_hidden1)  
    # dense_cl_hidden3 = Dense(32, activation="sigmoid", 
    #                         name="dense_cl_hidden3", init="he_uniform", kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3))(dense_cl_hidden2)  
    # dense_cl_hidden4 = Dense(16, activation="sigmoid", 
    #                         name="dense_cl_hidden4", init="he_uniform", kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3))(dense_cl_hidden3)                        
    dense_cl_output = Dense(1, name="dense_cl_output", init="he_uniform")(dense_cl_hidden1)
    dense_cl_sigmoid = Activation('sigmoid', name="cl_sigmoid")(dense_cl_output) 
    
    ## then the output of classifer is fed to another network
    ## which tries to fit the Mbb (here the example is fitting Mbb bins, not the Mbb value directly)
    ## one could try directly fitting Mbb, using linear output layer and 2-norm as loss
    dense_ad_hidden = Dense(n_dense_adversary, activation='tanh', name='dense_ad_hidden', 
                            init = "he_uniform", kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3) )(dense_cl_sigmoid)
    # dense_ad_hidden1 = Dense(64, activation='tanh', name='dense_ad_hidden1', 
    #                         init = "he_uniform", kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3) )(dense_ad_hidden)
    # dense_ad_hidden2 = Dense(32, activation='tanh', name='dense_ad_hidden2', 
    #                         init = "he_uniform", kernel_regularizer=l1_l2(l1=1e-3, l2=1e-3) )(dense_ad_hidden1)
    dense_ad_output = Dense(nMBBbins, activation='softmax', name='dense_ad_output', init = "he_uniform")(dense_ad_hidden)


    # Let's check if I'm getting the right stuff:
    MergeOutput = Concatenate(axis=1)( [ dense_cl_sigmoid, dense_ad_output ] )

    ANN_Model = Model(input=input_layer, output =MergeOutput)
    return ANN_Model


def TrainANN(dataset, lamb=1.0, gam=1.0, clpretrain = 50, adpretrain = 50, 
                     epoch=50,  batch_size = 20, nMBBbins = 16, lr=1e-3):

    ############################################################################
    ###  Function to  Train ANN network 
    ###  
    ###  input:
    ###         dataset:     dictionary[arrays], input data 
    ###         lamb:        float, penalization factor
    ###         clpretrain:  int, classifier pre-train epoches
    ###         adpretrain:  int, adversary pre-train epoches
    ###         epoch:       int, network training epoches
    ###         batchsize:   int, mini-batch size
    ###         nMBBbins:    int, number of discretized Mbb bins to fit
    ############################################################################

    ## delcare training history arrays for debuging
    history = {}
    history["cl"] = []
    history["ad"] = []
    history["ann_cl"] = []
    history["ann_ad"] = []
    history["ann"] = []

    ## unload data from dictionary
    ## X are the features
    ## y are the labels
    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']
    weights_train = dataset['weights_train']
    temp_weights = weights_train
    weights_train_bkg = weights_train[Y_train[:,-1]==0]
    weights_train_sig = weights_train[Y_train[:,-1]==1]
   
    # ===== Apply Gamma hyperparameter
    # weights_train = []
    # for i in range(0, len(temp_weights)):
    #     #Only edit the signal
    #     if Y_train[i, -1] == 1:
    #         entry = temp_weights[i] * gam
    #         weights_train.append(entry)
    #     else:
    #         weights_train.append(temp_weights[i])
    # weights_train = np.array(weights_train)

    # ===== Standardize around 1:
    # sig_mean = np.mean(weights_train_sig)
    # sig_std = np.std(weights_train_sig)
    # bkg_mean = np.mean(weights_train_bkg)
    # bkg_std = np.std(weights_train_bkg)

    # weights_train = []
    # for i in range(0, len(temp_weights)):
    #     if Y_train[i, -1] == 1: #sig
    #         entry = ((temp_weights[i] - sig_mean)/sig_std) + 1
    #         weights_train.append(entry)
    #     else: # is bkg:
    #         entry = ((temp_weights[i] - bkg_mean)/bkg_std) + 1
    #         weights_train.append(temp_weights[i])
    # weights_train = np.array(weights_train)

    # sigrr = (weights_train_sig-sig_mean)/sig_std + 1
    # plt.hist(sigrr, density=True, color="orange", histtype="step", label="train_sig", bins=50)
    # bkgrr = (weights_train_bkg-bkg_mean)/bkg_std + 1
    # plt.hist(bkgrr, density=True, color="blue", histtype="step", label="train_sig", bins=50)
    # plt.grid()
    # plt.minorticks_on()
    # plt.legend()
    # plt.savefig("weights_hist.png")
    # plt.close()


    print("\n\n\n\n here we are " + str(gam))
    print(weights_train[Y_train[:, -1]==1]/temp_weights[Y_train[:, -1]==1])
    print("\n\n\n\n here we are ")

    X_train_bkg = X_train[ Y_train[:,-1]==0]
    Y_train_bkg = Y_train[ Y_train[:,-1]==0]

    X_test_bkg = X_test[ Y_test[:,-1]==0]
    Y_test_bkg = Y_test[ Y_test[:,-1]==0]


    # # ===== Artificial Augment
    # # Takes X_train_bkg and for every training example, duplicates
    # # it nMBBbins times and feeds it a faked Y_train_bkg label.
    # # This is for use with X_train_bkg because the classifier should
    # # not be fed the same example 10 times because nothing is changing
    # # for it (the label for the classifier is just 1 or 0).
    # new_X_train_bkg = []
    # new_Y_train_bkg = []
    # new_weights_train_bkg = []
    # for i in range(0, X_train_bkg.shape[0]):
    #     for n in range(0, nMBBbins):
    #         new_X_train_bkg.append(X_train_bkg[i])
    #         Y_onehot = np.zeros(Y_train[i, :].shape)
    #         Y_onehot[n] = 1
    #         #if(np.random() > 0.95): print(Y_onehot)
    #         new_Y_train_bkg.append(Y_onehot)
    #         new_weights_train_bkg.append(weights_train_bkg[i])
    # X_train_bkg = np.array(new_X_train_bkg)
    # Y_train_bkg = np.array(new_Y_train_bkg)
    # weights_train_bkg = np.array(new_weights_train_bkg)
    # print("shape shape shape!! : " + str(X_train_bkg.shape))
    # print("shape shape shape!! : " + str(X_train.shape))

    # np.random.seed(10)
    # train_index_perm = np.random.permutation( np.array(range(X_train_bkg.shape[0])) )
    # X_train_bkg  = X_train_bkg[train_index_perm,:]
    # Y_train_bkg   = Y_train_bkg[train_index_perm,:]


    ## get the network model and set training hyper parameters
    print ("building ann")
    model = ANNStruct(X_train.shape[1:], nMBBbins = nMBBbins) 
    adam = Adam(lr = lr)


    ## === load old weights:
    #model.load_weights("ANN_lambda10.0_clpretrain2_adpretrain2_epoch50_minibatch256_mBBbins10_model_weights.h5")
    #print("loaded old weights")
    ## ===

    print (model.get_config())
    print ("model summary")
    print (model.summary())


    ## three separate loss needed 
    ## (1) cl only objective
    ## (2) ad only objective
    ## (3) linked objective penalizing cl objective by lamb*ad objective

    def custom_objective_cl_only(y_true, y_pred):
        c_pred = T.clip(y_pred[:,0], 0.0+1e-4, 1.0-1e-4)
        bce = T.nnet.binary_crossentropy(c_pred, y_true[:,-1]).mean()
        loss = bce
        return loss

    def custom_objective_ad_only(y_true, y_pred):
        '''
        bkg_id = y_true==0
        y_true = y_true[bkg_id]
        y_pred = y_pred[bkg_id]
        '''

        a_pred = T.clip(y_pred[:, 1:nMBBbins+1], 0.0+1e-4, 1.0-1e-4)
        cce = T.nnet.categorical_crossentropy(a_pred, y_true[:,0:nMBBbins]).mean()
        loss = cce
        return loss

    def custom_objective(y_true, y_pred):

        c_pred = T.clip(y_pred[:,0], 0.0+1e-4, 1.0-1e-4)
        bce = T.nnet.binary_crossentropy(c_pred, y_true[:,-1]).mean()

        '''
        bkg_id = y_true==0
        y_true = y_true[bkg_id]
        y_pred = y_pred[bkg_id]
        '''

        a_pred = T.clip(y_pred[:, 1:nMBBbins+1], 0.0+1e-4, 1.0-1e-4)
        cce = T.nnet.categorical_crossentropy(a_pred, y_true[:,0:nMBBbins]).mean()
        
        loss = bce - lamb* cce
        
        return loss

    #####################################################
    ##### Step 1. Pre-Train For the classification model
    ##### (not necessarily needed, please customize)
    #####################################################

    print ('----------------- Pre-train for classification-------------------------')
    print ('----------------- fix pars-------------------------')

    ## when pre-training classifer, adversary parameters need to be frozen
    for il in range(len(model.layers)):
        if "ad" in model.layers[il].name:
            model.layers[il].trainable = False
        if "cl" in model.layers[il].name:
            model.layers[il].trainable = True
            
    ## the model is trained with cl only objective
    model.compile(loss=custom_objective_cl_only, optimizer=adam, metrics=["accuracy"])
    #cw = compute_class_weight('balanced', np.unique(Y_train[:, 0]), Y_train[:, 0])
    try:
        history["cl"] = model.fit( X_train , Y_train, batch_size=batch_size, epochs=clpretrain, validation_split=0.2, shuffle = True, sample_weight=weights_train).history['loss']
        # sample_weight=weights_train
    except KeyError:
        history["cl"] = []
    #

    #####################################################
    ##### Step 2. Pre-Train For the adversary model
    ##### (not necessarily needed, please customize)
    #####################################################

    print ('----------------- Pre-train for ad-------------------------')
    print ('----------------- fix pars-------------------------')

    ## when pre-training adversary, classifer parameters need to be frozen
    for il in range(len(model.layers)):
        if "ad" in model.layers[il].name:
            model.layers[il].trainable = True
        if "cl" in model.layers[il].name:
            model.layers[il].trainable = False

    ## (standard step 2) the model is trained with ad only objective
    model.compile(loss=custom_objective_ad_only, optimizer=adam, metrics=["accuracy"])
    # try:
    #    history["ad"] = model.fit( X_train , Y_train, batch_size=batch_size, epochs=adpretrain, validation_split=0.2, shuffle = True).history['loss']
    # except KeyError:
    #    history["ad"] = []

    # alternate Step 2 (show only background)
    try:
        history["ad"] = model.fit( X_train_bkg, Y_train_bkg, batch_size=batch_size, epochs=adpretrain, validation_split=0.2, shuffle = True, sample_weight=weights_train_bkg).history['loss']
        #sampleweights^
    except KeyError:
        history["ad"] = []


    #####################################################
    ##### Step 3. Training both cl and ad at the same time
    #####################################################
    print  ("not finite", np.sum(X_train[ ~np.isfinite(X_train)]))
    print  ("not finite", np.sum(Y_train[ ~np.isfinite(Y_train)]))

    print ("max x", np.max(X_train), np.min(X_train))
    print ("max y", np.max(Y_train), np.min(Y_train))

    for ie in range(epoch):

        for il in range(len(model.layers)):
            if "ad" in model.layers[il].name:
                model.layers[il].trainable = False
            if "cl" in model.layers[il].name:
                model.layers[il].trainable = True

        model.compile(loss=custom_objective, optimizer=adam, metrics=["accuracy"])
        # STEP C.i
        model.fit( X_train , Y_train, batch_size=batch_size, epochs=1, validation_split=0.2, 
                   shuffle = True, sample_weight=weights_train).history['loss'][0] 
        #sample_weight=weights_train
         

        for il in range(len(model.layers)):
            if "ad" in model.layers[il].name:
                model.layers[il].trainable = True
            if "cl" in model.layers[il].name:
                model.layers[il].trainable = False

        model.compile(loss=custom_objective_ad_only, optimizer=adam, metrics=["accuracy"])
        # standard step 2
        # STEP C.ii
        print("batch size for c.ii: ", X_train.shape[0])
        #model.fit( X_train , Y_train, batch_size=X_train.shape[0], epochs=1, validation_split=0.2, 
        #          shuffle = True).history['loss'][0]
        #sample_weight=weights_train 
         
        # alternate Step 2 (show only background)
        model.fit( X_train_bkg , Y_train_bkg, batch_size=batch_size, epochs=1, validation_split=0.2, 
                   shuffle = True, sample_weight=weights_train_bkg).history['loss'][0] 
        #sample_weight=weights_train_bkg

        model.compile(loss=custom_objective_cl_only, optimizer=adam, metrics=["accuracy"])
        history["ann_cl"].append( model.evaluate(X_train[0:int(X_train.shape[0]*0.8)], Y_train[0:int(Y_train.shape[0]*0.8)])[0]  )
        
        model.compile(loss=custom_objective_ad_only, optimizer=adam, metrics=["accuracy"])
        # standard step 2
        #history["ann_ad"].append( model.evaluate(X_train[0:int(X_train.shape[0]*0.8)], Y_train[0:int(Y_train.shape[0]*0.8)])[0] )
        # alternate Step 2 (show only background)
        history["ann_ad"].append( model.evaluate(X_train_bkg[0:int(X_train_bkg.shape[0]*0.8)], Y_train_bkg[0:int(Y_train_bkg.shape[0]*0.8)])[0] )

        model.compile(loss=custom_objective, optimizer=adam, metrics=["accuracy"])
        history["ann"].append( model.evaluate(X_train[0:int(X_train.shape[0]*0.8)], Y_train[0:int(Y_train.shape[0]*0.8)])[0] )

        print ("cl loss", history["ann_cl"][-1], "ad loss", history["ann_ad"][-1])

    fileNameBase = "ANN_lambda"+str(lamb)+"_clpretrain"+str(clpretrain)+"_adpretrain"+str(adpretrain)+"_epoch"+str(epoch)+"_minibatch"+str(batch_size)+"_mBBbins"+str(nMBBbins)
    saveModel(fileNameBase, model)

    return model, history


def predictANN(model, dataset):

    ############################################################################
    ###  Example function to use the trained model to give predictions
    ###  
    ###  input:
    ###         model:       keras model object
    ###         dataset:     dictionary[arrays], input data 
    ############################################################################

    X_test = dataset['X_test']
    Y_test = dataset['Y_test']
    X_train = dataset['X_train']
    Y_train = dataset['Y_train']

    pred_train = model.predict( X_train)[:,0]
    pred_test = model.predict( X_test )[:,0]

    roc_train  = roc_curve( Y_train[:, -1], pred_train)
    roc_test   = roc_curve(Y_test[:, -1],  pred_test)

    auc_train  = round(auc(roc_train[1], roc_train[0], reorder=True), 3)
    auc_test   = round(auc(roc_test[1], roc_test[0], reorder=True), 3)

    # Precision-Recall Curve
    prc_train = precision_recall_curve( Y_train[:, -1], pred_train)
    prc_test = precision_recall_curve( Y_test[:, -1], pred_test)

    auc_prc_train = round(auc(prc_train[1], prc_train[0], reorder=True), 3)
    auc_prc_test = round(auc(prc_test[1], prc_test[0], reorder=True), 3)


    results = {"pred_train":    pred_train,
               "pred_test":     pred_test, 
               "roc_train":     roc_train,
               "roc_test":      roc_test,
               "auc_train":     auc_train,
               "auc_test":      auc_test,
               "prc_train":     prc_train,
               "prc_test":      prc_test,
               "auc_prc_train": auc_prc_train,
               "auc_prc_test":  auc_prc_test}

    return  results
