theano.config.optimizer='None'
theano.config.exception_verbosity='high'

import numpy as np
import root_numpy as rnp
from copy import deepcopy
import os
import scipy.stats as stats
import pickle
from sklearn.metrics import auc
import theano
import theano.tensor as T

from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.models import Sequential, Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten, TimeDistributedDense, Masking, Lambda
from keras.layers.normalization import BatchNormalization

def saveModel(fileNameBase, model):
    ###################################################
    ###  Function to save a trained keras model
    ###  Architecture saved in json, weights saved in h5
    ###  input:
    ###         fileNameBase:  str
    ###         model: keras model object
    ###################################################

    json_string = model.to_json()
    print 'base ', fileNameBase
    open(fileNameBase+"_architecture.json", 'w').write(json_string)
    model.save_weights(fileNameBase + '_model_weights.h5', overwrite=True)


def loadModel(fileNameBase)
    ###################################################
    ###  Function to load a keras model from json and h5
    ###  input:
    ###         fileNameBase:  str
    ###         model:         keras model object
    ###################################################

    model = model_from_json(open( fileNameBase+'_architecture.json').read())
    model.load_weights(fileNameBase + '_model_weights.h5')
    return  model


def ANNStruct(inputshape):
    ############################################################################
    ###  Function to build ANN network structure
    ###  input:
    ###         inputshape: array[int], 
    ###                     the input layer shape, should be the dimension of features
    ###  output:
    ###         ANN_Model:  keras model object (not trained, just a structure)
    ############################################################################


    ## hyper parmaeters of the network
    n_dense_classification = 512 
    n_dense_adversary = 100

    input_layer = Input(inputshape)

    ## first we have the classifier part 
    ## one can change the structure to add more rectifier layers and hidden layers
    ## the output in a binary classification problem should be sigmoid
    dense_cl_batch  = BatchNormalization(name="batch_cl")(input_layer)
    dense_cl_hidden = Dense(n_dense_classification, activation="tanh", name="dense_cl_hidden", init="he_uniform")(dense_cl_batch)
    dense_cl_output = Dense(1, name="dense_cl_output", init="he_uniform")(dense_cl_hidden)
    dense_cl_sigmoid = Activation('sigmoid', name="cl_sigmoid")(dense_cl_output) 
    
    ## then the output of classifer is fed to another network
    ## which tries to fit the Mbb (here the example is fitting Mbb bins, not the Mbb value directly)
    ## one could try directly fitting Mbb, using linear output layer and 2-norm as loss
    dense_ad_hidden = Dense(n_dense_adversary, activation='tanh', name='dense_ad_hidden', init = "he_uniform")(dense_cl_sigmoid)
    dense_ad_output = Dense(nMBBbins, activation='softmax', name='dense_ad_output', init = "he_uniform")(dense_ad_hidden)
    MergeOutput = merge([ dense_cl_sigmoid, dense_ad_output ], mode='concat')

    ANN_Model = Model(input=input_layer, output =MergeOutput)
    return ANN_Model


def TrainANN(dataset, lamb=1.0, clpretrain = 50, adpretrain = 50, 
                     epoch=50,  minibatch = 20, nMBBbins = 16):

    ############################################################################
    ###  Function to  Train ANN network 
    ###  
    ###  input:
    ###         dataset:     dictionary[arrays], input data 
    ###         lamb:        float, penalization factor
    ###         clpretrain:  int, classifier pre-train epoches
    ###         adpretrain:  int, adversary pre-train epoches
    ###         epoch:       int, network training epoches
    ###         minibatch:   int, mini-batch size
    ###         nMBBbins:    int, number of discretized Mbb bins to fit
    ############################################################################


    ## delcare training history arrays for debuging
    history = {}
    history["cl"] = []
    history["ad"] = []
    history["ann_cl"] = []
    history["ann_ad"] = []

    ## unload data from dictionary
    ## X are the features
    ## y are the labels
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    weights_train = dataset['weights_train']

    X_train_bkg = X_train[ y_train[:,-1]==0]
    y_train_bkg = y_train[ y_train[:,-1]==0]

    X_test_bkg = X_test[ y_test[:,-1]==0]
    y_test_bkg = y_test[ y_test[:,-1]==0]

    ## get the network model and set training hyper parameters
    print ("building ann")
    model = ANNStruct(X_train.shape[1:]) 
    adam = Adam(lr = 5e-5)
    adam_quick = Adam(lr = 5e-4)

    print (model.get_config())
    print "model summary"
    print model.summary()


    ## three separate loss needed 
    ## (1) cl only objective
    ## (2) ad only objective
    ## (3) linked objective penalizing cl objective by lamb*ad objective

    def custom_objective_cl_only(y_true, y_pred):
        c_pred = T.clip(y_pred[:,0], 0.0+1e-5, 1.0-1e-5)
        bce = T.nnet.binary_crossentropy(c_pred, y_true[:,-1]).mean()
        loss = bce
        return loss

    def custom_objective_ad_only(y_true, y_pred):
        a_pred = T.clip(y_pred[:, 1:nMBBbins+1], 0.0+1e-6, 1.0-1e-6)
        cce = T.nnet.categorical_crossentropy(a_pred, y_true[:,0:nMBBbins]).mean()
        loss = cce
        return loss

    def custom_objective(y_true, y_pred):
        c_pred = T.clip(y_pred[:,0], 0.0+1e-5, 1.0-1e-5)
        a_pred = T.clip(y_pred[:, 1:nMBBbins+1], 0.0+1e-5, 1.0-1e-5)
        bce = T.nnet.binary_crossentropy(c_pred, y_true[:,-1]).mean()
        cce = T.nnet.categorical_crossentropy(a_pred, y_true[:,0:nMBBbins]).mean()
        loss = bce - lamb* cce
        return loss

    #####################################################
    ##### Step 1. Pre-Train For the classification model
    ##### (not necessarily needed, please customize)
    #####################################################

    print '----------------- Pre-train for classification-------------------------'
    print '----------------- fix pars-------------------------'

    ## when pre-training classifer, adversary parameters need to be frozen
    for il in range(len(model.layers)):
        if "ad" in model.layers[il].name:
            model.layers[il].trainable = False
        if "cl" in model.layers[il].name:
            model.layers[il].trainable = True
            
    ## the model is trained with cl only objective
    model.compile(loss=custom_objective_cl_only, optimizer=adam, metrics=["accuracy"])
    history["cl"] = model.fit( X_train , y_train, batch_size=batch_size, nb_epoch=clpretrain, validation_split=0.2, shuffle = True, sample_weight = weights_train ).history['loss']


    #####################################################
    ##### Step 2. Pre-Train For the adversary model
    ##### (not necessarily needed, please customize)
    #####################################################

    print '----------------- Pre-train for ad-------------------------'
    print '----------------- fix pars-------------------------'

    ## when pre-training adversary, classifer parameters need to be frozen
    for il in range(len(model.layers)):
        if "ad" in model.layers[il].name:
            model.layers[il].trainable = True
        if "cl" in model.layers[il].name:
            model.layers[il].trainable = False

    ## the model is trained with ad only objective
    model.compile(loss=custom_objective_ad_only, optimizer=adam_quick, metrics=["accuracy"])
    try:
        history["ad"] = model.fit( X_train_bkg , y_train_bkg, batch_size=batch_size, nb_epoch=adpretrain, validation_split=0.2, shuffle = True).history['loss']
    except KeyError:
        history["ad"] = []


    #####################################################
    ##### Step 3. Training both cl and ad at the same time
    #####################################################
    for ie in range(epoch):
        print ie, 'adversary network training'

        mini_batch_size = X_train.shape[0]/minibatch

        for isub in range(minibatch):

            print 'epoch', ie, 'training on minibatch', isub
            for il in range(len(model.layers)):
                if "ad" in model.layers[il].name:
                    model.layers[il].trainable = False
                if "cl" in model.layers[il].name:
                    model.layers[il].trainable = True
            model.compile(loss=custom_objective, optimizer=adam, metrics=["accuracy"])
            indices = np.random.permutation(len(X_train))[:mini_batch_size]
            model.train_on_batch(X_train[indices], y_train[indices])

            history["ann_cl"].append( model.evaluate(X_test, y_test, batch_size=batch_size) )

            print 'epoch', ie, 'training on ad only for minibatch', isub
            for il in range(len(model.layers)):
                if "ad" in model.layers[il].name:
                    model.layers[il].trainable = True
                if "cl" in model.layers[il].name:
                    model.layers[il].trainable = False

            model.compile(loss=custom_objective_ad_only, optimizer=adam_quick, metrics=["accuracy"])
            model.fit( X_train_bkg , y_train_bkg, batch_size=batch_size, nb_epoch=1, validation_split=0.2, shuffle = True)

            history["ann_ad"].append( model.evaluate(X_test_bkg, y_test_bkg, batch_size=batch_size) )


    ## release all the parameters
    for il in range(len(model.layers)):
        if "ad" in model.layers[il].name:
            model.layers[il].trainable = True
        if "cl" in model.layers[il].name:
            model.layers[il].trainable = True

    fileNameBase = "ANN_lambda"+str(lamb)+"_clpretrain"+str(clpretrain)+"_adpretrain"+str(adpretrain)+"_epoch"+str(epoch)+"_minibatch"+str(minibatch)+"_mBBbins"+str(nMBBbins)
    saveModel(fileNameBase, model)

    return model, history


def PredictModel(model, dataset):

    ############################################################################
    ###  Example function to use the trained model to give predictions
    ###  
    ###  input:
    ###         model:       keras model object
    ###         dataset:     dictionary[arrays], input data 
    ############################################################################

    X_test = dataset['X_test']
    y_test = dataset['y_test']
    X_train = dataset['X_train']
    y_train = dataset['y_train']

    pred_train = model.predict( X_train)[:,0]
    pred_test = model.predict( X_test )[:,0]
        
    pred_sig_train =  pred_train[ y_train[:,-1]==1 ]
    pred_bkg_train =  pred_train[ y_train[:,-1]==0 ]
    pred_sig_test =  pred_test[ y_test[:,-1]==1 ]
    pred_bkg_test =  pred_test[ y_test[:,-1]==0 ]

