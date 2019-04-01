import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def buildBDT(dataset):

    max_depth = 1
    n_estimators = 300

    print ("building bdt")
    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    weights_train = dataset['weights_train']

    bdt_discrete = AdaBoostClassifier( DecisionTreeClassifier(max_depth=max_depth), 
                                       n_estimators=n_estimators, algorithm="SAMME.R", learning_rate=0.2)
    bdt_discrete.fit(X_train, Y_train)

    print ("finished building bdt")
    return bdt_discrete


def predictBDT(model, dataset):

    print ("predicting ")
        
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']
    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    
    pred_train = model.decision_function( X_train )
    pred_test  = model.decision_function( X_test )

    print( pred_test.shape)
    
    roc_train  = roc_curve( Y_train, pred_train)
    roc_test   = roc_curve( Y_test,  pred_test)

    auc_train  = round(auc(roc_train[1], roc_train[0], reorder=True), 3)
    auc_test   = round(auc(roc_test[1], roc_test[0], reorder=True), 3)

    results = {"pred_train": pred_train,
               "pred_test":  pred_test, 
               "roc_train":  roc_train,
               "roc_test":   roc_test,
               "auc_train":  auc_train,
               "auc_test":   auc_test}

    return  results
