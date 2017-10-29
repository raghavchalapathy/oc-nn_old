import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib
from sklearn import svm
nu = 0.04
dataPath = './data/'


colNames = ["sklearn-OCSVM-Linear-Train","sklearn-OCSVM-RBF-Train","sklearn-OCSVM-Linear-Test","sklearn-OCSVM-RBF-Test","sklearn-explicit-Linear-Train","sklearn-explicit-Sigmoid-Train","sklearn-explicit-Linear-Test","sklearn-explicit-Sigmoid-Test","tf-Linear-Train","tf-Sigmoid-Train","tf-Linear-Test","tf-Sigmoid-Test","tfLearn-Linear-Train","tfLearn-Sigmoid-Train","tfLearn-Linear-Test","tfLearn-Sigmoid-Test"]

# Create empty dataframe with given column names.
df_usps_scores = {}
df_fake_news_scores = {}
df_spam_vs_ham_scores = {}
df_cifar_10_scores = {}


def sklearn_OCSVM_linear(data_train,data_test):

    ocSVM = svm.OneClassSVM(nu = nu, kernel = 'linear')
    ocSVM.fit(data_train) 

    pos_decisionScore = ocSVM.decision_function(data_train)
    neg_decisionScore = ocSVM.decision_function(data_test)

    return [pos_decisionScore,neg_decisionScore]


def sklearn_OCSVM_rbf(data_train,data_test):
    
    ocSVM = svm.OneClassSVM(nu = nu, kernel = 'rbf')
    ocSVM.fit(data_train) 

    pos_decisionScore = ocSVM.decision_function(data_train)
    neg_decisionScore = ocSVM.decision_function(data_test)

    return [pos_decisionScore,neg_decisionScore]


def func_getDecision_Scores_sklearn_OCSVM(dataset,data_train,data_test):


    # print "Decision_Scores_sklearn_OCSVM Using Linear and RBF Kernels....."

    if(dataset=="USPS" ):
        
        result = sklearn_OCSVM_linear(data_train,data_test)
        df_usps_scores["sklearn-OCSVM-Linear-Train"] = result[0]
        df_usps_scores["sklearn-OCSVM-Linear-Test"] =  result[1]
 
        result = sklearn_OCSVM_rbf(data_train,data_test)
        df_usps_scores["sklearn-OCSVM-RBF-Train"] = result[0]
        df_usps_scores["sklearn-OCSVM-RBF-Test"] = result[1]


    if(dataset=="FAKE_NEWS" ):   
        result = sklearn_OCSVM_linear(data_train,data_test)
        df_fake_news_scores["sklearn-OCSVM-Linear-Train"] = result[0]
        df_fake_news_scores["sklearn-OCSVM-Linear-Test"] = result[1]
        
        result = sklearn_OCSVM_rbf(data_train,data_test)
        df_fake_news_scores["sklearn-OCSVM-RBF-Train"] = result[0]
        df_fake_news_scores["sklearn-OCSVM-RBF-Test"] = result[1]


    if(dataset=="SPAM_Vs_HAM" ):
        result = sklearn_OCSVM_linear(data_train,data_test)
        df_spam_vs_ham_scores["sklearn-OCSVM-Linear-Train"] = result[0] 
        df_spam_vs_ham_scores["sklearn-OCSVM-Linear-Test"] = result[1]
        
        result = sklearn_OCSVM_rbf(data_train,data_test)
        df_spam_vs_ham_scores["sklearn-OCSVM-RBF-Train"] = result[0]
        df_spam_vs_ham_scores["sklearn-OCSVM-RBF-Test"] = result[1]


    if(dataset=="CIFAR-10" ):
        result = sklearn_OCSVM_linear(data_train,data_test)
        df_cifar_10_scores["sklearn-OCSVM-Linear-Train"] = result[0]
        df_cifar_10_scores["sklearn-OCSVM-Linear-Test"] = result[1]
        
        result = sklearn_OCSVM_rbf(data_train,data_test)
        df_cifar_10_scores["sklearn-OCSVM-RBF-Train"] = result[0]
        df_cifar_10_scores["sklearn-OCSVM-RBF-Test"] = result[1]

    return [df_usps_scores,df_fake_news_scores,df_spam_vs_ham_scores,df_cifar_10_scores]


