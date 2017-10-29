import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib

from scipy.optimize import minimize

dataPath = './data/'

colNames = ["sklearn-OCSVM-Linear-Train","sklearn-OCSVM-RBF-Train","sklearn-OCSVM-Linear-Test","sklearn-OCSVM-RBF-Test","sklearn-explicit-Linear-Train","sklearn-explicit-Sigmoid-Train","sklearn-explicit-Linear-Test","sklearn-explicit-Sigmoid-Test","tf-Linear-Train","tf-Sigmoid-Train","tf-Linear-Test","tf-Sigmoid-Test","tfLearn-Linear-Train","tfLearn-Sigmoid-Train","tfLearn-Linear-Test","tfLearn-Sigmoid-Test"]

# Create empty dataframe with given column names.
df_usps_scores = {}
df_fake_news_scores = {}
df_spam_vs_ham_scores = {}
df_cifar_10_scores = {}
nu = 0.04

def relu(x):
    y = x
    y[y < 0] = 0
    return y

def dRelu(x):
    y = x
    y[x <= 0] = 0
    y[x > 0]  = np.ones((len(x[x > 0]),))
    return y

def svmScore(X, w):
    return X.dot(w)

def ocsvm_obj(theta, X, nu, D):
    
    w = theta[:D]
    r = theta[D:]
    
    term1 = 0.5 * np.sum(w**2)
    term2 = 1/nu * np.mean(relu(r - svmScore(X, w)))
    term3 = -r
    
    return term1 + term2 + term3

def ocsvm_grad(theta, X, nu, D):
    
    w = theta[:D]
    r = theta[D:]
    
    deriv = dRelu(r - svmScore(X, w))

    term1 = np.append(w, 0)
    term2 = np.append(1/nu * np.mean(deriv[:,np.newaxis] * (-X), axis = 0),
                      1/nu * np.mean(deriv))
    term3 = np.append(0*w, -1)

    grad = term1 + term2 + term3
    
    return grad



def sklearn_OCSVM_explicit_linear(data_train,data_test):

    X  = data_train
    D  = X.shape[1]


    np.random.seed(42);
    theta0 = np.random.normal(0, 1, D + 1);

    from scipy.optimize import check_grad
    print('Gradient error: %s' % check_grad(ocsvm_obj, ocsvm_grad, theta0, X, nu, D));

    res = minimize(ocsvm_obj, theta0, method = 'L-BFGS-B', jac = ocsvm_grad, args = (X, nu, D),
                   options = {'gtol': 1e-8, 'disp': True, 'maxiter' : 50000, 'maxfun' : 10000});

    pos_decisionScore = svmScore(data_train, res.x[0:-1]) - res.x[-1];
    neg_decisionScore = svmScore(data_test, res.x[0:-1]) - res.x[-1];


    return [pos_decisionScore,neg_decisionScore]


def sklearn_OCSVM_explicit_sigmoid(data_train,data_test):
    X  = data_train
    D  = X.shape[1]


    np.random.seed(42);
    theta0 = np.random.normal(0, 1, D + 1);

    from scipy.optimize import check_grad
    print('Gradient error: %s' % check_grad(ocsvm_obj, ocsvm_grad, theta0, X, nu, D));

    res = minimize(ocsvm_obj, theta0, method = 'L-BFGS-B', jac = ocsvm_grad, args = (X, nu, D),
                   options = {'gtol': 1e-8, 'disp': True, 'maxiter' : 50000, 'maxfun' : 10000});

    pos_decisionScore = svmScore(data_train, res.x[0:-1]) - res.x[-1];
    neg_decisionScore = svmScore(data_test, res.x[0:-1]) - res.x[-1];

    return [pos_decisionScore,neg_decisionScore]



def func_getDecision_Scores_sklearn_OCSVM_explicit(dataset,data_train,data_test):


    # print "Decision_Scores_sklearn_OCSVM Using Linear and RBF Kernels....."

    if(dataset=="USPS" ):
        
        result = sklearn_OCSVM_explicit_linear(data_train,data_test)
        df_usps_scores["sklearn-OCSVM-explicit-Linear-Train"] = result[0]
        df_usps_scores["sklearn-OCSVM-explicit-Linear-Test"] =  result[1]
 
        result = sklearn_OCSVM_explicit_sigmoid(data_train,data_test)
        df_usps_scores["sklearn-OCSVM-explicit-Sigmoid-Train"] = result[0]
        df_usps_scores["sklearn-OCSVM-explicit-Sigmoid-Test"] = result[1]


    if(dataset=="FAKE_NEWS" ):   
        result = sklearn_OCSVM_explicit_linear(data_train,data_test)
        df_fake_news_scores["sklearn-OCSVM-explicit-Linear-Train"] = result[0]
        df_fake_news_scores["sklearn-OCSVM-explicit-Linear-Test"] = result[1]
        
        result = sklearn_OCSVM_explicit_sigmoid(data_train,data_test)
        df_fake_news_scores["sklearn-OCSVM-explicit-Sigmoid-Train"] = result[0]
        df_fake_news_scores["sklearn-OCSVM-explicit-Sigmoid-Test"] = result[1]


    # if(dataset=="SPAM_Vs_HAM" ):
    #     result = sklearn_OCSVM_explicit_linear(data_train,data_test)
    #     df_spam_vs_ham_scores["sklearn-OCSVM-explicit-Linear-Train"] = result[0] 
    #     df_spam_vs_ham_scores["sklearn-OCSVM-explicit-Linear-Test"] = result[1]
        
    #     result = sklearn_OCSVM_explicit_sigmoid(data_train,data_test)
    #     df_spam_vs_ham_scores["sklearn-OCSVM-explicit-Sigmoid-Train"] = result[0]
    #     df_spam_vs_ham_scores["sklearn-OCSVM-explicit-Sigmoid-Test"] = result[1]


    if(dataset=="CIFAR-10" ):
        result = sklearn_OCSVM_explicit_linear(data_train,data_test)
        df_cifar_10_scores["sklearn-OCSVM-explicit-Linear-Train"] = result[0]
        df_cifar_10_scores["sklearn-OCSVM-explicit-Linear-Test"] = result[1]
        
        result = sklearn_OCSVM_explicit_sigmoid(data_train,data_test)
        df_cifar_10_scores["sklearn-OCSVM-explicit-Sigmoid-Train"] = result[0]
        df_cifar_10_scores["sklearn-OCSVM-explicit-Sigmoid-Test"] = result[1]

    return [df_usps_scores,df_fake_news_scores,df_spam_vs_ham_scores,df_cifar_10_scores]




