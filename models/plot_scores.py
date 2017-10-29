import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib.pyplot as plt

dataPath = './data/'


def plot_decision_scores(model,dataset,df_usps_scores,df_fake_news_scores,df_spam_vs_ham_scores,df_cifar_10_scores):

    df_usps = pd.DataFrame(df_usps_scores.items(), columns=df_usps_scores.keys())
    df_cifar = pd.DataFrame(df_cifar_10_scores.items(), columns=df_cifar_10_scores.keys())

    ## PLot for USPS 
    if(dataset=="USPS" ):
        plt.hist(df_usps["sklearn-OCSVM-Linear-Train"], bins = 25, label = 'Normal');
        plt.hist(df_usps["sklearn-OCSVM-Linear-Test"], bins = 25, label = 'Anomaly');
        plt.legend(loc = 'upper right');
        plt.title('OC-SVM Normalised Decision Score for : '+dataset+ ": "+ model);

        plt.hist(df_usps["sklearn-OCSVM-RBF-Train"], bins = 25, label = 'Normal');
        plt.hist(df_usps["sklearn-OCSVM-RBF-Test"], bins = 25, label = 'Anomaly');
        plt.legend(loc = 'upper right');
        plt.title('OC-SVM Normalised Decision Score for : '+dataset+ ": "+ model);


    ## PLot for CIFAR-10
    if(dataset=="CIFAR-10" ):
        plt.hist(df_cifar["sklearn-OCSVM-Linear-Train"], bins = 25, label = 'Normal');
        plt.hist(df_cifar["sklearn-OCSVM-Linear-Test"], bins = 25, label = 'Anomaly');
        plt.legend(loc = 'upper right');
        plt.title('OC-SVM Normalised Decision Score for : '+dataset+ ": "+ model);

        plt.hist(df_cifar["sklearn-OCSVM-RBF-Train"], bins = 25, label = 'Normal');
        plt.hist(df_cifar["sklearn-OCSVM-RBF-Test"], bins = 25, label = 'Anomaly');
        plt.legend(loc = 'upper right');
        plt.title('OC-SVM Normalised Decision Score for : '+dataset+ ": "+ model);

   
    return 

