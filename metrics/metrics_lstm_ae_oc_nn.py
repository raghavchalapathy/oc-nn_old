from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
K = 10
def au_prc(y_true,y_score):
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_true, y_score)

    print('Average precision-recall score: {0:0.4f}'.format(
        average_precision))

    # precision, recall, _ = precision_recall_curve(y_test, y_score)
    #
    # plt.step(recall, precision, color='b', alpha=0.2,
    #          where='post')
    # plt.fill_between(recall, precision, step='post', alpha=0.2,
    #                  color='b')
    #
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
    #     average_precision))

    return average_precision
def au_roc(y_true,y_score):
    from sklearn.metrics import roc_auc_score
    roc_score = roc_auc_score(y_true, y_score)

    print('ROC  score: {0:0.4f}'.format(
        roc_score))

    return roc_score
def compute_au_prc(y_true,df_score):
    ### OCSVM-Linear
    y_scores_pos = df_score["sklearn-OCSVM-Linear-Train"]
    y_scores_neg = df_score["sklearn-OCSVM-Linear-Test"]
    y_score = np.concatenate((y_scores_pos,y_scores_neg))
    ocsvm_lin_prc = au_prc(y_true,y_score)

    ### OCSVM- RBF
    y_scores_pos = df_score["sklearn-OCSVM-RBF-Train"]
    y_scores_neg = df_score["sklearn-OCSVM-RBF-Test"]
    y_score = np.concatenate((y_scores_pos,y_scores_neg))
    ocsvm_rbf_prc = au_prc(y_true,y_score)


    ### RPCA OCSVM_Linear
    y_scores_pos = df_score["rpca_ocsvm-Linear-Train"]
    y_scores_neg = df_score["rpca_ocsvm-Linear-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    rpca_ocsvm_prc_linear = au_prc(y_true, y_score)

    ### RPCA OCSVM_RBF
    y_scores_pos = df_score["rpca_ocsvm-rbf-Train"]
    y_scores_neg = df_score["rpca_ocsvm-rbf-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    rpca_ocsvm_prc_rbf = au_prc(y_true, y_score)

    ### Isolation forest
    y_scores_pos = df_score["isolation-forest-Train"]
    y_scores_neg = df_score["isolation-forest-Test"]
    y_score = np.concatenate((y_scores_pos,y_scores_neg))
    isolation_forest_prc = au_prc(y_true,y_score)


    ### OC-NN Linear
    y_scores_pos = df_score["tf_OneClass_NN-Linear-Train"]
    y_scores_neg = df_score["tf_OneClass_NN-Linear-Test"]
    y_score = np.concatenate((y_scores_pos,y_scores_neg))
    ocnn_lin_prc = au_prc(y_true,y_score)

    ### OC-NN Sigmoid
    y_scores_pos = df_score["tf_OneClass_NN-Sigmoid-Train"]
    y_scores_neg = df_score["tf_OneClass_NN-Sigmoid-Test"]
    y_score = np.concatenate((y_scores_pos,y_scores_neg))
    ocnn_sig_prc = au_prc(y_true,y_score)

    ## OC-NN Relu
    y_scores_pos = df_score["tf_OneClass_NN-Relu-Train"]
    y_scores_neg = df_score["tf_OneClass_NN-Relu-Test"]
    y_score = np.concatenate((y_scores_pos,y_scores_neg))
    ocnn_relu_prc = au_prc(y_true,y_score)

    y_scores_pos = df_score["lstm_ocsvm-linear-Train"]
    y_scores_neg = df_score["lstm_ocsvm-linear-Test"]
    y_score = np.concatenate((y_scores_pos,y_scores_neg))
    lstm_ocsvm_lin_prc = au_prc(y_true,y_score)

    ### OCSVM- RBF
    y_scores_pos = df_score["lstm_ocsvm-rbf-Train"]
    y_scores_neg = df_score["lstm_ocsvm-rbf-Test"]
    y_score = np.concatenate((y_scores_pos,y_scores_neg))
    lstm_ocsvm_rbf_prc = au_prc(y_true,y_score)

    ### AE2- Linear
    y_scores_pos = df_score["ae_svdd-linear-Train"]
    y_scores_neg = df_score["ae_svdd-linear-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ae_svdd_lin_prc = au_prc(y_true, y_score)

    ### AE2- RBF
    y_scores_pos = df_score["ae_svdd-rbf-Train"]
    y_scores_neg = df_score["ae_svdd-rbf-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ae_svdd_rbf_prc = au_prc(y_true, y_score)
    return [ocnn_lin_prc,ocnn_sig_prc,ocnn_relu_prc,lstm_ocsvm_lin_prc,lstm_ocsvm_rbf_prc,ae_svdd_lin_prc,ae_svdd_rbf_prc,ocsvm_lin_prc,ocsvm_rbf_prc,rpca_ocsvm_prc_linear,rpca_ocsvm_prc_rbf,isolation_forest_prc]



def compute_au_roc(y_true, df_score):
    ### OCSVM-Linear
    y_scores_pos = df_score["sklearn-OCSVM-Linear-Train"]
    y_scores_neg = df_score["sklearn-OCSVM-Linear-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ocsvm_lin_roc = au_roc(y_true, y_score)

    ### OCSVM- RBF
    y_scores_pos = df_score["sklearn-OCSVM-RBF-Train"]
    y_scores_neg = df_score["sklearn-OCSVM-RBF-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ocsvm_rbf_roc = au_roc(y_true, y_score)

    ### RPCA OCSVM_Linear
    y_scores_pos = df_score["rpca_ocsvm-Linear-Train"]
    y_scores_neg = df_score["rpca_ocsvm-Linear-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    rpca_ocsvm_roc_linear = au_roc(y_true, y_score)

    ### RPCA OCSVM_RBF
    y_scores_pos = df_score["rpca_ocsvm-rbf-Train"]
    y_scores_neg = df_score["rpca_ocsvm-rbf-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    rpca_ocsvm_roc_rbf = au_roc(y_true, y_score)

    ### Isolation forest
    y_scores_pos = df_score["isolation-forest-Train"]
    y_scores_neg = df_score["isolation-forest-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    isolation_forest_roc = au_roc(y_true, y_score)

    ### OC-NN Linear
    y_scores_pos = df_score["tf_OneClass_NN-Linear-Train"]
    y_scores_neg = df_score["tf_OneClass_NN-Linear-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ocnn_lin_roc = au_roc(y_true, y_score)

    ### OC-NN Sigmoid
    y_scores_pos = df_score["tf_OneClass_NN-Sigmoid-Train"]
    y_scores_neg = df_score["tf_OneClass_NN-Sigmoid-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ocnn_sig_roc = au_roc(y_true, y_score)

    ### OC-NN RELU
    y_scores_pos = df_score["tf_OneClass_NN-Relu-Train"]
    y_scores_neg = df_score["tf_OneClass_NN-Relu-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ocnn_relu_roc = au_roc(y_true, y_score)

    y_scores_pos = df_score["lstm_ocsvm-linear-Train"]
    y_scores_neg = df_score["lstm_ocsvm-linear-Test"]
    y_score = np.concatenate((y_scores_pos,y_scores_neg))
    lstm_ocsvm_lin_roc = au_roc(y_true,y_score)

    ### OCSVM- RBF
    y_scores_pos = df_score["lstm_ocsvm-rbf-Train"]
    y_scores_neg = df_score["lstm_ocsvm-rbf-Test"]
    y_score = np.concatenate((y_scores_pos,y_scores_neg))
    lstm_ocsvm_rbf_roc = au_roc(y_true,y_score)

    ### AE2- Linear
    y_scores_pos = df_score["ae_svdd-linear-Train"]
    y_scores_neg = df_score["ae_svdd-linear-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ae_svdd_lin_roc = au_roc(y_true, y_score)

    ### AE2- RBF
    y_scores_pos = df_score["ae_svdd-rbf-Train"]
    y_scores_neg = df_score["ae_svdd-rbf-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ae_svdd_rbf_roc = au_roc(y_true, y_score)



    return [ocnn_lin_roc, ocnn_sig_roc,ocnn_relu_roc,ocsvm_lin_roc, lstm_ocsvm_lin_roc,lstm_ocsvm_rbf_roc,ae_svdd_lin_roc,ae_svdd_rbf_roc,ocsvm_rbf_roc, rpca_ocsvm_roc_linear,rpca_ocsvm_roc_rbf,isolation_forest_roc]
def compute_prec_at_10(y_true, df_score):
    ### OCSVM-Linear
    y_scores_pos = df_score["sklearn-OCSVM-Linear-Train"]
    y_scores_neg = df_score["sklearn-OCSVM-Linear-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ocsvm_lin_prec = compute_precAtK(y_true, y_score)

    ### OCSVM- RBF
    y_scores_pos = df_score["sklearn-OCSVM-RBF-Train"]
    y_scores_neg = df_score["sklearn-OCSVM-RBF-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ocsvm_rbf_prec = compute_precAtK(y_true, y_score)

    ### RPCA OCSVM_Linear
    y_scores_pos = df_score["rpca_ocsvm-Linear-Train"]
    y_scores_neg = df_score["rpca_ocsvm-Linear-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    rpca_ocsvm_prec_linear = compute_precAtK(y_true, y_score)

    ### RPCA OCSVM_RBF
    y_scores_pos = df_score["rpca_ocsvm-rbf-Train"]
    y_scores_neg = df_score["rpca_ocsvm-rbf-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    rpca_ocsvm_prec_rbf = compute_precAtK(y_true, y_score)

    ### Isolation forest
    y_scores_pos = df_score["isolation-forest-Train"]
    y_scores_neg = df_score["isolation-forest-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    isolation_forest_prec = compute_precAtK(y_true, y_score)

    ### OC-NN Linear
    y_scores_pos = df_score["tf_OneClass_NN-Linear-Train"]
    y_scores_neg = df_score["tf_OneClass_NN-Linear-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ocnn_lin_prec = compute_precAtK(y_true, y_score)

    ### OC-NN Sigmoid
    y_scores_pos = df_score["tf_OneClass_NN-Sigmoid-Train"]
    y_scores_neg = df_score["tf_OneClass_NN-Sigmoid-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ocnn_sig_prec = compute_precAtK(y_true, y_score)

    y_scores_pos = df_score["tf_OneClass_NN-Relu-Train"]
    y_scores_neg = df_score["tf_OneClass_NN-Relu-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ocnn_relu_prec = compute_precAtK(y_true, y_score)

    y_scores_pos = df_score["lstm_ocsvm-linear-Train"]
    y_scores_neg = df_score["lstm_ocsvm-linear-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    lstm_ocsvm_lin_prec = compute_precAtK(y_true, y_score)

    ### OCSVM- RBF
    y_scores_pos = df_score["lstm_ocsvm-rbf-Train"]
    y_scores_neg = df_score["lstm_ocsvm-rbf-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    lstm_ocsvm_rbf_prec = compute_precAtK(y_true, y_score)

    ### AE2- Linear
    y_scores_pos = df_score["ae_svdd-linear-Train"]
    y_scores_neg = df_score["ae_svdd-linear-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ae_svdd_lin_prec = compute_precAtK(y_true, y_score)

    ### AE2- RBF
    y_scores_pos = df_score["ae_svdd-rbf-Train"]
    y_scores_neg = df_score["ae_svdd-rbf-Test"]
    y_score = np.concatenate((y_scores_pos, y_scores_neg))
    ae_svdd_rbf_prec = compute_precAtK(y_true, y_score)


    return [ocnn_lin_prec, ocnn_sig_prec,ocnn_relu_prec,lstm_ocsvm_lin_prec,lstm_ocsvm_rbf_prec,ocsvm_lin_prec,ae_svdd_lin_prec,ae_svdd_rbf_prec, ocsvm_rbf_prec,rpca_ocsvm_prec_linear,rpca_ocsvm_prec_rbf,isolation_forest_prec]


def compute_precAtK(y_true, y_score, K = 10):

    if K is None:
        K = y_true.shape[0]

    # label top K largest predicted scores as + one's've

    idx = np.argsort(y_score)
    predLabel = np.zeros(y_true.shape)

    predLabel[idx[:K]] = 1

    prec = precision_score(y_true, predLabel)

    return prec


