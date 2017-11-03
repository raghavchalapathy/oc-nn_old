import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib
from scipy.optimize import minimize
import tensorflow as tf
import numpy as np
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as srn


dataPath = './data/'

colNames = ["sklearn-OCSVM-Linear-Train","sklearn-OCSVM-RBF-Train","sklearn-OCSVM-Linear-Test","sklearn-OCSVM-RBF-Test","sklearn-explicit-Linear-Train","sklearn-explicit-Sigmoid-Train","sklearn-explicit-Linear-Test","sklearn-explicit-Sigmoid-Test","tf-Linear-Train","tf-Sigmoid-Train","tf-Linear-Test","tf-Sigmoid-Test","tfLearn-Linear-Train","tfLearn-Sigmoid-Train","tfLearn-Linear-Test","tfLearn-Sigmoid-Test"]

# Create empty dataframe with given column names.
df_usps_scores = {}
df_fake_news_scores = {}
df_spam_vs_ham_scores = {}
df_cifar_10_scores = {}
nu = 0.04
K  = 4



def tf_OneClass_NN_linear(data_train,data_test):

    tf.reset_default_graph()

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    train_X = data_train

     # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 4                # Number of hidden nodes
    y_size = 1   # Number of outcomes (3 iris flowers)
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K*D + 1) 
    rvalue = np.random.normal(0,1,(len(train_X),y_size))
    nu = 0.04



    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape,mean=0, stddev=0.00001)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        w_1 = tf.cast(w_1, tf.float64)
        w_2 = tf.cast(w_2, tf.float64)
        h    = (tf.matmul(X, w_1))  # 
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    g   = lambda x : 1/(1 + tf.exp(-x))

    def nnScore(X, w, V, g):
        w = tf.cast(w, tf.float64)
        V = tf.cast(V, tf.float64)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
    #     y[y < 0] = 0
        return y

    def ocnn_obj(theta, X, nu, w1, w2, g,r):
        
        w = w1
        V = w2

       
        X = tf.cast(X, tf.float64)
        w = tf.cast(w1, tf.float64)
        V = tf.cast(w2, tf.float64)

        
        term1 = 0.5  * tf.reduce_sum(w**2)
        term2 = 0.5  * tf.reduce_sum(V**2)
        term3 = 1/nu * tf.reduce_mean(relu(r - nnScore(X, w, V, g)))
        term4 = -r
        
        return term1 + term2 + term3 + term4




    # For testing the algorithm
    test_X = data_test


    # Symbols
    X = tf.placeholder("float64", shape=[None, x_size])

    r = tf.get_variable("r", dtype=tf.float64,shape=(),trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)


    # Backward propagation
    # cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    cost    = ocnn_obj(theta, X, nu, w_1, w_2, g,r)
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1
    for epoch in range(100):
            # Train with each example
                sess.run(updates, feed_dict={X: train_X,r:rvalue})
                rvalue = nnScore(train_X, w_1, w_2, g)
                with sess.as_default():
                    rvalue = rvalue.eval()
                    rvalue = np.percentile(rvalue,q=100*0.04)
                print("Epoch = %d, r = %f"
                  % (epoch + 1,rvalue))

                
    train = nnScore(train_X, w_1, w_2, g)
    test = nnScore(test_X, w_1, w_2, g)
    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()
    #     rstar = r.eval()

    rstar =rvalue
    sess.close()
    print "Session Closed!!!"


    pos_decisionScore = arrayTrain-rstar
    neg_decisionScore = arrayTest-rstar


    return [pos_decisionScore,neg_decisionScore]



def tf_OneClass_NN_sigmoid(data_train,data_test):

    tf.reset_default_graph()

    train_X = data_train

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

     # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 4                # Number of hidden nodes
    y_size = 1   # Number of outcomes (3 iris flowers)
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K*D + 1) 
    rvalue = np.random.normal(0,1,(len(train_X),y_size))
    nu = 0.04



    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape,mean=0, stddev=0.00001)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        w_1 = tf.cast(w_1, tf.float64)
        w_2 = tf.cast(w_2, tf.float64)
        h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    g   = lambda x : 1/(1 + tf.exp(-x))

    def nnScore(X, w, V, g):
        w = tf.cast(w, tf.float64)
        V = tf.cast(V, tf.float64)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
    #     y[y < 0] = 0
        return y

    def ocnn_obj(theta, X, nu, w1, w2, g,r):
        
        w = w1
        V = w2

       
        X = tf.cast(X, tf.float64)
        w = tf.cast(w1, tf.float64)
        V = tf.cast(w2, tf.float64)

        
        term1 = 0.5  * tf.reduce_sum(w**2)
        term2 = 0.5  * tf.reduce_sum(V**2)
        term3 = 1/nu * tf.reduce_mean(relu(r - nnScore(X, w, V, g)))
        term4 = -r
        
        return term1 + term2 + term3 + term4




    # For testing the algorithm
    test_X = data_test


    # Symbols
    X = tf.placeholder("float64", shape=[None, x_size])

    r = tf.get_variable("r", dtype=tf.float64,shape=(),trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)


    # Backward propagation
    # cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    cost    = ocnn_obj(theta, X, nu, w_1, w_2, g,r)
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1
    for epoch in range(100):
            # Train with each example
                sess.run(updates, feed_dict={X: train_X,r:rvalue})
                rvalue = nnScore(train_X, w_1, w_2, g)
                with sess.as_default():
                    rvalue = rvalue.eval()
                    rvalue = np.percentile(rvalue,q=100*0.04)
                print("Epoch = %d, r = %f"
                  % (epoch + 1,rvalue))

                
    train = nnScore(train_X, w_1, w_2, g)
    test = nnScore(test_X, w_1, w_2, g)
    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()
    #     rstar = r.eval()

    rstar =rvalue
    sess.close()
    print "Session Closed!!!"


    pos_decisionScore = arrayTrain-rstar
    neg_decisionScore = arrayTest-rstar


    return [pos_decisionScore,neg_decisionScore]








def func_getDecision_Scores_tf_OneClass_NN(dataset,data_train,data_test):


    # print "Decision_Scores_sklearn_OCSVM Using Linear and RBF Kernels....."

    if(dataset=="USPS" ):
        
        result = tf_OneClass_NN_linear(data_train,data_test)
        df_usps_scores["tf_OneClass_NN-Linear-Train"] = result[0]
        df_usps_scores["tf_OneClass_NN-Linear-Test"] =  result[1]
 
        result = tf_OneClass_NN_sigmoid(data_train,data_test)
        df_usps_scores["tf_OneClass_NN-Sigmoid-Train"] = result[0]
        df_usps_scores["tf_OneClass_NN-Sigmoid-Test"] = result[1]


    if(dataset=="FAKE_NEWS" ):   
        result = tf_OneClass_NN_linear(data_train,data_test)
        df_fake_news_scores["tf_OneClass_NN-Linear-Train"] = result[0]
        df_fake_news_scores["tf_OneClass_NN-Linear-Test"] = result[1]
        
        result = tf_OneClass_NN_sigmoid(data_train,data_test)
        df_fake_news_scores["tf_OneClass_NN-Sigmoid-Train"] = result[0]
        df_fake_news_scores["tf_OneClass_NN-Sigmoid-Test"] = result[1]


    # if(dataset=="SPAM_Vs_HAM" ):
    #     result = tf_OneClass_NN_linear(data_train,data_test)
    #     df_spam_vs_ham_scores["tf_OneClass_NN-Linear-Train"] = result[0] 
    #     df_spam_vs_ham_scores["tf_OneClass_NN-Linear-Test"] = result[1]
        
    #     result = tf_OneClass_NN_sigmoid(data_train,data_test)
    #     df_spam_vs_ham_scores["tf_OneClass_NN-Sigmoid-Train"] = result[0]
    #     df_spam_vs_ham_scores["tf_OneClass_NN-Sigmoid-Test"] = result[1]


    if(dataset=="CIFAR-10" ):
        result = tf_OneClass_NN_linear(data_train,data_test)
        df_cifar_10_scores["tf_OneClass_NN-Linear-Train"] = result[0]
        df_cifar_10_scores["tf_OneClass_NN-Linear-Test"] = result[1]
        
        result = tf_OneClass_NN_sigmoid(data_train,data_test)
        df_cifar_10_scores["tf_OneClass_NN_Sigmoid-Train"] = result[0]
        df_cifar_10_scores["tf_OneClass_NN_Sigmoid-Test"] = result[1]

    return [df_usps_scores,df_fake_news_scores,df_spam_vs_ham_scores,df_cifar_10_scores]




