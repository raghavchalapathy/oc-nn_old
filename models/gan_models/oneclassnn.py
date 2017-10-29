from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression, oneClassNN
import tensorflow as tf
import tflearn
import numpy as np
import tflearn.variables as va
import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as srn

from data_fetch import prepare_usps_mlfetch

[Xtrue,Xlabels] = prepare_usps_mlfetch()

data  = Xtrue
label = Xlabels
data_train    = data[0:220]
data_test     = data[220:231]
targets_train = label[0:220]
targets_test  = label[220:231]
# Clear all the graph variables created in previous run and start fresh
# tf.reset_default_graph()

## Set up the data for running the algorithm
data_train = data[0:220]
target = Xlabels
X = data_train
Y = targets_train
Y = Y.tolist()
Y = [[i] for i in Y]

# For testing the algorithm
X_test = data_test
Y_test = targets_test
Y_test = Y_test.tolist()
Y_test = [[i] for i in Y_test]

No_of_inputNodes = X.shape[1]
K = 4
nu = 0.04
D = X.shape[1]

# Define the network
input_layer = input_data(shape=[None, No_of_inputNodes])  # input layer of size

np.random.seed(42)
theta0 = np.random.normal(0, 1, K + K*D + 1) *0.0001
#theta0 = np.random.normal(0, 1, K + K*D + 1) # For linear
# hidden_layer = fully_connected(input_layer, 4, bias=False, activation='sigmoid', name="hiddenLayer_Weights",
#                                weights_init="normal")  # hidden layer of size 2
#
#
# output_layer = fully_connected(hidden_layer, 1, bias=False,  activation='linear', name="outputLayer_Weights",
#                                weights_init="normal")  # output layer of size 1
#


hidden_layer = fully_connected(input_layer, 4,  activation='sigmoid', name="hiddenLayer_Weights",
                               weights_init="normal")  # hidden layer of size 2


output_layer = fully_connected(hidden_layer, 1,  activation='linear', name="outputLayer_Weights",
                               weights_init="normal")  # output layer of size 1

# Hyper parameters for the one class Neural Network
v = 0.04
nu = 0.04

# Initialize rho
value = 0.0001
init = tf.constant_initializer(value)
rho = va.variable(name='rho', dtype=tf.float32, shape=[], initializer=init)

rcomputed = []
auc = []


sess = tf.Session()
sess.run(tf.initialize_all_variables())
print sess.run(tflearn.get_training_mode()) #False
tflearn.is_training(True, session=sess)
print sess.run(tflearn.get_training_mode())  #now True

X  = data_train
D  = X.shape[1]
nu = 0.04




# temp = np.random.normal(0, 1, K + K*D + 1)[-1]

temp = theta0[-1] *1000000

# temp = tflearn.variables.get_value(rho, session=sess)

oneClassNN = oneClassNN(output_layer, v, rho, hidden_layer, output_layer, optimizer='sgd',
                        loss='OneClassNN_Loss',
                            learning_rate=1)

model = DNN(oneClassNN, tensorboard_verbose=3)

model.set_weights(output_layer.W, theta0[0:K][:,np.newaxis])
model.set_weights(hidden_layer.W, np.reshape(theta0[K:K +K*D],(D,K)))


iterStep = 0
while (iterStep < 100):
    print "Running Iteration :", iterStep
    # Call the cost function
    y_pred = model.predict(data_train)  # Apply some ops
    tflearn.is_training(False, session=sess)
    y_pred_test = model.predict(data_test)  # Apply some ops
    tflearn.is_training(True, session=sess)
    value = np.percentile(y_pred, v * 100)
    tflearn.variables.set_value(rho, value,session=sess)
    rStar = rho
    model.fit(X, Y, n_epoch=2, show_metric=True, batch_size=100)
    iterStep = iterStep + 1
    rcomputed.append(rho)
    temp = tflearn.variables.get_value(rho, session=sess)

    # print "Rho",temp
    # print "y_pred",y_pred
    # print "y_predTest", y_pred_test

# g = lambda x: x
g   = lambda x : 1/(1 + tf.exp(-x))

def nnScore(X, w, V, g):
    return tf.matmul(g((tf.matmul(X, w))), V)


# Format the datatype to suite the computation of nnscore
X = X.astype(np.float32)
X_test = data_test
X_test = X_test.astype(np.float32)
# assign the learnt weights
# wStar = hidden_layer.W
# VStar = output_layer.W
# Get weights values of fc2
wStar = model.get_weights(hidden_layer.W)
VStar = model.get_weights(output_layer.W)

print "Hideen",wStar
print VStar

train = nnScore(X, wStar, VStar, g)
test = nnScore(X_test, wStar, VStar, g)

# Access the value inside the train and test for plotting
# Create a new session and run the example
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
arrayTrain = train.eval(session=sess)
arrayTest = test.eval(session=sess)

print "Train Array:",arrayTrain
print "Test Array:",arrayTest
print "Rho Final:",temp

plt.hist(arrayTrain-temp,  bins = 25,label='Normal');
plt.hist(arrayTest-temp, bins = 25, label='Anomalies');
plt.legend(loc='upper right')
plt.title('r = %1.6f- Sigmoid Activation ' % temp)
plt.show()
