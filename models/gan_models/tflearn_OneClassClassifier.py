from data_fetch import prepare_usps_mlfetch
[Xtrue,Xlabels] = prepare_usps_mlfetch()
data = Xtrue
target = Xlabels
from sklearn.cross_validation import train_test_split
train_data, test_data, train_target, test_target = train_test_split(data, target, train_size = 0.8)
train_data.shape


# We learn the digits on the first half of the digits
data_train, targets_train = train_data,train_target
# Now predict the value of the digit on the second half:
data_test, targets_test = test_data,test_target

import tflearn
from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression,oneClassNN
from tflearn.metrics import binary_accuracy_op

import tflearn.variables


#Training examples
X = data_train
# Y = [[0], [0], [0], [0]]
Y = targets_train
# Y = list(Y)
Y = Y.tolist()
Y= [[i] for i in Y]

# For testing the algorithm
X_test = data_test
Y_test = targets_test
Y_test= Y_test.tolist()
Y_test= [[i] for i in Y_test]

m,n = data_train.shape
No_of_inputNodes = n
No_of_hiddenNodes=n
print "No_of_hiddenNodes",No_of_hiddenNodes



input_layer = input_data(shape=[None, No_of_inputNodes]) #input layer of size 2
hidden_layer = fully_connected(input_layer , No_of_hiddenNodes, activation='tanh',name="hiddenLayer_Weights",weights_init="normal") #hidden layer of size 2
output_layer = fully_connected(hidden_layer, 1, activation='sigmoid',name="outputLayer_Weights") #output layer of size 1

# Hyper parameters for the one class Neural Network
v = 0.4
# rho=0.3

# rho=input_data(shape=[None, 1])

import tflearn.variables as va
rho = va.variable(name='rho',shape=[])


#use Stohastic Gradient Descent and Binary Crossentropy as loss function
oneClassNN = oneClassNN(output_layer,v,rho,hidden_layer,output_layer
                        ,optimizer='sgd', loss='OneClassNN_Loss', learning_rate=5)

model = DNN(oneClassNN,tensorboard_verbose=3)

#fit the model
model.fit(X, Y, n_epoch=50, show_metric=True);


#predict all examples
print ('Expected:  ', [i for i in Y_test])
print ('Predicted: ', [i for i in model.predict(X_test)])

# y_pred = model.predict(X_test) # Apply some ops
# acc_op = binary_accuracy_op(y_pred, Y_test)
#
vars = tflearn.variables.get_all_trainable_variable()
for v in vars:
    print v

# Calculate accuracy by feeding data X and labels Y
# binary_accuracy = sess.run(acc_op, feed_dict={input_data: X, Y_test: Y})
