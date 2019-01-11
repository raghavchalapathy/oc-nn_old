# Keras-Tensorflow Implementation of One Class Neural Networks.


This repository provides a Keras-Tensorflow implementation of the One Class Neural Network method presented in our paper ”Anomaly Detection using One Class Neural Networks”.

# Citations and Contact.

You find a PDF of the One Class Neural Network paper at:

If you use our work, please also cite the paper:

> I’ve always been more interested
> in the future than in the past.

If you would like to get in touch, please contact .



# Abstract

>We propose a one-class neural network (OC-NN) model to detect anomalies in complex data sets. OC-NN combines the ability of >deep networks to extract a progressively rich representation of data with the one-class objective of creating a tight >envelope around normal data. The OC-NN approach breaks new ground for the following crucial reason: data representation in >the hidden layer is driven by the OC-NN objective and is thus customized for anomaly detection. This is a departure from >other approaches which use a hybrid approach of learning deep features using an autoencoder and then feeding the features >into a separate anomaly detection method like one-class SVM (OC-SVM). The hybrid OC-SVM approach is sub-optimal because it is >unable to influence representational learning in the hidden layers. A comprehensive set of experiments demonstrate that on >complex data sets (like CIFAR and GTSRB), OC-NN performs on par with state-of-the-art methods and outperformed conventional >shallow methods in some scenarios.



# Installation

This code is written in Python 3.7 and requires the packages listed in requirements.txt.

Clone the repository to your local machine and directory of choice:



# Running experiments

We currently have implemented the MNIST (http://yann.lecun.com/exdb/mnist/) and CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html) datasets and simple LeNet-type networks.



## MNIST Example



## CIFAR-10 Example



## Sample Results 


### MNIST
Example of the  most normal (left) and  most anomalous (right) test set examples per class on MNIST according to One Class Neural Networks and Robust Convolution Autoencoder (RCAE) anomaly scores.



### CIFAR-10
Example of the  most normal (left) and  most anomalous (right) test set examples per class on CIFAR-10 according to One Class Neural Networks and Robust Convolution Autoencoder (RCAE) anomaly scores.


# License
MIT


# Disclosure
This implementation is based on the repository https://github.com/lukasruff/Deep-SVDD, which is licensed under the MIT license. The Deep SVDD repository is an implementation of the paper "Deep One-Class Classification by Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Deecke, Lucas and Siddiqui, Shoaib A. and Binder, Alexander and M{\"u}ller, Emmanuel and Kloft, Marius

