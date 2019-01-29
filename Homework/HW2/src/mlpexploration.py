# Problem 4: MLP model exploration for the data from Problem 3
# Author: Zifan Wang

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import h5py
import matplotlib.pyplot as plt

'''
    1. Load data. Load mismatched_v and mismatched_y data from Problem 3.4. You will train
    a neural network to learn f([xk, xk−1, xk−2]) = f(vk) = yk. 
'''
# Load Neural Network
inputData = h5py.File('../dataset3/lms_fun_v2.hdf5','r+')
# data from inputData and seperate to two datasets
mismatched_v = np.asarray(inputData['mismatched_v'])            # size (600, 501, 3)
mismatched_y = np.asarray(inputData['mismatched_y'])            # size (600, 501)
# Check the shape 
# print(mismatched_v.shape)
# print(mismatched_y.shape)
'''
    TODO: Reshape Data
    Reshape mismatched_v[600][501][3] to mismatched_v[300600][3]
    Reshape mismatched_y[600][501] to mismatched_y[300600]
    It is critical to keep the input and output data matched by index 
    so that f(mismatched_v[k]) = mismatched_y[k] for all k.
'''
mismatched_v_reshape = np.reshape(mismatched_v,(mismatched_v.shape[0]*mismatched_v.shape[1],mismatched_v.shape[2]))   # size: (300600, 3)
mismatched_y_reshape = np.reshape(mismatched_y,(mismatched_y.shape[0]*mismatched_y.shape[1]))                         # size: (300600, )

'''
    2. Split data. Use sklearn.model_selection.train_test_split to perform an 70/30 split
    of your data. This is very important and you will learn more about overfitting later in
    this course. You should now have 4 vectors: train_X[210420][3], train_y[210420][1],
    test_X[90180][3], and test_y[90180][1].
'''
train_X, test_X, train_y, test_y = train_test_split(mismatched_v_reshape, mismatched_y_reshape, 
                                                    test_size=0.3, train_size=0.7, random_state=42)
# Shape checker
# print(train_X.shape)        # size (210420, 3)
# print(test_X.shape)         # size (90180, 3)
# print(train_y.shape)        # size (210420,)
# print(test_y.shape)         # size (90180,)

'''
    3. Define network. Use sklearn.neural_network.MLPRegressor to define your neural network layout. 
    Use the ReLU activation function (activation = ‘relu’). 
    Use hidden_layer_sizes to define the layout. Use a maximum of 10 neurons. 
    Experiment with different configurations. Vary the depth and the number of neurons at each layer to optimize the performance
    (below). You may also adjust training parameters. The code snippet below gives suggested
    values. Note: scikit-learn infers the input and output sizes based on the data so you only need
    to specify the hidden layers. The three input and one output neuron do not count against
    your 10 neuron budget.
'''
nn = MLPRegressor(hidden_layer_sizes=(10,10),
                    activation='relu',
                    solver='adam',
                    learning_rate='adaptive',
                    max_iter=1000,
                    learning_rate_init=0.01,
                    alpha=0.01)

'''
    4. Train network. Use nn.fit to train your network. Use only the training data (train_X
    and train_y). Do not cheat and use your testing data. It may seem to improve accuracy but
    it will greatly reduce performance against new data (due to overfitting).
'''
n = nn.fit(train_X,train_y)
print("Training set score: %f" % nn.score(train_X, train_y))

'''
    5. Evaluate performance. Use nn.predict to produce predict_y using test_X. Use test_y
    only as a means to evaluate predict_y. The goal is to reduce the error between predict_y
    and test_y. You should also check the input vs output plot for subsets of the predictions
    to visually verify network is working. Experiment with different network configurations to
    improve performance.
'''
predict_y = nn.predict(test_X)
print("Test set score: %f" % nn.score(test_X, test_y))

mse = mean_squared_error(test_y,predict_y)
print("Mean Square Error after training: %f" % mse)

'''
    6. Same nn coeff.
'''
print(type(np.asarray(nn.coefs_)))
# f = h5py.File('problem4.hdf5', 'w')
# f.create_dataset('coefs',nn.coefs_)
# f.create_dataset('intercepts',nn.intercepts_)
