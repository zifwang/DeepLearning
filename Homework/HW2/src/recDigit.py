# Problem 1: Character recognition using a trained MLP
# Author: Zifan Wang

import numpy as np
import h5py
import matplotlib.pyplot as plt

'''
Problem 1: Introduction: The MNIST dataset of handwritten digits is widely used as a beginner dataset for benchmarking
machine learning classifiers. It has 784 input features (pixel values in each image) and 10 output
classes representing numbers 0–9. We have trained a MLP on MNIST, with a 784-neuron input
layer, 2 hidden layers of 200 and 100 neurons, and a 10-neuron output layer. The activation
functions used are ReLU for the hidden layers and softmax for the output layer.
'''

'''
	A. Extract the weights and biases of the trained network from mnist_network_params.hdf5.
The file has 6 keys corresponding to numpy arrays W1, b1, W2, b2, W3, b3. You may want
to check their dimensions by using the ‘shape’ attribute of a numpy array.
'''
# Load Neural Network
mnist_network_params = h5py.File('../dataset1/mnist_network_params.hdf5','r+')

# Get Weight and Baises in first hidden layer
W1 = np.asarray(mnist_network_params['W1'])
b1 = np.asarray(mnist_network_params['b1'])
# check the shape
print('W1 shape: ', W1.shape)   # (200, 784)
print('b1 shape: ', b1.shape)   # (200,) which is (200,1) use reshape to mod
# Get Weight and Baises in second hidden layer
W2 = np.asarray(mnist_network_params['W2'])
b2 = np.asarray(mnist_network_params['b2'])
# check the shape
print('W2 shape: ', W2.shape)	# (100, 200)
print('b2 shape: ', b2.shape)   # (100,)
# Get Weight and Baises in output layer
W3 = np.asarray(mnist_network_params['W3'])
b3 = np.asarray(mnist_network_params['b3'])
# check the shape
print('W3 shape: ', W3.shape)	# (10, 100)
print('b3 shape: ', b3.shape)   # (10,)


'''
	B.The file mnist_testdata.hdf5 contains 10,000 images in the key ‘xdata’ and their corresponding labels in the key ‘ydata’. Extract these. Note that each image is 784-dimensional
and each label is one-hot 10-dimensional. So if the label for an image is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
it means the image depicts a 3.
'''
# Load test data
mnist_testdata = h5py.File('../dataset1/mnist_testdata.hdf5','r+')
# Get input data
xdata = np.asarray(mnist_testdata['xdata'])
# check the shape
print('xdata shape: ', xdata.shape)	# (10000, 784)
# GEt output data (labels) which is corresponding to input data
ydata = np.asarray(mnist_testdata['ydata'])
print('ydata shape: ', ydata.shape)	# (10000, 10)


'''
	C. Write functions for calculating ReLU and softmax. These are given as
		- Relu(x) = max(0,x)
		- Softmax(x) = 【e^(x1)/sum(e^(xi))，，，，，，】
'''
def ReLU(x):
	'''
	Implement the RELU function

	Arguments: 
	x -- Output of the linear layer, of any shape

	Returns:
	y -- Post-activation parameter, of the same shape as Z
	'''
	A = np.maximum(0,x)
	assert(A.shape == x.shape) # Check shape 

	return A

def Softmax(x):
	'''
	Implement the Softmax function

	Arguments:
	x -- A numpy array of shape (n,m)
	
	Returns:
	y -- A numpy matrix equal to the softmax of x, of shape (n,m)
	'''
	x_exp = np.exp(x)	# Take exp
	x_sum = np.sum(x_exp, axis = 1, keepdims = True)	# get sum of x_exp
	y = x_exp/x_sum		# Output

	assert(x.shape == y.shape)	# Size check

	return y

'''
	D. Using numpy, program a MLP that takes a 784-dimensional input image and calculates its
10-dimensional output. Use ReLU activation for the 2 hidden layers and softmax for the
output layer
'''











