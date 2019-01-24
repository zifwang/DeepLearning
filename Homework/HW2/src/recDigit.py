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
xdata_trans = np.transpose(xdata)
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
	x_sum = np.sum(x_exp, axis = 0, keepdims = True)	# get sum the col.
	y = x_exp/x_sum		# Output

	assert(x.shape == y.shape)	# Size check

	return y

'''
	D. Using numpy, program a MLP that takes a 784-dimensional input image and calculates its
10-dimensional output. Use ReLU activation for the 2 hidden layers and softmax for the
output layer
'''

def forward_linear_propagation(x_prev,W,b):
	'''
	Implement the forward propagation function

	Arguments:
	x_prev -- A numpy array of shape (n,m), activation from previous layer
	W -- weights matrix
	b -- bias matrix
	
	Returns:
	x -- A numpy matrix equal to the softmax of x, of shape (n,m)
	'''
	b = np.reshape(b,(b.shape[0],1))
	x = np.dot(W,x_prev) + b

	# Size check
	assert(x.shape == (W.shape[0],x_prev.shape[1]))

	return x

def foward_activation_propagation(x_prev,W,b,activation):
	'''
	Implement the forward propagation function

	Arguments:
	x_prev -- A numpy array of shape (n,m), activation from previous layer
	W -- weights matrix
	b -- bias matrix
	activation -- activation function: currently has two function 'relu' and 'softmax'
	
	Returns:
	x -- A numpy matrix equal to the softmax,relu of x, of shape (n,m)
	'''

	if activation == 'relu':
		y = forward_linear_propagation(x_prev,W,b)
		x = ReLU(y)
	elif activation == 'softmax':
		y = forward_linear_propagation(x_prev,W,b)
		x = Softmax(y)

	return x

def set_largest_probability(y_input,activation):
	'''
	Implement the function which after softmax function:
		set the largest probability in the array to 1 and the other to 0

	Arguments:
	y_input -- output from the softmax function

	Returns:
	y_output -- A numpy matrix equal to the softmax,relu of x, of shape (n,m)
	'''
	if activation == 'softmax':
		y_output = np.zeros_like(y_input)  # Creat a zero array
		y_output[np.arange(len(y_input)), y_input.argmax(1)] = 1	# Set largest probability = 1 and other to 0
	else:
		y_output = y_input

	return y_output


# Calculate the output
# Output of first hidden layer
output_1 = foward_activation_propagation(xdata_trans,W1,b1,activation = 'relu')
output_2 = foward_activation_propagation(output_1,W2,b2,activation = 'relu')
y_generate = foward_activation_propagation(output_2,W3,b3,activation = 'softmax')
y_generate = np.transpose(y_generate)    # Transpose to size (10000,10)
y_output = set_largest_probability(y_generate,activation = 'softmax')

# print(y_output[0])
# print(ydata[0])

'''
	E. Compare the output with the true label from ‘ydata’. The input is correctly classified if the
position of the maximum element in the MLP output matches with the position of the 1 in
ydata. Find the total number of correctly classified images in the whole set of 10,000. You
should get 9790 correct.
'''
correct = 0 
correct_list = list() 				  # Empty list to hold correct index
error_list = list() 				  # Empty list to hold error index
assert(y_output.shape == ydata.shape) # Size check
num_rows = y_output.shape[0]
for i in range (0,num_rows):
	if(np.array_equal(y_output[i],ydata[i])):
		correct = correct + 1
		correct_list.append(i)
	else:
		error_list.append(i)

print('The correct result after MLP is: ',correct)  # 9790

'''
	F. Sample some cases with correct classification and some with incorrect classification. Visually
inspect them. For those that are incorrect, is the correct class obvious to you? You can use
matplotlib for visualization:
'''
# Plot correct cases
plt.imshow(xdata[correct_list[0]].reshape(28,28))
plt.title('Correct Figure')
plt.show()

plt.imshow(xdata[correct_list[1]].reshape(28,28))
plt.title('Correct Figure')
plt.show()

# Plot error cases
plt.imshow(xdata[error_list[0]].reshape(28,28))
plt.title('Error Figure')
plt.show()
plt.imshow(xdata[error_list[1]].reshape(28,28))
plt.title('Error Figure')
plt.show()



