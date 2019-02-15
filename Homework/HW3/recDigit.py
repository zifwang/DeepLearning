'''
    This program aims to recognize digits by MLP
    Author: Zifan Wang
    Data: 02/14/2019
'''
import numpy as np
from sklearn.model_selection import train_test_split        # This function is used to sperate the training and validation data
import h5py
import math
import matplotlib.pyplot as plt



'''
    Data Preparation:
        In this training MLP: use mnist_traindata.hdf5 file which contains 60,000 images in the key 'xdata',
        and their corresponding labels in the key 'ydata'. 
        Split them into 50,000 images for training and 10,000 for validation.

        Mini_batches method

        test_data method
'''
def dataPrep(filename):
    '''
        Implement the function to read in the data file.
        Keys of the data file: 'xdata' and 'ydata'
        Argument: the pwd/filename of the object
        Returns: x_train, x_validation, y_train, y_validation
    '''
    mnist_traindata = h5py.File(filename,'r')
    keys = list(mnist_traindata.keys())
    xData = np.asarray(mnist_traindata[keys[0]])        # xdata is in the keys[0]
    yData = np.asarray(mnist_traindata[keys[1]])        # xdata is in the keys[1]
    x_train, x_validation, y_train, y_validation = train_test_split(xData,yData,
                                                                    test_size = 0.16666,
                                                                    random_state = 42)

    return x_train, x_validation, y_train, y_validation

def random_mini_batches(x, y, mini_batch_size = 100, seed = 0):
    '''
        Implement the function to create random minibatches from input train_x and train_y
        Arguments:
            x -- Input training data: train_x.shape == (input size, number of samples)
            y -- GroundTruth Training data: train_y.shape == (output size, number of samples)
            mini_batch_size -- size of the mini-batches, integer
        Returns: 
            mini_batch (a list): (mini_batch_x, mini_batch_y)
    '''
    mini_batches = []       # return list
    np.random.seed(seed)
    # number of training samples 
    numSamples = x.shape[1]  
    # output data shape in one sample
    ySize = y.shape[0]

    # Data shuffle
    permutation = list(np.random.permutation(numSamples))
    shuffled_X = x[:, permutation]
    shuffled_Y = y[:, permutation].reshape((ySize,numSamples))
    
    # number of complete mini batches
    num_complete_minibatches = math.floor(numSamples/mini_batch_size)
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handle reset of mini batch (last mini_batch < mini_batch_size)
    if numSamples % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : numSamples]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : numSamples]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def test_data_init(filename):
    test_data = h5py.File(filename,'r')
    keys = list(test_data.keys())
    xdata = np.asarray(test_data[keys[0]])
    ydata = np.asarray(test_data[keys[1]])

    return xdata,ydata


'''
    Define activation function section:
        1. relu 
        2. tanh 
        3. softmax -- used in the last layer
'''
def relu(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 0, keepdims= True)
    
    return x_exp/x_sum


'''
    Derivative of activation function section:
        1. relu
        2. tanh
        3. softmax
        4. cross_entropy with softmax
'''
def relu_backward(dA):
    return (dA > 0).astype(int)

def tanh_backward(dA):
    return 1 - tanh(dA)*tanh(dA)

def softmax_backward(dA):
    s = dA.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def cross_entropy_softmax(x,y):
    m = y.shape[1]
    return 1./m * (x - y)


'''
    Coss function section：
        1. Cross-entropy Cost
        2. Quadratic cost (MSE)
'''
def crossEntropy_cost(x,y):
    '''
        Implement the cross entropy cost function
        Arguments: x -- output from fully connected layer
                   y -- ground truth label
                   x and y have the same shape
        Return: crossEntropyCost -- value of the cost function
    '''
    m = y.shape[1]
    cost = -(np.multiply(np.log(x),y) + np.multiply(np.log(1-x),1-y))
    crossEntropyCost = 1./m * np.sum(cost)

    return crossEntropyCost

def mse_cost(x,y):
    '''
        Implement the MSE function
        Arguments: x -- output from fully connected layer
                   y -- ground truth label
                   x and y have the same shape
        Return: cost -- value of the cost function
    '''
    m = y.shape[1]
    cost = 1/m * np.sum(np.multiply(y-x,y-x), axis = 1)

    return cost

'''
    initalization function sections: implement two initalization functions:
    1. he initalization
    2. random initalization
'''
def he_init(layerDims):
    '''
        Implement the weight and bias initalzation function use HE init.
        Arguments:
            layerDims (array or list type) -- contains the dimensions of each layer in nn
        Return: [784,200,100,10]
            dicts (dictionary type) -- contains the weight and bias: 'W1', 'b1', 'W2', 'b2', ... , 'Wn', 'bn'
                                                               W1 -- weight matrix of shape (layerDims[1], layerDims[0])
                                                               b1 -- bias vector of shape (layerDims[1], 1) 
                                                               W2 -- weight matrix of shape (layerDims[2], layerDims[1])
                                                               b2 -- bias vector of shape (layerDims[2], 1) 
                                                               W3 -- weight matrix of shape (layerDims[3], layerDims[2])
                                                               b3 -- bias vector of shape (layerDims[3], 1) 
    '''
    np.random.seed(3)                   # random number generator
    dicts = {}

    for l in range(1, len(layerDims)):
        dicts['W' + str(l)] = np.random.randn(layerDims[l],layerDims[l-1])*np.sqrt(2/layerDims[l-1])
        dicts['b' + str(l)] = np.zeros((layerDims[l],1))
 
    return dicts

def random_init(layerDims):
    '''
        Implement the weight and bias initalzation function use random init.
        Arguments:
            layerDims (array or list type) -- contains the dimensions of each layer in nn
        Return:
            dicts (dictionary type) -- contains the weight and bias: 'W1', 'b1', 'W2', 'b2', ... , 'Wn', 'bn'
                                                               W1 -- weight matrix of shape (layerDims[1], layerDims[0])
                                                               b1 -- bias vector of shape (layerDims[1], 1) 
    '''
    np.random.seed(3)                   # random number generator
    dicts = {}

    for l in range(1, len(layerDims)):
        dicts['W' + str(l)] = np.random.randn(layerDims[l],layerDims[l-1])*0.03
        dicts['b' + str(l)] = np.zeros((layerDims[l],1))
 
    return dicts

def parameters_init(layerDims,initialization):
    '''
        Implement the weight and bias initalzation function
        Arguments:
            layerDims (array or list type) -- contains the dimensions of each layer in nn
            initialzation (string type) -- method used to initialze the weight and bias.
                                         1. initialzation = 'random'.   2. initialzation = 'he'
        Return:
            parameters (dictionary type) -- contains the weight and bias
    '''
    parameters = {}
    
    # Check whether initialization is valid
    assert(initialization == 'he' or initialization == 'random')    # Error: unrecognize initalization
 
    if(initialization == 'he'):
        parameters = he_init(layerDims)
    elif(initialization == 'random'):
        parameters = random_init(layerDims)

    return parameters


'''
    Optimizers section: Implement two optimizers.
    1. Momentum optimizer
    2. Adam optimizer
'''
def momentum_init(parameters):
    '''
        Momentum optimizer
        Argument: 
            parameters (dictionary type) -- with keys: 'W1','b1',...,'Wn','bn'
        Return:
            momentumDict (dictionary type): with keys: 'dW1','db1',...,'dWn','dbn'
                                            and init. corresponding value to zero
    '''
    momentumDict = {}
    
    for l in range(len(parameters)//2):
        momentumDict["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        momentumDict["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return momentumDict

def adam_init(parameters):
    '''
        Adam optimizer
        Argument: 
            parameters (dictionary type) -- with keys: 'W1','b1',...,'Wn','bn'
        Returns: v(the exponentially weighted average of the gradient)
                 s(the exponentially weighted average of the squared gradient)
            v (dictionary type): with keys: 'dW1','db1',...,'dWn','dbn'
                                            and init. corresponding value to zero
            s (dictionary type): with keys: 'dW1','db1',...,'dWn','dbn'
                                            and init. corresponding value to zero
    '''
    v = {}
    s = {}
    for l in range(len(parameters)//2):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return v,s

'''
    Update paramenter section:
        1. Stochastic Gradient Descent
        2. Momentum optimizer Gradient Descent
        3. Adam optimizer Gradient Descent
'''
def update_parameters_gd(parameters,gradients,learning_rate):
    '''
        Stochastic Gradient Descent:
        Arguements:
            parameters (dictionary type): contains weight and bias before updating
            gradients (dictionary type): contains derivative of weight and bias
            learning_rate (double type): learning rate
        returns:
            parameters (dictionary type): contains updated weight and bias
    '''
    for l in range(len(parameters)//2):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * gradients['dW' + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * gradients['db' + str(l+1)]
    
    return parameters

def update_parameters_momentum(parameters, gradients, momentumDict, beta, learning_rate):
    '''
        Momentum optimizer Gradient Descent:
        Arguements:
            parameters (dictionary type): contains weight and bias before updating
            gradients (dictionary type): contains derivative of weight and bias
            momentumDict (dictionary type): contains current velocities
            beta: the Momentum Parameter
            learning_rate (double type): learning rate
        returns:
            parameters (dictionary type): contains updated weight and bias
            momentumDict (dictionary type): contains updated velocities
    '''
    for l in range(len(parameters)//2):
        # velocities
        momentumDict["dW" + str(l+1)] = beta*momentumDict["dW" + str(l+1)]+(1-beta)*gradients['dW' + str(l+1)]
        momentumDict["db" + str(l+1)] = beta*momentumDict["db" + str(l+1)]+(1-beta)*gradients['db' + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*momentumDict["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*momentumDict["db" + str(l+1)]
    
    return parameters, momentumDict

def update_parameters_adam(parameters, gradients, v, s, t,
                           learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    '''
        Momentum optimizer Gradient Descent:
        Arguements:
            parameters (dictionary type): contains weight and bias before updating
            gradients (dictionary type): contains derivative of weight and bias
            v (dictionary type): contains gradient
            s (dictionary type): contains squared gradient
            t: time
            beta1: Exponential decay hyperparameter for the first moment estimates 
            beta2: Exponential decay hyperparameter for the second moment estimates 
            learning_rate (double type): learning rate
            epsilon -- hyperparameter preventing division by zero in Adam updates
        returns:
            parameters (dictionary type): contains updated weight and bias
            v (dictionary type): contains updated gradient
            s (dictionary type): contains updated squared gradient
    '''
    v_bias_correction = {}
    s_bias_correction = {}

    for l in range(len(parameters)//2):
        # Update gradient and square gradient
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1-beta1) * gradients['dW' + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1) * gradients['db' + str(l+1)]
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * gradients['dW' + str(l+1)]**2
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * gradients['db' + str(l+1)]**2
        # Compute the bias corrections of v and s
        v_bias_correction["dW" + str(l+1)] = v["dW" + str(l+1)] / (1-beta1 ** t)
        v_bias_correction["db" + str(l+1)] = v["db" + str(l+1)] / (1-beta1 ** t)
        s_bias_correction["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-beta2 ** t)
        s_bias_correction["db" + str(l+1)] = s["db" + str(l+1)] / (1-beta2 ** t)

        # Update the parameter
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_bias_correction["dW" + str(l+1)] / (s_bias_correction["dW" + str(l+1)]**0.5 + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_bias_correction["db" + str(l+1)] / (s_bias_correction["db" + str(l+1)]**0.5 + epsilon)
        
    return parameters, v, s

'''
    Regularizer section
        L2
'''


'''
    Propagation section:
        1. forward propagation
        2. backward propagation
'''
def forward_propagation(x,parameters,activations):
    '''
        This function is for the forward propagation
        Arguments:
            x -- input dataset(in shape(inputSize,numOfSamples))
            parameters -- weight and bias
            activations -- a list of activation methods 
        Returns:
            cache_dicts -- contains all outputes of wx+b, outputes of activation, weightes, and biases
            keys(z,a,W,b)
    '''
    cache_dicts = {}
    cache_dicts['a0'] = x
    a = x
    for i in range (len(parameters)//2):
        z = np.dot(parameters["W"+str(i+1)],a) + parameters["b"+str(i+1)]   # linear
        cache_dicts["z"+str(i+1)] = z          # append output of wx+b
        # z to activation function 
        if(activations[i] == 'relu'):
            a = relu(z)
            cache_dicts["a"+str(i+1)] = a
        if(activations[i] == 'tanh'):
            a = tanh(z)
            cache_dicts["a"+str(i+1)] = a
        if(activations[i] == 'softmax'):
            a = softmax(z)
            cache_dicts["a"+str(i+1)] = a
        cache_dicts["W"+str(i+1)] = parameters["W"+str(i+1)]
        cache_dicts["b"+str(i+1)] = parameters["b"+str(i+1)]

    # # Use three layer first
    # # retrieve parameters
    # W1 = parameters["W1"]
    # b1 = parameters["b1"]
    # W2 = parameters["W2"]
    # b2 = parameters["b2"]
    # W3 = parameters["W3"]
    # b3 = parameters["b3"]
    
    # # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    # z1 = np.dot(W1, x) + b1
    # a1 = relu(z1)
    # z2 = np.dot(W2, a1) + b2
    # a2 = relu(z2)
    # z3 = np.dot(W3, a2) + b3
    # a3 = softmax(z3)
    
    # cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    
    return cache_dicts["a"+str(len(parameters)//2)],cache_dicts
    # return a3,cache


def backward_propagation(x,y,cache_dicts,activations):
    '''
        This function is for the backward propagation
        Arguments:
            x -- input dataset(in shape(inputSize,numOfSamples))
            y -- ground truth
            cache_dicts -- cache_dicts output from forward propagation
            activations -- a list of activation methods 
        Returns:
            gradients -- a gradient dictionary
    '''
    gradients = {}
    for i in range(len(activations)-1,-1,-1):
        if(activations[i] == 'softmax'):
            dz = cross_entropy_softmax(cache_dicts["a"+str(i+1)],y)
            dW = np.dot(dz, cache_dicts["a"+str(i)].T)
            db = np.sum(dz,axis=1, keepdims=True)
            gradients["dz"+str(i+1)] = dz
            gradients["dW"+str(i+1)] = dW
            gradients["db"+str(i+1)] = db
        if(activations[i] == 'relu'):
            da = np.dot(cache_dicts["W"+str(i+2)].T,gradients["dz"+str(i+2)])
            dz = np.multiply(da,relu_backward(cache_dicts["a"+str(i+1)]))
            dW = np.dot(dz, cache_dicts["a"+str(i)].T)
            db = np.sum(dz,axis=1, keepdims=True)
            gradients["da"+str(i+1)] = da
            gradients["dz"+str(i+1)] = dz
            gradients["dW"+str(i+1)] = dW
            gradients["db"+str(i+1)] = db
        if(activations[i] == 'tanh'):
            da = np.dot(cache_dicts["W"+str(i+2)].T,gradients["dW"+str(i+2)])
            dz = np.multiply(da,tanh_backward(cache_dicts["a"+str(i+1)]))
            dW = np.dot(dz, cache_dicts["a"+str(i)].T)
            db = np.sum(dz,axis=1, keepdims=True)
            gradients["da"+str(i+1)] = da
            gradients["dz"+str(i+1)] = dz
            gradients["dW"+str(i+1)] = dW
            gradients["db"+str(i+1)] = db

    # m = x.shape[1]
    # (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache_dicts
    
    # dz3 = 1./m * (a3 - y)
    # dW3 = np.dot(dz3, a2.T)
    # db3 = np.sum(dz3, axis=1, keepdims = True)

    # da2 = np.dot(W3.T, dz3)
    # dz2 = np.multiply(da2, np.int64(a2 > 0))
    # dW2 = np.dot(dz2, a1.T)
    # db2 = np.sum(dz2, axis=1, keepdims = True)
    # print('da2.shape:', da2.shape)                #100,100
    # print('a2.shape:', np.int64(a2 > 0).shape)    #100,100


    # da1 = np.dot(W2.T, dz2)
    # dz1 = np.multiply(da1, np.int64(a1 > 0))
    # dW1 = np.dot(dz1, x.T)
    # db1 = np.sum(dz1, axis=1, keepdims = True)
    # print('da1.shape:', da1.shape)                #200,100
    # print('a1.shape:', np.int64(a1 > 0).shape)    #200,100
    
    # gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
    #              "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
    #              "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    
    return gradients

'''
    prediction method
'''
def predict(x,parameters,activations):
    '''
        Predict the label of a single test example (image).

        Arguments:
            x : numpy.array
        Returns: int
            Predicted label of example (image).

    '''
    a3, _ = forward_propagation(x,parameters,activations)
    return a3

'''
    Validation method
'''
def validation(x,y,parameters,activations):
    predict_result = predict(x,parameters,activations)
    numCorrect = 0
    y = y.T
    predict_result = predict_result.T
    for i in range(y.shape[0]):
        if(np.argmax(y[i]) == np.argmax(predict_result[i])):
            numCorrect = numCorrect + 1
    return numCorrect/y.shape[0]

'''
    Training accuracy
'''
def train_accuracy(train_result,groundTruth):
    numCorrect = 0
    groundTruth = groundTruth.T
    train_result = train_result.T
    for i in range(groundTruth.shape[0]):
        if(np.argmax(groundTruth[i]) == np.argmax(train_result[i])):
            numCorrect = numCorrect + 1

    return numCorrect/groundTruth.shape[0]



'''
    Training process
'''
def model_train(X, Y, x_val, y_val, layersDims, activations, initialization, optimizer, learning_rate = 0.0007, learning_rate_decay = True, mini_batch_size = 100, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 50, verbose = True):
    '''
    '''
    # L = len(layersDims)       # number of layers in nn
    costs = []                  # list to track the cost
    train_accuracies = []       # list to track the train accuracies
    val_accuracies = []         # list to track the val accuracies
    t = 0                       # adam t parameter
    seed = 10

    # Initialize parameters
    parameters = parameters_init(layersDims,initialization)

    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = momentum_init(parameters)
    elif optimizer == "adam":
        v, s = adam_init(parameters)

    for epoch in range(num_epochs):
        # Define the random minibatches. 
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        if(learning_rate_decay and epoch == num_epochs//2):
            learning_rate = learning_rate/2

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            train_result, caches = forward_propagation(minibatch_X, parameters,activations)

            # Compute cost
            cost = crossEntropy_cost(train_result, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches, activations)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2, epsilon)

        # Calculate train_accuracy and val_accuracy after each epoch
        train_accuracy = validation(X,Y,parameters,activations)
        train_accuracies.append(train_accuracy)
        val_accuracy = validation(x_val,y_val,parameters,activations)
        val_accuracies.append(val_accuracy)
        costs.append(cost)

        if(verbose):
            # Print the cost 
            print("Epoch %i/%i"%(epoch,num_epochs))
            print("-loss: %f - training_acc: %f - validation_acc: %f"%(cost,train_accuracy,val_accuracy))

    return parameters, costs, train_accuracies, val_accuracies



if __name__ == "__main__":
    # Get trainning and validation data
    '''
        x_train.shape = (50000, 784)
        x_validation.shape = (10000, 784)
        y_train.shape = (50000, 10)
        y_validation.shape = (10000, 10)
        
    '''
    x_train, x_validation, y_train, y_validation = dataPrep('/Users/zifwang/Desktop/mnist_traindata.hdf5')
    x_train = x_train.T
    x_validation = x_validation.T
    y_train = y_train.T
    y_validation = y_validation.T
    # Define layers and activations here
    layers_dims = [x_train.shape[0], 200, 100, 10]
    activations = ['relu', 'relu', 'softmax']
    parameters, cost, train_acc, val_acc = model_train(x_train, y_train, x_validation, y_validation, layers_dims, 
                                                        activations, initialization = 'he', optimizer = "adam",
                                                        learning_rate = 0.007, learning_rate_decay = True, mini_batch_size = 100, 
                                                        beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 10, verbose = True)

    # testing data
    x_test, y_test = test_data_init('mnist_testdata.hdf5')
    test_acc = validation(x_test.T,y_test.T,parameters,activations)
    print(test_acc)

    # Save parameter


    # plot the cost
    # plt.plot(cost)
    # plt.ylabel('cost')
    # plt.xlabel('epochs')
    # plt.title("Learning rate = " + str(0.0007))
    # plt.show()

    # # plot the accuracy
    # plt.plot(train_acc)
    # plt.plot(val_acc)
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()


