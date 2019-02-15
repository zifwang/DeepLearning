from testCases_Propagation import *
from testCases_Optimization import *

from recDigit import random_mini_batches
from recDigit import momentum_init, adam_init
from recDigit import update_parameters_gd, update_parameters_momentum, update_parameters_adam









'''
    Mini batches initialization checing
'''
def test_mini_batches():
    X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
    mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

    print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    print ("shape of the 4th mini_batch_X: " + str(mini_batches[3][0].shape))
    print ("shape of the 5th mini_batch_X: " + str(mini_batches[4][0].shape))
    print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
    print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    print ("shape of the 4th mini_batch_Y: " + str(mini_batches[3][1].shape))
    print ("shape of the 5th mini_batch_Y: " + str(mini_batches[4][1].shape))
    print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:5]))


'''
    momentum_init and adam_init checking
'''
def test_momentum_init():
    print("Testing momentum initialization")
    parameters = initialize_velocity_test_case()

    v = momentum_init(parameters)
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))

    print("Done checking momentum initialization")

def test_adam_init():
    parameters = initialize_adam_test_case()

    v, s = adam_init(parameters)
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))
    print("s[\"dW1\"] = " + str(s["dW1"]))  
    print("s[\"db1\"] = " + str(s["db1"]))
    print("s[\"dW2\"] = " + str(s["dW2"]))
    print("s[\"db2\"] = " + str(s["db2"]))


'''
    Gradient descent checking
'''
def test_gd():
    # gd
    print("Testing gradient descent")
    parameters, grads, learning_rate = update_parameters_with_gd_test_case()

    parameters = update_parameters_gd(parameters, grads, learning_rate)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    print("Done checking gradient descent")

def test_momentum_gd():
    # momentum
    print("Testing momentum gradient descent")
    parameters, grads, v = update_parameters_with_momentum_test_case()

    parameters, v = update_parameters_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))

    print("Done checking momentum gradient descent")

def test_adam_gd():
    print("Testing adam gradient descent")
    parameters, grads, v, s = update_parameters_with_adam_test_case()
    parameters, v, s  = update_parameters_adam(parameters, grads, v, s, t = 2)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("v[\"dW1\"] = " + str(v["dW1"]))
    print("v[\"db1\"] = " + str(v["db1"]))
    print("v[\"dW2\"] = " + str(v["dW2"]))
    print("v[\"db2\"] = " + str(v["db2"]))
    print("s[\"dW1\"] = " + str(s["dW1"]))
    print("s[\"db1\"] = " + str(s["db1"]))
    print("s[\"dW2\"] = " + str(s["dW2"]))
    print("s[\"db2\"] = " + str(s["db2"]))
        
    print("Done checking adam gradient descent")

if __name__ == "__main__":
    # Some test_case about mini_batches preparation
    test_mini_batches()
    # print("               ")
    # # Some test_case about initialization
    # test_momentum_init()
    # print("               ")
    # test_adam_init()
    # print("               ")
    # # Some test_case about gd, momentum, and adam
    # test_gd()
    # print("               ")
    # test_momentum_gd()
    # print("               ")
    # test_adam_gd()
    # print("               ")

