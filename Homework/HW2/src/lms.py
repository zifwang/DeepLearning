# Problem 3: LMS
# Author: Zifan Wang

import numpy as np
import h5py
import matplotlib.pyplot as plt


# Load dataset
inputDatasetFile = h5py.File('../dataset3/lms_fun_v2.hdf5','r')
'''
    Problem 2.
'''
# Get keys of dataset
inputDatasetFile_keys = list(inputDatasetFile.keys())
# Aissgn data from inputdataSet as numpy array to matched_10_x, matched_10_z and matched_3_x, matched_3_z
matched_10_x = np.asarray(inputDatasetFile['matched_10_x'])     # (600, 501)
matched_10_v = np.asarray(inputDatasetFile['matched_10_v'])     # (600, 501, 3)
matched_10_y = np.asarray(inputDatasetFile['matched_10_y'])     # (600, 501)
matched_10_z = np.asarray(inputDatasetFile['matched_10_z'])     # (600, 501)

matched_3_x = np.asarray(inputDatasetFile['matched_3_x'])       # (600, 501)
matched_3_v = np.asarray(inputDatasetFile['matched_3_v'])       # (600, 501, 3)
matched_3_y = np.asarray(inputDatasetFile['matched_3_y'])       # (600, 501)
matched_3_z = np.asarray(inputDatasetFile['matched_3_z'])       # (600, 501)

# a. Program the LMS algorithm with input (regressor) vn and \noisy target" zn.
#    This corresponds to the example given in lecture.
def lms(inputSignal,desiredSignal,numTaps,learningRate):
    '''
        Implement the lms algorithm.
        Arguements: 
            inputSignal:
            desiredSignal:
            numTaps:
            learningRate:
        Return:
            MSE_list:
            weight_list: 
    '''
    
    assert(inputSignal.shape[2] == numTaps)     # check whether the inputsignal has the same size with number of taps

    numSequence = inputSignal.shape[0]          # number of data sequences
    numData = inputSignal.shape[1]              # number of data in one sequence
    MSE_list = []                               # create an empty MSE list 
    weight_list = []                            # create an empty weight_list
    for i in range (0,numSequence):
        w_est = np.zeros((numTaps))             # weight initialization
        mse = np.zeros(numData)
        for j in range (0,numData):
            v = np.zeros(inputSignal.shape[2])  # init. input signal to system
            for k in range (0,inputSignal.shape[2]):
                v[k] = inputSignal[i,j,k]
            # print(np.dot(w_est,v))
            error = desiredSignal[i,j] - np.dot(w_est,v)
            mse[j] = error*error
            w_est = w_est + learningRate*error*v
        MSE_list.append(mse)
        weight_list.append(w_est)

    return MSE_list,weight_list

def averageData(dataSequence,dataLength):
    '''
        Implement the average data function and change the data in dB
        Argument:
            dataSequence: input data --> list type
            dataLength: number of data in a sequence --> int type
        Return:
            avgData: numpy type: data in dB
    '''
    size = len(dataSequence)
    avgData = np.zeros(dataLength)
    for i in range (0,dataLength):
        temp = 0
        for j in range (0,size):
            temp = temp + dataSequence[j][i]
        avgData[i] = temp/size
    avgData = 10*np.log10(avgData)
    return avgData

# Compute LMS Algorithm
MSE_list_10_005,weight_list_10_005 = lms(matched_10_v,matched_10_z,3,0.05)
MSE_list_10_015,weight_list_10_015 = lms(matched_10_v,matched_10_z,3,0.15)
# Cacluate average
MSE_avg_10_005 = averageData(MSE_list_10_005,matched_10_v.shape[1])
MSE_avg_10_015 = averageData(MSE_list_10_015,matched_10_v.shape[1])

plt.figure()
plt.plot(MSE_avg_10_005,'r',label = '10dB, eta=0.05')
plt.plot(MSE_avg_10_015,'b',label = '10dB, eta=0.15')
plt.legend()
plt.title('Learning Curve')
plt.show()


MSE_list_3_005,weight_list_3_005 = lms(matched_3_v,matched_3_z,3,0.05)
MSE_list_3_015,weight_list_3_015 = lms(matched_3_v,matched_3_z,3,0.15)
MSE_avg_3_005 = averageData(MSE_list_3_005,matched_3_v.shape[1])
plt.figure()
plt.plot(MSE_avg_10_005,'r',label = '10dB, eta=0.05')
plt.plot(MSE_avg_3_005,'b',label = '3dB, eta=0.05')
plt.legend()
plt.title('Learning Curve')
plt.show()
