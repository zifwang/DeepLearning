# Problem 3: LMS
# Author: Zifan Wang

import numpy as np
import h5py
import matplotlib.pyplot as plt

COLORCONSTANT = ['r','g','b','y']           # Define color constant here

def getData(inputh5pyFile,key):
    '''
        Implement the getData function
        Arguement:
             inputh5pyFile: a hdf5 type data
             key: valid key in a hdf5 type data
        Return:
             data: numpy type
    '''
    inputDatasetFile_keys = list(inputDatasetFile.keys())
    if(key in inputDatasetFile_keys):
        return np.asarray(inputh5pyFile[key])
    else:
        print('Error: invalid input key')
        return None

# a. Program the LMS algorithm with input (regressor) vn and \noisy target" zn.
#    This corresponds to the example given in lecture.
def lms(inputSignal,desiredSignal,numTaps,learningRate):
    '''
        Implement the lms algorithm.
        Arguements: 
            inputSignal:    numpy type
            desiredSignal:  numpy type
            numTaps:        int
            learningRate:   double
        Return: all list type return 
            MSE_list:
            weight_list: 
            weight_tracking_list: 
    '''
    MSE_list = []                                       # create an empty MSE list 
    weight_final_list = []                              # create an empty final_weight_list
    weight_tracking_list = []                           # create a weight tracking list
    if(len(inputSignal.shape) == 3):
        numSequence = inputSignal.shape[0]              # number of data sequences
        numData = inputSignal.shape[1]                  # number of data in one sequence
        temp_weight_list = []
        for i in range (0,numSequence):
            w_est = np.zeros((numTaps))                 # weight initialization
            mse = np.zeros(numData)
            # print(w_est.shape)
            for j in range (0,numData): 
                error = desiredSignal[i,j] - np.dot(inputSignal[i,j],w_est)
                mse[j] = error*error
                w_est = w_est + learningRate*error*inputSignal[i,j]
                temp_weight_list.append(w_est)
            MSE_list.append(mse)
            weight_final_list.append(w_est)
            weight_tracking_list.append(temp_weight_list)
    else:
        if(numTaps>1):
            numData = inputSignal.shape[0]                  # number of data in one sequence
            w_est = np.zeros((numTaps))                     # weight initialization
            mse = np.zeros(numData)
            temp_weight_list = []
            for i in range (0,numData):
                error = desiredSignal[i] - np.dot(inputSignal[i],w_est)
                mse[i] = error*error
                w_est = w_est + learningRate*error*inputSignal[i]
                temp_weight_list.append(w_est)
            MSE_list.append(mse)
            weight_final_list.append(w_est)
            weight_tracking_list.append(temp_weight_list)
        else:
            numSequence = inputSignal.shape[0]              # number of data sequences
            numData = inputSignal.shape[1]                  # number of data in one sequence
            temp_weight_list = []
            for i in range (0,numSequence):
                w_est = np.zeros(numTaps)
                mse = np.zeros(numData)
                for j in range (0,numData):
                    error = desiredSignal[i,j] - np.dot(inputSignal[i,j],w_est)
                    mse[j] = error*error
                    w_est = w_est + learningRate*error*inputSignal[i,j]
                    temp_weight_list.append(w_est)
                MSE_list.append(mse)
                weight_final_list.append(w_est)
                weight_tracking_list.append(temp_weight_list)

    return MSE_list,weight_final_list,weight_tracking_list


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

def inputCorrelation(data):
    '''
        Implement correlation method
        Argument:
            data: a 3D numpy matrix
        Return:
            correlationList: a list of correlation matrix
    '''
    correlationList = []
    for i in range (0,data.shape[0]):
        temp = data[i]
        correlation_temp = np.dot(np.transpose(temp),temp)
        sum_correlation = np.sum(correlation_temp,axis = 1)
        correlation_temp = correlation_temp/sum_correlation
        correlationList.append(correlation_temp)
        
    return correlationList

def outputInputCorrelation(inputData,outputData):
    '''
        Implement correlation method
        Argument:
            inputData: a 3D numpy matrix -> input to the system
            outputData: a 2D numpy matrix -> desired output to the system
        Return:
            correlationList: a list of correlation matrix
    '''
    assert(inputData.shape[0] == outputData.shape[0])   # Error: Number of sequence does not match
    assert(inputData.shape[1] == outputData.shape[1])   # Error: Number of data does not match
    correlationList = []
    for i in range (0,inputData.shape[0]):
        inputTemp = inputData[i]
        outputTemp = outputData[i]
        correlation_temp = np.dot(outputTemp,inputTemp)
        sum_correlation = np.sum(correlation_temp)
        correlation_temp = correlation_temp/sum_correlation
        correlationList.append(correlation_temp)
    
    return correlationList

if __name__ == "__main__":
    # Load dataset
    inputDatasetFile = h5py.File('../dataset3/lms_fun_v2.hdf5','r')

    '''
        Problem 2.
    '''
    # Get keys of dataset

    # Aissgn data from inputdataSet as numpy array to matched_10_x, matched_10_z and matched_3_x, matched_3_z
    matched_10_x = getData(inputDatasetFile,'matched_10_x')         # (600, 501)
    matched_10_v = getData(inputDatasetFile,'matched_10_v')         # (600, `501`, 3)
    matched_10_y = getData(inputDatasetFile,'matched_10_y')         # (600, 501)
    matched_10_z = getData(inputDatasetFile,'matched_10_z')         # (600, 501)

    matched_3_x = getData(inputDatasetFile,'matched_3_x')           # (600, 501)
    matched_3_v = getData(inputDatasetFile,'matched_3_v')           # (600, 501, 3)
    matched_3_y = getData(inputDatasetFile,'matched_3_y')           # (600, 501)
    matched_3_z = getData(inputDatasetFile,'matched_3_z')           # (600, 501)

    # Compute LMS Algorithm
    MSE_list_10_005,weight_list_10_final_005,weight_tracking_10_005 = lms(matched_10_v,matched_10_z,3,0.05)
    MSE_list_10_015,weight_list_10_final_015,weight_tracking_10_015 = lms(matched_10_v,matched_10_z,3,0.15)
    # Cacluate average
    MSE_avg_10_005 = averageData(MSE_list_10_005,matched_10_v.shape[1])
    MSE_avg_10_015 = averageData(MSE_list_10_015,matched_10_v.shape[1])

    plt.figure()
    plt.plot(MSE_avg_10_005,'r',label = '10dB, eta=0.05')
    plt.plot(MSE_avg_10_015,'b',label = '10dB, eta=0.15')
    plt.legend()
    plt.title('Learning Curve')
    plt.xlabel('Updates')
    plt.ylabel('MSE (dB)')
    plt.show()


    MSE_list_3_005,weight_list_3_final_005,weight_tracking_3_005 = lms(matched_3_v,matched_3_z,3,0.05)
    MSE_list_3_015,weight_list_3_final_015,weight_tracking_3_015 = lms(matched_3_v,matched_3_z,3,0.15)
    MSE_avg_3_005 = averageData(MSE_list_3_005,matched_3_v.shape[1])
    plt.figure()
    plt.plot(MSE_avg_10_005,'r',label = '10dB, eta=0.05')
    plt.plot(MSE_avg_3_005,'b',label = '3dB, eta=0.05')
    plt.legend()
    plt.title('Learning Curve')
    plt.xlabel('Updates')
    plt.ylabel('MSE (dB)')
    plt.show()


    '''
        3. In this part, there is a single realization an x (v and y sequence, each of length 501 -
        e.g., timevarying_v, timevarying_z. In this case the data were generated using a time-
        varying linear lter { coecients in (2) actually vary with n. There is a dataset called
        timevarying_coefficents that has the 3 coecients as they change with time. Plot these
        coecients vs. time (n). Run the LMS algorithm using the x and z datasets and and vary
        the learning rate to nd a good value of  where the LMS algorithm tracks the coecient
        variations well. Plot the estimated coecients along with the true coecients for this case.
    '''
    timevarying_coefficents = getData(inputDatasetFile,'timevarying_coefficents')       # (501, 3)                
    timevarying_v = getData(inputDatasetFile,'timevarying_v')                           # (501, 3)
    timevarying_x = getData(inputDatasetFile,'timevarying_x')                           # (501,)     
    timevarying_y = getData(inputDatasetFile,'timevarying_y')                           # (503,)     
    timevarying_z = getData(inputDatasetFile,'timevarying_z')                           # (503,)
  
    '''
        There is a dataset called timevarying_coefficents that has the 3 coecients as they change with time.
        Plot these coecients vs. time (n).
    '''

    plt.figure()
    for i in range (0,timevarying_coefficents.shape[1]):
        plt.plot(timevarying_coefficents[:,i], COLORCONSTANT[i], label = 'weight'+str(i))
    plt.legend()
    plt.title('Coefficents')
    plt.xlabel('Update')
    plt.ylabel('Weight')
    plt.show()
    '''
        Run the LMS algorithm using the x and z datasets and vary the learning rate to n
        a good value of where the LMS algorithm tracks the coecientvariations well.
    '''
    # Data Modify
    timevarying_z_mod = timevarying_z[0:501,]
    MSE_timvary_005,weight_timevary_final_005,weight_tracking_timevary_005 = lms(timevarying_v,timevarying_z_mod,timevarying_v.shape[1],0.05)
    MSE_avg_timevary_005 = averageData(MSE_timvary_005,timevarying_v.shape[0])
    # Dereference weight_tracking_timevary_005 list
    timevarying_coeff_LMS = np.asarray(weight_tracking_timevary_005[0])

    plt.figure()
    for i in range (0,timevarying_coeff_LMS.shape[1]):
        plt.plot(timevarying_coeff_LMS[:,i], COLORCONSTANT[i], label = 'weight'+str(i))
    plt.legend()
    plt.title('Coefficents')
    plt.xlabel('Update')
    plt.ylabel('Weight')
    plt.show()

    mismatched_x = getData(inputDatasetFile,'mismatched_x')     # size (600, 501)
    mismatched_v = getData(inputDatasetFile,'mismatched_v')     
    mismatched_y = getData(inputDatasetFile,'mismatched_y')     # size (600, 501)

    MSE_mismatched_005,weight_mismatched_final_005,weight_tracking_mismatched_005 = lms(mismatched_v,mismatched_y,3,0.05)
    MSE_avg_mismatched_005 = averageData(MSE_mismatched_005,mismatched_v.shape[1])
    plt.figure()
    plt.plot(MSE_avg_mismatched_005,'r',label = 'mismatch, eta=0.05')
    plt.legend()
    plt.title('Learning Curve')
    plt.xlabel('Updates')
    plt.ylabel('MSE (dB)')    
    plt.show()

    correlation_v_list = inputCorrelation(mismatched_v)
    correlation_v_y_list = outputInputCorrelation(mismatched_v,mismatched_y)
    # print(correlation_v_y_list[0])
