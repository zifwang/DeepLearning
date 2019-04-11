"""
    Author: Zifan Wang
    This file is used to generate more training data
"""
import json
import sys
import os
import numpy as np
import h5py


def readJsonFile(fileName):
    """
        Function to open train_files.json (contains labels of each voice)
        English is represented as 0
        Hindi is represented as 1
        Mandarin is represented as 2
        Argument: fileName: the location of train_files.json
        Return: english, hindi, and mandarin dictionary with key = audio number and value = label (english = 0, hindi = 1, mandarin = 2)
    """
    with open(fileName) as json_file:
        datas = json.load(json_file)
        # Create english, hindi, and mandarin lists to hold value
        english = {}
        hindi = {}
        mandarin = {}
        for data in datas:
            language = data[0:data.index('/')]                      # find language type: english, hindi, mandarin
            audioNum = data[data.index('-')+1:data.index('-')+3]    # find audio number
            if(language == 'english'):
                english[audioNum] = datas[data]
            if(language == 'hindi'):
                hindi[audioNum] = datas[data]
            if(language == 'mandarin'):
                mandarin[audioNum] = datas[data]
    
    return english, hindi, mandarin

def readTrainNumpy(location):
    """
        Function to open speaker-XX-file-00.npy which are audio files after MFCC feature extraction
        Argument: location: the location of audio npy files
        Return: audio dictionary with key = audio number and value = numpy array of audio
    """
    audio = {}
    for filename in os.listdir(location):
        data_npy = np.load(os.path.join(location,filename),'r')
        audioNum = filename[filename.index('-')+1] + filename[filename.index('-')+2] 
        audio[audioNum] = data_npy

    return audio

def data_generator(audio,groundTruthLabel,length):
    """
        Function to generate more training wave
        Arguments: audio: a dictionary data type contains training audio with numpy type. key = audio number, and value = audio
                   groundTruthLabel: a dictionary data type contains training audio ground truth label. key = audio number, and value = label
                   length: time_steps used in RNN
        Return: X_train a list contains numpy array data with each shape = (length，64)
                y_train a list contains ground truth labels for X_train with each shape = (length,3)
        Constrains: len(X_train) == len(y_train)
    """
    # Create two list to hold training data
    X_train = []
    y_train = []
    for key, value in audio.items():
        # features = 64 and time_steps are vary
        features,time_steps = value.shape
        # devide time_steps by length.
        numOfTraining = int(time_steps/length)
        label = groundTruthLabel[key]
        # As prof. require: I do not understand
        oneHotLabel = np.zeros((length,3))
        if(label == 0):
            oneHotLabel[:,0] = np.ones(length)
        if(label == 1):
            oneHotLabel[:,1] = np.ones(length)
        if(label == 2):
            oneHotLabel[:,2] = np.ones(length)

        for i in range (numOfTraining):
            x_data = value[:,i*length:(i+1)*length]
            x_data = np.transpose(x_data)
            feature,width = x_data.shape
            assert(feature == length and width == features)
            X_train.append(x_data)
            # y_train.append(label)             # my original understanding
            y_train.append(oneHotLabel)         # prof. requirement

    assert(len(X_train) == len(y_train))

    return X_train,y_train

def data_concatenator(X_train_1,X_train_2,X_train_3,y_train_1,y_train_2,y_train_3):
    """
        Function to add X_train together and y_train together 
        Arguments: X_train_1,X_train_2,X_train_3,y_train_1,y_train_2,y_train_3
        Return: X_train, y_train
    """
    X_train_list = X_train_1+X_train_2+X_train_3
    y_train_list = y_train_1+y_train_2+y_train_3
    assert(len(X_train_list) == len(y_train_list))
    X_train = np.asarray(X_train_list)
    y_train = np.asarray(y_train_list)

    return X_train, y_train

def to_hdf5(X_train,y_train):
    """
        Function to save X_train and y_train into hdf5 file
        with two key: X_train and y_train
        Arguments: X_train: input training data
                   y_train: ground truth of training data
        Returns: no returns 
    """
    with h5py.File('trainingData.hdf5','w') as file:
        file.create_dataset('X_train',data = X_train)
        file.create_dataset('y_train',data = y_train)
    file.close()


def data_generator_main(file_1,file_2,file_3,file_4,length):
    """
        This is the main function to generate training data and save them into a hdf5 file with two datasets: X_train & y_train
        Arguments: file_1: ground truth label file which is .json file
                   file_2: english train data which is a directory
                   file_3: hindi train data which is a directory
                   file_4: mandarin train data which is a directory
                   length: time_steps used in RNN
        Returns: no returns but generate a hdf5 file with two datasets: X_train & y_train
    """
    # Create ground truth label
    english_gt,hindi_gt,mandarin_gt = readJsonFile(file_1)  
    # Create training audio 
    english_audio = readTrainNumpy(file_2)         #(keyNum = 65, (64,n))        which means 65 numbers of data with 64 key features and last n seconds
    hindi_audio = readTrainNumpy(file_3)           #(keyNum = 21, (64,n))
    mandarin_audio = readTrainNumpy(file_4)        #(keyNum = 44, (64,n))
    # Generate more traininig data by divided original training data by selected time length
    english_X_train,english_y_train = data_generator(english_audio,english_gt,length)
    hindi_X_train,hindi_y_train = data_generator(hindi_audio,hindi_gt,length)
    mandarin_X_train,mandarin_y_train = data_generator(mandarin_audio,mandarin_gt,length)
    # X_train.shape = (286632,64,30)
    # y_train.shape = (286632,1)
    X_train, y_train = data_concatenator(english_X_train,hindi_X_train,mandarin_X_train,english_y_train,hindi_y_train,mandarin_y_train)
    # Shuffle data
    pi = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[pi]
    y_train_shuffled = y_train[pi]
    # print(X_train_shuffled.shape)
    # print(y_train_shuffled.shape)
    # Save to h5 file
    to_hdf5(X_train_shuffled,y_train_shuffled)
    print("Generate a training file: trainingData.hdf5")
    print("Use X_train key to access input of training data")
    print("Use y_train key to access output of training data")

# data_generator_main('train_files.json','./train/english','./train/hindi','./train/mandarin',30)















