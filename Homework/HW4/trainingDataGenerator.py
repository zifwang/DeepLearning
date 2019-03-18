import numpy as np
import pandas as pd
import os 
import h5py
import pickle
import cv2

def get_images(location):
    """
        Function to load images from givien directory
        Arguments: training image directory
        Return: a image dict: key->image's name. value: resized image (size 350*350*3)
    """
    images = {}
    for filename in os.listdir(location):
        img = cv2.imread(os.path.join(location,filename))   # read image by opencv library
        if img is not None:
            img_mod = cv2.resize(img,(350,350))             # resize image to 350*350*3
            images[filename] = img_mod
    return images

def get_ground_truth(location):
    """
        Function to load images label from givien directory
        Arguments: ground truth csv file location
        Returns: a one hot labeled ground truth label dictionary: key->image's name. value: one hot label ground truth
                 a one hot labeled emotion dictionary: key-> emotion. value: corresponding one hot label
    """
    ground_truth_dic = {}                                    # returned ground truth dictionary
    emotionDic = {}                                          # returned emtion dictionary

    # get data frame by pandas library
    df = pd.read_csv(location,names=['name','emotions'],header=None)
    df_emotion = df.loc[:,'emotions']
    emotions = list(df_emotion)
    emotionList = []
    # Generate one hot label encoder
    for emotion in emotions:
        if emotion not in emotionList:
            emotionList.append(emotion)
    oneHotEncoder = np.identity(len(emotionList))
    
    # Generate emotionDic
    i = 0                                                      # index 
    for emotion in emotionList:
        emotionDic[emotion] = oneHotEncoder[i,:]
        i += 1
    
    # Generate ground_truth_dic using one hot labeling
    for _,row in df.iterrows():
        ground_truth_dic[row['name']] = emotionDic[row['emotions']]
    
    return ground_truth_dic, emotionDic

def generate_xy_data(x,y):
    """
        Function to convert the input data and its corresponding ground truth label data in dictionary type to numpy array
        Arguments: input data(dictionary type), ground truth label(dictionary type)
        Returns: input data(numpy type), ground truth label(numpy type)
    """
    x_list = []
    y_list = []
    for key,value in x.items():
        x_list.append(value)
        y_list.append(y[key])

    # Change to numpy type
    inputData = np.asarray(x_list)
    labelData = np.asarray(y_list)
    
    return inputData, labelData

def h5py_training_generator(directory_1,dictionary_2):
    """
        Function to generate two file: one for training data, another for label dictionary
        Input: image directory and label directory
        Training Data: filename->'emotionData.h5' with two argument: 'emotion' & 'label'
        label Dictionary: filename->'emotionDics.pkl' with key->emotions. value->one hot label
    """
    images = get_images(directory_1)   # most image are in 3*350*350
    labels, emotionDic = get_ground_truth(dictionary_2)
    xData, yData = generate_xy_data(images,labels)
    hf = h5py.File('emotionData.h5','w')
    hf.create_dataset('emotion',data = xData)
    hf.create_dataset('label',data = yData)
    hf.close()
    f = open('emotionDics.pkl','wb')
    pickle.dump(emotionDic,f)
    f.close()

h5py_training_generator('/home/zifwang/Desktop/train_image','/home/zifwang/Desktop/train.csv')