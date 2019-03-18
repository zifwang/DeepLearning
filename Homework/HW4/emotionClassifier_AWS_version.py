# Author: Zifan Wang
# My PC(i7-4790,GTX970,16G RAM)
# With the ram limit, the numpy memory error would happen in the personal computer
# This type of emotion classifier is runable in AWS 

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img   # for image preprocessing
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
import cv2
import os           # input images folder
from trainingDataGenerator import get_images, get_ground_truth, generate_xy_data, h5py_training_generator

if __name__ == "__main__":
    if not os.path.exists('emotionData.h5'):
        h5py_training_generator('/home/zifwang/Desktop/train_image','/home/zifwang/Desktop/train.csv')
    # Read Train data
    trainData = h5py.File('emotionData.h5','r')
    # Read dictionary of emotions
    pickel_in = open('emotionDics.pkl','rb')
    oneHotDicts = pickle.load(pickel_in)
    xData = trainData['emotion']                        # numpy type (12993,350,350,3)
    yData = trainData['label']                          # numpy type (12993,8)
    # train_size = 70%, validation_size = 15%, test_size = 15%
    assert(xData.shape[0] == yData.shape[0])            # number of samples dimension does not match
    number,height,width,channel = xData.shape
    numberY,numclasses = yData.shape
    x_train = xData[0:int(number*0.7),:,:,:]                        # 9095
    x_validation = xData[int(number*0.7):int(number*0.85),:,:,:]    # 1949
    x_test = xData[int(number*0.85):number,:,:,:]                   # 1949
    y_train = yData[0:int(number*7),:]                              # 9095
    y_validation = yData[int(number*0.7):int(number*0.85),:]        # 1949
    y_test = yData[int(number*0.85):number,:]                       # 1949

    # Use keras image generator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        data_format='channels_last'
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Build model
    model = Sequential()
    # Convolution neural network setting
    model.add(Conv2D(32,(3,3),input_shape=(height,width,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # MLP nerual network setting
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(numclasses))
    model.add(Activation('softmax'))

    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

    # Display model Summary
    print(model.summary())
    
    # Train model
    model.fit_generator(
        train_datagen.flow(x_train,y_train,batch_size=32),
        steps_per_epoch=int(number*0.7/32),
        epochs=50,
        validation_data=validation_datagen.flow(x_validation,y_validation,batch_size=32),
        verbose=1,
    )

    # Save weigth
    model.save_weights('model.h5')
    print('Saved model to disk')

    # Test set accuracy:
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

