# Author: Zifan Wang
# This version of emtion classifier is runable in my PC(i7-4790,GTX970,16G RAM)

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img   # for image preprocessing
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import regularizers
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
# import cv2
import os           # input images folder
# from trainingDataGenerator import get_images, get_ground_truth, generate_xy_data, h5py_training_generator

if __name__ == "__main__":
    # if not os.path.exists('emotionData.h5'):
        # h5py_training_generator('/home/zifwang/Desktop/train_image','/home/zifwang/Desktop/train.csv')
    # Read Data
    trainData = h5py.File('emotionData.h5','r')
    # Read Dictionary
    pickel_in = open('emotionDics.pkl','rb')
    oneHotDicts = pickle.load(pickel_in)
    X = np.asarray(trainData['emotion'])                            # numpy type (90944,128,128)
    xData = X.reshape(90944, 128, 128, 1)
    yData = np.asarray(trainData['label'])                          # numpy type (90944,8)

    # train_size = 70%, validation_size = 15%, test_size = 15%
    assert(xData.shape[0] == yData.shape[0])            # number of samples dimension does not match
    x_train, x_test, y_train, y_test = train_test_split(xData,yData,test_size=0.15,random_state=42)
    numberX,height,width,channel = xData.shape
    numberY,numclasses = yData.shape

    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    # # Build model
    model = Sequential()
    """
        function:
        keras.layers.Conv2D(filters, kernel_size, strides=(1, 1),
                          padding='valid', data_format=None, dilation_rate=(1, 1), 
                          activation=None, use_bias=True, 
                          kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                          kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                          kernel_constraint=None, bias_constraint=None)
        filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. 
                 Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
        padding: one of "valid" or "same" (case-insensitive). Note that "same" is slightly inconsistent across backends with  strides != 1,
                 same: results in padding the input such that the output has the same length as the original input.
                 valid: no padding.
                 causal: result in causal (dilated) convolutions.
        data_format: A string, one of "channels_last" or "channels_first". The ordering of the dimensions in the inputs.  "channels_last" corresponds to inputs with shape  (batch, height, width, channels) while "channels_first" corresponds to inputs with shape  (batch, channels, height, width)
        dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution.
        activation: Activation function to use (see activations).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix
        bias_initializer: Initializer for the bias vector 
        kernel_regularizer: Regularizer function applied to the kernel weights matrix
        bias_regularizer: Regularizer function applied to the bias vector 
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to the kernel matrix 
        bias_constraint: Constraint function applied to the bias vector
    """
    # Convolution neural network setting
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', input_shape=(height,width,channel), kernel_regularizer=regularizers.l2(0.001)))
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=64,kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64,kernel_size=(7,7), strides=(1,1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=128,kernel_size=(7,7), strides=(1,1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128,kernel_size=(9,9), strides=(1,1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    # model.add(Conv2D(filters=512,kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(filters=512,kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.5))

    # MLP nerual network setting
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(numclasses,kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Activation('softmax'))

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    # Display model Summary
    print(model.summary())

    # Train model
    model.fit(x=x_train,
              y=y_train, 
              batch_size=32, 
              epochs=30, 
              verbose=1,
              validation_split=0.15,
              shuffle=True
              )

    # Save weigth
    model.save('model.h5')
    print('Saved model to disk')

    # Test set accuracy:
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
