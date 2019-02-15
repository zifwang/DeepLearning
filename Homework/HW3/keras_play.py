import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import matplotlib.pyplot as plt



'''
    Data preparation
'''
inputData = h5py.File('binary_random_sp2019.hdf5','r')
keys = list(inputData.keys())
humanData = np.asarray(inputData[keys[0]])                                      #(2400,20)
machineData = np.asarray(inputData[keys[1]])                                    #(2400,20)
humanOutput = np.ones((humanData.shape[0],1))                                   # Create human label: 1
machineOutput = np.zeros((machineData.shape[0],1))                              # Create computer label: 0
train_input = np.concatenate((humanData, machineData), axis=0)                  #(4800,20)
train_groundTruth = np.concatenate((humanOutput, machineOutput), axis=0)        #(4800,1)
# train and validation data split
x_train, x_validation, y_train, y_validation = train_test_split(train_input,train_groundTruth, test_size = 0.2, random_state = 10)

# Creat keras model
model = Sequential()
model.add(Dense(units=20, activation='relu', input_dim=humanData.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=25, batch_size=16, verbose = 1 , validation_data=(x_validation, y_validation))

model.save('hw3p3.h5') 

# Print accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
