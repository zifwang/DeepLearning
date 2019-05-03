import os
import numpy as np
import h5py
from data_generator import data_generator_main
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM, GRU
from keras.layers import CuDNNLSTM, CuDNNGRU

def load_file(fileName):
	"""
		Function to load training data file
		Arguments: fileName: training data file's name
		Returns: X_train: input training data 
				 y_train: output training data
	"""
	with h5py.File(fileName) as file:
		X_train = np.asarray(file['X_train'])
		y_train = np.asarray(file['y_train'])
	file.close()

	return X_train,y_train

def create_rnn_model(rnnModel,type,inputSize):
	"""
		Function to create my rnn neural network
		Arguments: rnnModel: keras rnnModel
				   type: string input: choose model: GRU, LSTM
				   inputSize: training input size with shape (time_length,features)
		Return: model after set up
	"""
	# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/ explain return_sequences & return_state
	# return_sequences = True to access the hidden state output for each input time step.
	# when stacking LSTM or GRU layers we have to set return_sequences = True 
	# when need to access the sequence of hidden state outputs, set return_sequences = True 
	# when predicting a sequence of outputs with a Dense output layer wrapped in a TimeDistributed layer, set return_sequences = True 
	if(type == 'GRU'):
		rnnModel.add(GRU(units=32, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', 
                        recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, 
                        activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
                        dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, go_backwards=False, 
                        stateful=False, unroll=False, reset_after=False, input_shape=inputSize))

	if(type == 'LSTM'):
		rnnModel.add(LSTM(units=32, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', 
                    recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, 
                    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
                    dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, return_state=False, go_backwards=False, 
                    stateful=False, unroll=False, input_shape=inputSize))

	rnnModel.add(Dense(3,activation='softmax'))
	rnnModel.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

	return rnnModel


if __name__ == '__main__':
	# Check whether trainingData.hdf5 exists. If not, generate it. If yes, load file.
	exists = os.path.exists('trainingData.hdf5')
	if not exists: 
		print('trainingData.hdf5 does not exist yet. Generating trainingData.hdf5 file ......')
		data_generator_main('train_files.json','./train/english','./train/hindi','./train/mandarin',1000)
	else:
		print('trainingData.hdf5 exists.')
	X, y = load_file('./trainingData.hdf5')
	X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.15,random_state=42,shuffle=True)
	print(X_train.shape)
	print(y_train.shape)
	numofData,time_length,features = X_train.shape

	rnnModel = Sequential()
	rnnModel = create_rnn_model(rnnModel,'LSTM',(time_length,features))
	# Print model summary
	rnnModel.summary()
	# train
	rnnModel.fit(x=X_train,
          y=y_train, 
          batch_size=128, 
          epochs=50, 
          verbose=1,
          validation_data=(X_val,y_val),
          shuffle=True
          )
    # Save Model
	rnnModel.save('my_rnn_model.h5')
	print('Save Model to Disk')

