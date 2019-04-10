import os
import numpy as np
import h5py
from data_generator import data_generator_main
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

密码：pm437c


def create_rnn_model(rnnModel,type,inputSize):
	"""
		Function to create my rnn neural network
		Arguments: rnnModel: keras rnnModel
				   type: string input: choose model: GRU, LSTM
				   inputSize: training input size with shape (number of data, time_length,features)
		Return: model after set up
	"""
	# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/ explain return_sequences & return_state
	# return_sequences = True to access the hidden state output for each input time step.
	# when stacking LSTM or GRU layers we have to set return_sequences = True 
	# when need to access the sequence of hidden state outputs, set return_sequences = True 
	# when predicting a sequence of outputs with a Dense output layer wrapped in a TimeDistributed layer, set return_sequences = True 
	if(type == 'GRU'):
		rnnModel.add(CuDNNGRU(units=10, kernel_initializer='random_uniform', recurrent_initializer='orthogonal', 
			bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, 
			bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 
			recurrent_constraint=None, bias_constraint=None, return_sequences=False, 
			return_state=False, stateful=False))

	if(type == 'LSTM'):
		rnnModel.add(keras.layers.CuDNNLSTM(units=10, kernel_initializer='random_uniform', recurrent_initializer='orthogonal', 
			bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, 
			bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
			return_sequences=False, return_state=False, stateful=False))

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
	X_train, y_train = load_file('./trainingData.hdf5')

	rnnModel = Sequential()
	rnnModel = create_rnn_model(rnnModel)
	# Print model summary
	rnnModel.summary()
	# train
	rnnModel.fit(x=x_train,
          y=y_train, 
          batch_size=32, 
          epochs=10, 
          verbose=1,
          validation_split=0.15,
          shuffle=True
          )

	# Save Model
    rnnModel.save('my_rnn_model.h5')
    print('Save Model to Disk')

