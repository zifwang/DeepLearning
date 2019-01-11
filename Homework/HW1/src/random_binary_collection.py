## copyright, Keith Chugg
##  EE599, 2019

#################################################
## this is a template to illustrate hdf5 files
##
## also can be used as template for HW1 problem
##################################################

import h5py
import numpy as np
import matplotlib.pyplot as plt 

DEBUG = False
DATA_FNAME = 'chugg_keith_hw1_1.hdf5'
DATA_FNAME_zifwang = 'zifan_wang_hw1_1.hdf5'

### Enter your data here...
### Be sure to generate the data by hand:
x_list_2 = [
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,1],
    [1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,1],
    [1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,1],
    [1,0,1,0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,0,1],
    [1,0,1,0,0,1,1,1,0,1,1,0,0,1,1,0,0,0,0,1],
    [1,0,1,0,0,1,1,0,1,1,1,0,0,1,1,0,0,0,0,1],
    [1,0,1,0,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,1],
    [1,0,1,0,0,1,1,0,0,1,0,0,0,1,1,0,0,0,0,1],
    [1,0,1,0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,0,1],
    [1,0,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,0,0,1],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,0,1,0,0,0,0,1],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,0,0,0,0,0,1],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,0,0,1],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,1],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,1],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,1],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,0],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,0,1,0],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,0,1],
    [1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0],
    [1,0,1,0,0,1,0,1,0,1,1,0,0,1,1,0,0,0,0,1],
]
# Change x_list_2 to np array
human_binary_2 = np.asarray(x_list_2)
###     copy-n-paste
###     use a random number generator
# x_list_2 = np.random.randint(2,size=(25,20))
###
x_list = [
    [ 0, 1, 1, 0],
    [ 1, 1, 0, 0],
    [ 0, 0, 0, 1]
]

# convert list to a numpy array...
human_binary = np.asarray(x_list)

### do some error trapping:
# print out the human_binary numpy array shape
# print(human_binary.shape) # return (3,4)
# print(human_binary_2.shape) # return (25,20)

if DEBUG:
    num_sequences = 3
    sequence_length = 4
    # assert method will check whether to arguments are equal or not. If yes, then program continuous to execute. If not, return Error and end the program.
    assert human_binary.shape[0] == num_sequences, 'Error: the number of sequences was entered incorrectly'
    assert human_binary.shape[1] == sequence_length, 'Error: the length of the seqeunces is incorrect'
    # the with statement opens the file, does the business, and close it up for us...
    with h5py.File(DATA_FNAME, 'w') as hf:
        hf.create_dataset('human_binary', data = human_binary)
    ## note you can write several data arrays into one hdf5 file, just give each a different name.

    ###################
    # Let's read it back from the file and then check to make sure it is as we wrote...
    with h5py.File(DATA_FNAME, 'r') as hf:
        hb_copy = hf['human_binary'][:]

    ### this will throw and error if they are not the same...
    np.testing.assert_array_equal(human_binary, hb_copy)
    
else:
    num_sequences = 25
    sequence_length = 20
    # assert method will check whether to arguments are equal or not. If yes, then program continuous to execute. If not, return Error and end the program.
    assert human_binary_2.shape[0] == num_sequences, 'Error: the number of sequences was entered incorrectly'
    assert human_binary_2.shape[1] == sequence_length, 'Error: the length of the seqeunces is incorrect'
    # the with statement opens the file, does the business, and close it up for us...
    with h5py.File(DATA_FNAME_zifwang, 'w') as hf:
        hf.create_dataset('human_binary_2', data = human_binary_2)
    ## note you can write several data arrays into one hdf5 file, just give each a different name.

    ###################
    # Let's read it back from the file and then check to make sure it is as we wrote...
    with h5py.File(DATA_FNAME_zifwang, 'r') as hf:
        hb_copy = hf['human_binary_2'][:]

    # print(hb_copy)
    ### this will throw and error if they are not the same...
    np.testing.assert_array_equal(human_binary_2, hb_copy)
