# This repository is for language classification using RNN
Author: Zifan Wang.
# Enviornment: 
    OS: Ubuntu 18.04.2.LTS
    Python:3.6.7
    Keras:2.2.4
    languageClassification_gpu.py: requires GPU to run.
    languageClassification_cpu.py: does not require GPU.
This repository contains:
    1. Training Classifier File:
        languageClassification_gpu/cpu.py: use to training RNN to classify language (English, Hindi, Mandarin). This file will generate a RNN model (my_rnn_model.h5) in the same directory.
    2. Training Data Generator File:
        data_generator.py: this file is for training data generation and augmentation.
    3. Required data file: trainingData.hdf5 & train_files.json to run languageClassification_gpu/cpu.py
# Run file: python3 languageClassification_gpu/cpu.py