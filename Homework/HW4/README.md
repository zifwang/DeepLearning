# This repository is for emotion classification using CNN and Keras. 
# Enviornment: 
    OS: Ubuntu 18.04.2.LTS
    Python:3.6.7
    Keras:2.2.4
This repository contains:
    1. Training Classifier File:
        emotionClassifier_AWS_version.py: this code is used to train emotion classifier in AWS. (.h5 input)
        emotionClassifier_myPC_version_32.py: this code is used to train emotion classifier by using 32*32 training image. (.h5 input)
        emotionClassifier_myPC_version_64.py: this code is used to train emotion classifier by using 64*64 training image. (.h5 input)
        emotionClassifier_myPC_version_128.py: this code is used to train emotion classifier by using 128*128 training image. (.h5 input)
        hw4EmotionClassifier.py: this code is used to train emotion classifier using 32*23 training image with .npz data compression.
    2. Homework Turn in File: 
        hw4p1v11tpl.py: this file is for homework turn in.
    3. Training Data Generator File:
        trainingDataGenerator.py: this file is for training data generation and augmentation.
# Run file: python3 filename.py
