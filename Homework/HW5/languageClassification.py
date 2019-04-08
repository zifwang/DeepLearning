import json
import numpy as np






def openJsonFile(fileName):
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
            if(language == mandarin):
                mandarin[audioNum] = datas[data]
    
    return english, hindi, mandarin

def 


english_gt,hindi_gt,mandarin_gt = openJsonFile('train_files.json')  # Create ground truth label
