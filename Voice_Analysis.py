from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import os
import numpy as np
import pickle
import argparse
import librosa
import pandas as pd
from librosa import feature
import librosa.display
import numpy as np
from glob import glob
mysp=__import__("my-voice-analysis")


def pad(x, max_len=48000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (2, num_repeats))[:, :max_len][0]

    return padded_x



data_path = "/home/yahmadia/dataset_add/ADD_Data/ADD_train"
label_path = "/home/yahmadia/dataset_add/ADD_Data/label/train_label.txt"
output_path = "/home/yahmadia/dataset_add/ADD_Data/trainSBt.pkl"

# read in labels
class ADD(Dataset):
    filename2label = {}
    for line in open(label_path):
        line = line.split()
        filename, label = line[0], line[-1]
        filename2label[filename] = label
    feats = []
    for filepath in os.listdir(data_path):
        filename = filepath.split()[0]
        if filename not in filename2label:
            continue
        label = filename2label[filename]
        print("filename:", os.path.join(data_path, filepath))
        sig, rate = sf.read(os.path.join(data_path, filepath))
        #sig = pad(sig)
        sr = 16000
        print("rate:", rate)
        p = 'ADD_{}.wav'.format(file_name)
        c = data_path
        features_semantic= mysptotal(p,c)
        '''
                           number_ of_syllables    
                           number_of_pauses          
                           rate_of_speech             
                           articulation_rate         
                           speaking_duration       
                           original_duration       
                           balance                 
                           f0_mean               
                           f0_std                
                           f0_median              
                           f0_min                  
                           f0_max                   
                           f0_quantile25          
                           f0_quan75                
        '''
        print(features_prosody.shape)
        # rmse = np.reshape(rmse, (1, rmse.size))
        print("features_prosody:", features_prosody.shape)
        feats.append((features_semantic, label))
        with open(output_path, 'wb') as outfile:
            torch.save(feats, outfile)
