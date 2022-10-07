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
import seaborn as sns
import disvoice
from disvoice.phonation import Phonation
phonationf=Phonation()
from disvoice.prosody import Prosody
prosodyf=Prosody()


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
output_path = "/home/yahmadia/dataset_add/ADD_Data/trainP.pkl"
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
        features_prosody=prosodyf.extract_features_path(data_path, static=True, plots=False, fmt="npy") #fmt can also be csv, torch or txt
        print(features_prosody.shape)
        print("features_prosody:", features_prosody.shape)
        feats.append((features_prosody, label))
        with open(output_path, 'wb') as outfile:
            torch.save(feats, outfile)
