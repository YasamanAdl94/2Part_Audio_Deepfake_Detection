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
        sig = pad(sig)
        sr = 16000
        print("rate:", rate)
        S_full, phase = librosa.magphase(librosa.stft(sig))
        S_filter = librosa.decompose.nn_filter(S_full,
                                               aggregate=np.median,
                                               metric='cosine',
                                               width=int(librosa.time_to_frames(2, sr=sr)))
        S_filter = np.minimum(S_full, S_filter)
        margin_i, margin_v = 2, 10
        power = 2
        mask_i = librosa.util.softmask(S_filter,
                                       margin_i * (S_full - S_filter),
                                       power=power)

        mask_v = librosa.util.softmask(S_full - S_filter,
                                       margin_v * S_filter,
                                       power=power)
        S_foreground = mask_v * S_full
        S_background = mask_i * S_full
        print(S_background.shape)
        # rmse = np.reshape(rmse, (1, rmse.size))
        print("feat S_background:", S_background.shape)
        feats.append((S_background, label))
        with open(output_path, 'wb') as outfile:
            torch.save(feats, outfile)
