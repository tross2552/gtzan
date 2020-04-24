#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 19:42:28 2019

@author: tross
"""

import librosa
import pandas as pd
import numpy as np
import pickle as pkl

# Load the example clip

genre_names = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

input_data = []
output_data = []

for i in range(len(genre_names)):
    
    for j in range(0,100):
        
        path = "./genres/" + genre_names[i] + "/" + genre_names[i] + ".000"
        if j < 10: path += "0"
        path = path + str(j) + ".wav"
        print(path)


        y, sr = librosa.load(path)
        
        # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
        hop_length = 2048

        
        # Compute MFCC features from the raw signal
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=12)
        mfcc = mfcc[:,:300]

        input_data.append(mfcc)
        output_data.append(i)


input_data = np.stack(input_data)



print(input_data)
print(input_data.shape)


from keras.utils import to_categorical
output_data = to_categorical(output_data)

print(output_data)


label_file = open("label.pkl","wb")
pkl.dump(output_data, label_file)

mfcc_file = open("mfcc.pkl","wb")
pkl.dump(input_data, mfcc_file)
