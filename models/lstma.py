#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 18:11:01 2019

@author: tross
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:33:36 2019

@author: tross
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pandas.plotting import scatter_matrix
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, scale
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Bidirectional, LSTM, Input, Multiply
from keras.optimizers import Adam


from keras.layers.core import Permute, Reshape


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("./musicfeatures"))



import pickle as pkl
from keras import regularizers

mfcc_file = open("mfcc.pkl","rb")
mfcc_arr = pkl.load(mfcc_file)

label_file = open("label.pkl","rb")
label_arr = pkl.load(label_file)

print(mfcc_arr.shape)

x_train, x_test, y_train, y_test = train_test_split(
    mfcc_arr, label_arr, test_size=0.2,stratify=label_arr)

x_train = np.transpose(x_train, (0, 2, 1))
x_test = np.transpose(x_test, (0, 2, 1))

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(300, kernel_regularizer = regularizers.l2(l = 0.01), activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


inputs = Input(shape=(300, 12,))
attention_mul = attention_3d_block(inputs)
attention_mul = LSTM(128, dropout=0.1, recurrent_dropout=0.35, return_sequences=False)(attention_mul)
output = Dense(10, activation='softmax')(attention_mul)
model = Model(input=[inputs], output=output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy','accuracy'])

history = model.fit(x_train, y_train,
          batch_size=40,
          epochs=60,
          validation_data=[x_test, y_test])

fig, axs = plt.subplots(2,1, figsize=(12,9), constrained_layout=True)

# summarize history for accuracy
axs[0].plot(history.history['acc'], label='acc')
axs[0].plot(history.history['val_acc'], label='val_acc')

# summarize history for loss
axs[1].plot(history.history['loss'], label='loss')
axs[1].plot(history.history['val_loss'], label='val_loss')

axs[0].set_title('model accuracy')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')
axs[0].grid(True, which='major')
# axs[0].legend(loc='upper left')

axs[1].set_title('model loss')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].grid(True, which='major')
# axs[1].legend(loc='upper left')

plt.show()

'''
# summarize history for accuracy
plt.plot(history.history['acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
'''
