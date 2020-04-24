#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:33:36 2019

@author: tross
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import cm
from pandas.plotting import scatter_matrix
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, scale
from sklearn.model_selection import train_test_split, StratifiedKFold

#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("./musicfeatures"))

# Any results you write to the current directory are saved as output.

music_df = pd.read_csv("./musicfeatures/data.csv")
print(music_df.head())

features = list(music_df.columns)
features.remove('filename')
features.remove('label')
print(features)

labeled_groups = music_df.groupby('label')
labels = list(music_df['label'].unique())

# Group class labels by median value of feature
for feat in features:
    feat_groups = labeled_groups[feat]
    feat_med_by_group = [(group[0], group[1].median()) for group in list(feat_groups)]
    feat_med_by_group = sorted(feat_med_by_group, key=lambda x: x[1])
    feat_labels_ordered_by_median, ordered_medians = zip(*feat_med_by_group)


music_features_df = music_df[features]
print(music_features_df.head(3))
music_features_norm_df = pd.DataFrame(scale(music_features_df))
print(music_features_norm_df.head(3))

le = LabelEncoder()
new_labels = pd.DataFrame(le.fit_transform(music_df['label']))
music_df['label'] = new_labels
print(music_df.head(3))

model_ready_df = music_features_norm_df.copy()
model_ready_df['label'] = music_df['label']



# Splits the data into 10 different folds, each containing the whole set
# The folds contain two parts:
# index:0 the larger (9/10's) piece
# index:1 the smaller (1/10's) piece
folds = 10
random_state = random_state = random.randint(1, 65536)
cv = StratifiedKFold(n_splits=folds,
                     shuffle=True,
                     random_state=random_state,
                     )

data = list(cv.split(music_features_df, music_df['label']))

model = Sequential()
model.add(Dense(28, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

# Let's start with the first fold of data just to see that everything works
first_fold = data[0]
train_indices, test_indices = first_fold[0], first_fold[1]
train_data = music_features_norm_df.iloc[train_indices]
train_labels = music_df['label'].iloc[train_indices]
test_data = music_features_norm_df.iloc[test_indices]
test_labels = music_df['label'].iloc[test_indices]

history = model.fit(train_data.values, train_labels.values, epochs=120, validation_data=(test_data.values, test_labels.values))

test_loss, test_acc = model.evaluate(test_data.values, test_labels.values)

print('\nTest accuracy:', test_acc)

ig, axs = plt.subplots(2,1, figsize=(12,9), constrained_layout=True)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
axs[0].plot(history.history['acc'], label='acc')
axs[0].plot(history.history['val_acc'], label='val_acc')
axs[0].set_title('model accuracy')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')
axs[0].grid(True, which='major')
# summarize history for loss
axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'], label='val_loss')
axs[1].set_title('model loss')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].grid(True, which='major')

# axs[0].legend(loc='upper left')



# axs[1].legend(loc='upper left')

plt.show()

'''
for i, fold_ind in enumerate(data[:]):
    print('Training on fold {} ...'.format(i))
    train_indices, test_indices = fold_ind[0], fold_ind[1]
    train_data = music_features_norm_df.iloc[train_indices]
    train_labels = music_df['label'].iloc[train_indices]
    test_data = music_features_norm_df.iloc[test_indices]
    test_labels = music_df['label'].iloc[test_indices]
    
    model = keras.Sequential([
        keras.layers.Dense(28, activation='relu'),
        keras.layers.Dense(19, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(train_data.values,
                        train_labels.values,
                        epochs=100,
                        batch_size=16,
                        validation_data=(test_data.values, test_labels.values),
                        verbose=0
                       )
    
    # summarize history for accuracy
    axs[0].plot(history.history['acc'], label='acc_'+str(i))
    axs[0].plot(history.history['val_acc'], label='val_acc_'+str(i))

    # summarize history for loss
    axs[1].plot(history.history['loss'], label='loss_'+str(i))
    axs[1].plot(history.history['val_loss'], label='val_loss_'+str(i))

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

from tensorflow.keras import layers

fig, axs = plt.subplots(2,1, figsize=(12,9), constrained_layout=True)

for i, fold_ind in enumerate(data[:]):
    print('Training on fold {} ...'.format(i))
    train_indices, test_indices = fold_ind[0], fold_ind[1]
    train_data = music_features_norm_df.iloc[train_indices]
    train_labels = music_df['label'].iloc[train_indices]
    test_data = music_features_norm_df.iloc[test_indices]
    test_labels = music_df['label'].iloc[test_indices]
    
    model = keras.Sequential([
        keras.layers.Dense(28, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(19, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(train_data.values,
                        train_labels.values,
                        epochs=100,
                        batch_size=16,
                        validation_data=(test_data.values, test_labels.values),
                        verbose=0
                       )
    
    # summarize history for accuracy
    axs[0].plot(history.history['acc'], label='acc_'+str(i))
    axs[0].plot(history.history['val_acc'], label='val_acc_'+str(i))

    # summarize history for loss
    axs[1].plot(history.history['loss'], label='loss_'+str(i))
    axs[1].plot(history.history['val_loss'], label='val_loss_'+str(i))

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
'''
