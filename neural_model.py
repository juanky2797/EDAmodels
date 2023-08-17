#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:09:11 2020

@author: apatane
"""
import os
import random

import numpy as np
from keras.src.layers import Conv1D, MaxPooling1D
from keras.src.regularizers import l2, l1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall


import tensorflow as tf
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.python.keras.utils.vis_utils import plot_model

from my_utils import load_data_features, normalisation1, data_augmentation, load_and_splice_raw_signal, augment_eda_data
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight

np.random.seed(25)
import keras
import pydot
import pydotplus
from pydotplus import graphviz

import pydot_ng as pydot




from keras.optimizers import RMSprop, legacy
from keras.layers import BatchNormalization

from tensorflow.keras.optimizers import schedules


def perform_kfold_validation(signals, tasks, userIDs, classifier, feature_selection, max_num_of_feats, labels,
                             names=[''],
                             win_size=299, overlap=False, num_folds=5):



    # Call this function at the beginning of your script
    train_acc = []
    val_acc = []

    x, y, _ = load_and_splice_raw_signal(userIDs, signals, tasks, labels=labels, win_size=win_size, overlap=overlap)
    x = np.array(x)
    y = np.array(y)

    # Create a scaler instance
    scaler = MinMaxScaler()

    # Fit the scaler to the data and transform the data
    # Apply scaler on each feature separately
    x = np.array([scaler.fit_transform(feature) for feature in x])

    # Splitting the data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.36, shuffle=True)

    # Augmenting the training data
    x_train, y_train = augment_eda_data(x_train, y_train)

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.002, decay=1e-6)
    # EarlyStopping callback
    es = EarlyStopping(monitor='val_loss', patience=20)

    # Training the model

    def create_model_architecture(layers):
        model = Sequential()

        for layer in layers:
            if layer['type'] == 'Conv1D':
                model.add(Conv1D(layer['filters'], layer['kernel_size'], activation=layer['activation'],
                                 kernel_regularizer=l2(0.05),
                input_shape=(x_train.shape[1], x_train.shape[2])))
            elif layer['type'] == 'BatchNormalization':
                model.add(BatchNormalization())
            elif layer['type'] == 'MaxPooling1D':
                model.add(MaxPooling1D(layer['pool_size']))
            elif layer['type'] == 'Flatten':
                model.add(Flatten())
            elif layer['type'] == 'Dense':
                model.add(Dense(layer['units'], activation=layer['activation'],
                                kernel_regularizer=l2(0.05)))  # And here
            elif layer['type'] == 'Dropout':
                model.add(Dropout(layer['rate']))

        return model

    # Define a list of architectures
    architectures = [

        [
            {'type': 'Conv1D', 'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
            {'type': 'BatchNormalization'},
            {'type': 'MaxPooling1D', 'pool_size': 2},
            {'type': 'Flatten'},
            {'type': 'Dense', 'units': 64, 'activation': 'relu'},
            {'type': 'Dropout', 'rate': 0.5},
            {'type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
        ],
        [
            {'type': 'Conv1D', 'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
            {'type': 'BatchNormalization'},
            {'type': 'MaxPooling1D', 'pool_size': 2},
            {'type': 'Conv1D', 'filters': 128, 'kernel_size': 3, 'activation': 'relu'},
            {'type': 'BatchNormalization'},
            {'type': 'MaxPooling1D', 'pool_size': 2},
            {'type': 'Conv1D', 'filters': 256, 'kernel_size': 3, 'activation': 'relu'},
            {'type': 'BatchNormalization'},
            {'type': 'MaxPooling1D', 'pool_size': 2},
            {'type': 'Flatten'},
            {'type': 'Dense', 'units': 128, 'activation': 'relu'},
            {'type': 'Dropout', 'rate': 0.5},
            {'type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
        ],
        [
            {'type': 'Conv1D', 'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
            {'type': 'BatchNormalization'},
            {'type': 'MaxPooling1D', 'pool_size': 2},
            {'type': 'Conv1D', 'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
            {'type': 'BatchNormalization'},
            {'type': 'MaxPooling1D', 'pool_size': 2},
            {'type': 'Conv1D', 'filters': 128, 'kernel_size': 3, 'activation': 'relu'},
            {'type': 'BatchNormalization'},
            {'type': 'MaxPooling1D', 'pool_size': 2},
            {'type': 'Conv1D', 'filters': 256, 'kernel_size': 3, 'activation': 'relu'},
            {'type': 'BatchNormalization'},
            {'type': 'MaxPooling1D', 'pool_size': 2},
            {'type': 'Flatten'},
            {'type': 'Dense', 'units': 128, 'activation': 'relu'},
            {'type': 'Dropout', 'rate': 0.5},
            {'type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
        ],

        # Add more architectures here
    ]

    # Iterate through architectures and train each one
    for i, arch in enumerate(architectures):
        model = create_model_architecture(arch)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', Precision(), Recall()])
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=[es])

        train_acc.append(history.history['accuracy'][-1])
        val_acc.append(history.history['val_accuracy'][-1])

        print(f"Model {i} Performance")
        print(f"Training Accuracy: {history.history['accuracy'][-1]}")
        print(f"Validation Accuracy: {history.history['val_accuracy'][-1]}")

        print("-----------------------------")

        dot_img_file = '/Users/juanky/Documents/model1.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)


        # If you want to calculate mean validation accuracy over all epochs, you can do this
        mean_val_accuracy = np.mean(history.history['val_accuracy'])
        print(f"Mean Validation Accuracy for architecture {i + 1}: {mean_val_accuracy}")
        print("-----------------------------")



    return train_acc, val_acc

def parametric_analysis_of_methods(signals, tasks, userIDs, classifiers, labels,
                                   feat_selects=['none', 'MI', 'SFS', 'BFS'], max_features=np.inf, step_hop=5,
                                   names_vec=[],
                                   win_size=300, overlap=False):
    mat_training_means = []
    mat_val_means = []
    for classifier in classifiers:
        print('Currently using ' + classifier)

        curr_training_means = []
        curr_val_means = []
        if 'none' in feat_selects:
            print('none')
            train_acc, val_acc = perform_kfold_validation(signals,
                                                                          tasks,
                                                                          userIDs,
                                                                          classifier,
                                                                          'none',
                                                                          np.nan,
                                                                          labels,
                                                                          win_size=win_size,
                                                                          overlap=overlap)
            curr_training_means.append(train_acc)
            curr_val_means.append(train_acc)



        mat_training_means.append(curr_training_means)
        mat_val_means.append(curr_val_means)

    return mat_training_means, mat_val_means



if __name__ == '__main__':
    # signals = ['eda']

    # win_sizes_vec = [5,15,30,60,120,300]
    # overlap_vec = [True,True,True,True,True,False]

    additional_res_id = ''
    signals = ['eda']
    labels = 'objective'
    userIDs = ['1', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16']
   # methods = ['NB', 'SVM', 'RFC']
    methods = ['RNN']
    #feat_selects = ['none', 'MI', 'PCA', 'ANOVA', 'SFS', 'BFS']
    #feat_selects = ['naming_imu']
    feat_selects = ['none']
    # feat_selects = ['none','MI','naming']
    # names_vec = [['mfcc'],['chroma'],['mel'],['mfcc','chroma'],['mfcc','mel'],['chroma','mel'] ]
    # names_vec = ['time','freq','acc','gyro','gyro_time','acc_time','gyro_freq','acc_freq','_m']
    # task_to_analyse = ['stroop','reading','subtraction','all_tasks']
    task_to_analyse = ['reading']
    step_hop = 5
    # task_to_analyse = ['reading']
    for m in methods:
        for tt in task_to_analyse:
            if tt == 'all_tasks':
                tasks = ['stroop', 'subtraction', 'reading']
            else:
                tasks = [tt]
            classifiers = [m]
            print(tasks[0])

            mat_val_means, mat_val_stds = parametric_analysis_of_methods(signals,
                                                                         tasks,
                                                                         userIDs,
                                                                         classifiers,
                                                                         labels,
                                                                         feat_selects=feat_selects,
                                                                         step_hop=step_hop,
                                                                         names_vec=np.nan,
                                                                         win_size=1000,
                                                                         overlap=False)

            results_file_name = ('/Users/juanky/Documents/nokia_data/' +
                                 labels + '_' +
                                 m + '_' +
                                 tt + '_' +
                                 signals[0] + '_' +
                                 additional_res_id +
                                 '.p')
           # pickle.dump([mat_val_means, mat_val_stds], open(results_file_name, 'wb'))
            #mat_val_means, mat_val_stds = pickle.load(open(results_file_name, 'rb'))
            #print(np.mean(mat_val_means, axis=2))
