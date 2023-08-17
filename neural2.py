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
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
import seaborn as sns

import tensorflow as tf
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.python.keras.utils.vis_utils import plot_model

from my_utils import load_data_features, normalisation1, data_augmentation, load_and_splice_raw_signal, \
    augment_eda_data, load_and_splice_raw_signal2
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


def perform_cnn_model(signals, tasks, userIDs, labels, results,  win_size, overlap=False, ):
    mean_train_acc = []
    mean_val_acc = []
    mean_train_precision = []
    mean_val_precision = []
    mean_train_recall = []
    mean_val_recall = []
    mean_train_auc = []
    mean_val_auc = []

    x, y, _ = load_and_splice_raw_signal(userIDs, signals, tasks, labels=labels, win_size=win_size, overlap=overlap)
    x = np.array(x)
    y = np.array(y)

    # Create a scaler instance
    scaler = MinMaxScaler()

    # Fit the scaler to the data and transform the data
    # Apply scaler on each feature separately
    x = np.array([scaler.fit_transform(feature) for feature in x])

    x, y = augment_eda_data(x, y)

    # Splitting the data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.36, shuffle=True)

    # Augmenting the training data

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
        ]

        # Add more architectures here
    ]

    # Iterate through architectures and train each one
    mean_train_acc = []
    mean_val_acc = []
   # plt.figure(figsize=(10, 10))
    val_acc_histories = []

    # Iterate through architectures and train each one
    for i, arch in enumerate(architectures):
        model = create_model_architecture(arch)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', Precision(), Recall()])
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=[es])

        # Calculate mean validation accuracy over all epochs and store
       # mean_val_acc.append(np.mean(history.history['val_accuracy']))

        results.setdefault(signals[0], {}).setdefault(tasks[0], {}).setdefault('architecture' + str(i + 1), {})
        val_acc_histories.append(history.history['val_loss'])

        # Then, assign the values
        results[signals[0]][tasks[0]]['architecture' + str(i + 1)] = {
            "loss": np.mean(history.history['loss']),
            "val_loss": np.mean(history.history['val_loss']),
        }


    '''
    # After training all models, plot the results
    models = [f'Model {i + 1}' for i in range(len(architectures))]
    data = {'Model Architecture': models, 'Mean Validation Accuracy': mean_val_acc}

    # Create a figure with specific size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set color palette
    palette = sns.color_palette("Blues", len(models))

    # Create a seaborn barplot
    plot = sns.barplot(x='Model Architecture', y='Mean Validation Accuracy', data=data, palette=palette, ax=ax)

    # Annotate the bars with the accuracy values
    for p in plot.patches:
        plot.annotate(format(p.get_height(), '.2f'),
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center',
                      xytext=(0, 9),
                      textcoords='offset points')

    ax.set_xlabel('CNN Model Architecture')
    ax.set_ylabel('Validation Accuracy Over All Epochs')
    ax.set_title('Comparison of CNN Architectures for reading task')
    ax.grid(False)  # No gridlines

    # Show the plot with a tight layout
    plt.tight_layout()

    # Adjust the layout of the figure
    fig.subplots_adjust(right=0.85)  # Leave space for the legend

    plt.show()
    '''

    return results, val_acc_histories


if __name__ == '__main__':
    # signals = ['eda']

    # win_sizes_vec = [5,15,30,60,120,300]
    # overlap_vec = [True,True,True,True,True,False]

    additional_res_id = ''
    signals = ['temp']
    labels = 'objective'
    userIDs = ['1', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    # methods = ['NB', 'SVM', 'RFC']
    # feat_selects = ['none', 'MI', 'PCA', 'ANOVA', 'SFS', 'BFS']
    # feat_selects = ['naming_imu']
    # feat_selects = ['none','MI','naming']
    # names_vec = [['mfcc'],['chroma'],['mel'],['mfcc','chroma'],['mfcc','mel'],['chroma','mel'] ]
    # names_vec = ['time','freq','acc','gyro','gyro_time','acc_time','gyro_freq','acc_freq','_m']
    # task_to_analyse = ['stroop','reading','subtraction','all_tasks']
    task_to_analyse = ['stroop', 'reading', 'subtraction']
    step_hop = 5
    results = {}
    val_loss_all_tasks = []
    results_final = {}
    # task_to_analyse = ['reading']
    for tt in task_to_analyse:
        if tt == 'all_tasks':
            tasks = ['stroop', 'subtraction', 'reading']
        else:
            tasks = [tt]

        results_final, val_loss_histories = perform_cnn_model(signals, tasks, userIDs, labels, results, win_size=1000, overlap=False)
        val_loss_all_tasks.append(val_loss_histories)

        results_file_name = ('/Users/juanky/Documents/nokia_data/results_cnn_bvp' +
                             '.p')
        # pickle.dump([mat_val_means, mat_val_stds], open(results_file_name, 'wb'))
        # mat_val_means, mat_val_stds = pickle.load(open(results_file_name, 'rb'))
        # print(np.mean(mat_val_means, axis=2))
    print(val_loss_all_tasks)
    print(results_final)

'''
    fig, axs = plt.subplots(3, figsize=(10, 15))
    for i, task_acc_histories in enumerate(val_acc_all_tasks):
        for j, acc_history in enumerate(task_acc_histories):
            axs[i].plot(acc_history, label=f'Architecture {j + 1}')
            axs[i].set_title(f'Task {task_to_analyse[i]}')
            axs[i].set_ylabel('Accuracy')
            axs[i].set_xlabel('Epoch')
            axs[i].legend()
    plt.tight_layout()
    plt.show()
'''