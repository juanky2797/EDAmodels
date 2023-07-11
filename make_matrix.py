#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:53:20 2020

@author: apatane
"""

import numpy as np
import os
import scipy.signal as scisig
from scipy import interpolate
from scipy.io import wavfile
from my_utils import butter_lowpass_filter


# from clean_mic import remove_timer

# def butter_lowpass(cutoff, fs, order=5):
#     # Filtering Helper functions
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
#     return b, a

# def butter_lowpass_filter(data, cutoff, fs, order=5):
#     # Filtering Helper functions
#     b, a = butter_lowpass(cutoff, fs, order=order)
#     y = scisig.lfilter(b, a, data)
#     return y


def keep_subtask(mat, initTime, finTime):
    idxs = mat[:, 0] <= finTime
    mat = mat[idxs, :]
    idxs = mat[:, 0] >= initTime
    mat = mat[idxs, :]
    mat[:, 0] = mat[:, 0] - mat[0, 0]

    return mat


def couple_data_with_label(userID, signals, data_dir='/Users/juanky/Documents/nokia_data/', save_dir='/Users/juanky/Documents/nokia_data/new_data/',
                           tasks=['stroop', 'subtraction', 'reading']):
    # userID directory
    user_dir = data_dir + userID + '/'
    save_dir = save_dir + userID + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # order: stroop (easy,hard); subtraction (easy,hard); reading (easy,hard)
    ordered_tasks = ['stroop_1', 'stroop_0', 'subtraction_1', 'subtraction_0', 'reading_1', 'reading_0']

    for task in ordered_tasks:
        print(task)
        try:
            if task[:-2] in tasks:
                task_data_dir = user_dir + task + '/'
                curr_save_dir = save_dir + task + '/'

                if not os.path.exists(curr_save_dir):
                    os.makedirs(curr_save_dir)

                for s in signals:
                    data = np.loadtxt(task_data_dir + s + '.csv', delimiter=',')
                    task_times = [data[0, 0],
                                  data[-1, 0]]  # Assuming first column of CSV is time, and tasks cover entire duration
                    curr_data_window = keep_subtask(data, task_times[0], task_times[1])

                    # Signal Cleaning implemented below in if-else statements depending on signal
                    if s == 'eda':
                        curr_data_window[:, 1] = butter_lowpass_filter(curr_data_window[:, 1], 0.25, 4, 6)

                    np.savetxt(curr_save_dir + s + '.csv', curr_data_window, delimiter=',')
        except Exception as e:
            print('Processing for task ' + task + ' failed with error: ' + str(e))

    return


if __name__ == '__main__':
    userIDs = ['1', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    signals = ['bvp', 'eda', 'temp']
    tasks = ['stroop', 'reading', 'subtraction']
    for u in userIDs:
        print(u)
        couple_data_with_label(u, signals, tasks=tasks)