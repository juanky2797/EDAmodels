#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:34:55 2020

@author: apatane
"""

import os
import numpy as np
import scipy.signal as scisig
from scipy import interpolate
from scipy.io import wavfile
import pickle


def get_subjective_labels(userID, task, difficulty):
    # task order: stroop, subtraction, reading
    # difficulty order: easy, hard
    subjective_ranking = [
        [[1, 4], [2, 5], [2, 4]],
        [],
        [[1, 2], [1, 3], [2, 4]],
        [[1, 4], [1, 5], [1, 3]],
        [[1, 2], [3, 5], [3, 3]],
        [[1, 4], [1, 4], [2, 5]],
        [],
        [[2, 3], [1, 3], [1, 1]],
        [[1, 2], [1, 4], [3, 4]],
        [[2, 5], [1, 3], [2, 5]],
        [[1, 1], [1, 4], [2, 3]],
        [[4, 1], [1, 5], [2, 5]],
        [[1, 5], [1, 4], [3, 3]],
        [[2, 4], [1, 4], [2, 5]],
        [[2, 4], [1, 4], [2, 3]],
        [[1, 4], [1, 3], [2, 3]]
    ]
    if task == 'stroop':
        task_idx = 0
    elif task == 'subtraction':
        task_idx = 1
    elif task == 'reading':
        task_idx = 2

    if difficulty == 'easy':
        difficulty_idx = 0
    elif difficulty == 'hard':
        difficulty_idx = 1

    return subjective_ranking[int(userID) - 1][task_idx][difficulty_idx]


def load_and_splice_raw_signal(users, signals, tasks, dataDir='record/data/', labels='objective', win_size=300,
                               overlap=False):
    xs = []
    ys = []
    ts = []
    if overlap:
        step_size = int(0.5 * win_size)
    else:
        step_size = win_size

    for uID in users:
        # print(uID)
        for t in tasks:
            # print(t)
            for s in signals:
                hard_file = dataDir + uID + '/' + t + '_0/' + s + '.csv'
                easy_file = dataDir + uID + '/' + t + '_1/' + s + '.csv'

                if os.path.exists(hard_file):

                    hard_signal = np.loadtxt(hard_file, delimiter=',')
                    start_point = 0
                    end_point = start_point + win_size
                    while start_point < 300:  # it's 5 minutes of signal
                        # print([start_point,end_point])
                        hard_idxs = (hard_signal[:, 0] <= end_point) & (hard_signal[:, 0] >= start_point)
                        hard_signal_curr = hard_signal[hard_idxs, :]
                        if len(hard_signal_curr) > 0:
                            hard_signal_curr[:, 0] = hard_signal_curr[:, 0] - hard_signal_curr[0, 0]
                            xs.append(hard_signal_curr)
                            if labels == 'objective':
                                ys.append(1)
                            elif labels == 'subjective':
                                y_curr = get_subjective_labels(uID, t, 'hard')
                                ys.append(y_curr)
                            ts.append(t + '_hard')
                        start_point = start_point + step_size
                        end_point = end_point + step_size

                if os.path.exists(easy_file):

                    easy_signal = np.loadtxt(easy_file, delimiter=',')
                    start_point = 0
                    end_point = start_point + win_size
                    while start_point < 300:  # it's 5 minutes of signal
                        # print([start_point,end_point])
                        easy_idxs = (easy_signal[:, 0] <= end_point) & (easy_signal[:, 0] >= start_point)
                        easy_signal_curr = easy_signal[easy_idxs, :]
                        if len(easy_signal_curr) > 0:
                            easy_signal_curr[:, 0] = easy_signal_curr[:, 0] - easy_signal_curr[0, 0]
                            xs.append(easy_signal_curr)
                            if labels == 'objective':
                                ys.append(0)
                            elif labels == 'subjective':
                                y_curr = get_subjective_labels(uID, t, 'easy')
                                ys.append(y_curr)
                            ts.append(t + '_easy')
                        start_point = start_point + step_size
                        end_point = end_point + step_size
    return xs, ys, ts


def load_and_splice_raw_audio_signal_and_save(users, signals, tasks, dataDir='record/data/', win_size=5, overlap=0.5):
    if not (signals == ['mic_clean']) or (signals == ['mic']):
        print('this function is only for mic signal')
        return

    step_size = (1 - overlap) * win_size
    for uID in users:
        print(uID)
        for t in tasks:

            for s in signals:

                hardResultDir = dataDir + uID + '/' + t + '_0/' + 'windows/'
                if not os.path.exists(hardResultDir):
                    os.mkdir(hardResultDir)
                hardResultDir = hardResultDir + s + '/'
                if not os.path.exists(hardResultDir):
                    os.mkdir(hardResultDir)
                hardResultDir = hardResultDir + str(win_size) + '_' + str(overlap) + '/'
                if not os.path.exists(hardResultDir):
                    os.mkdir(hardResultDir)

                easyResultDir = dataDir + uID + '/' + t + '_1/' + 'windows/'
                if not os.path.exists(easyResultDir):
                    os.mkdir(easyResultDir)
                easyResultDir = easyResultDir + s + '/'
                if not os.path.exists(easyResultDir):
                    os.mkdir(easyResultDir)
                easyResultDir = easyResultDir + str(win_size) + '_' + str(overlap) + '/'
                if not os.path.exists(easyResultDir):
                    os.mkdir(easyResultDir)

                hard_file = dataDir + uID + '/' + t + '_0/' + s + '.wav'
                easy_file = dataDir + uID + '/' + t + '_1/' + s + '.wav'

                if os.path.exists(hard_file):

                    # hard_signal = np.loadtxt(hard_file,delimiter = ',')
                    w_hard = []
                    sr_hard, w_hard = wavfile.read(hard_file)
                    time_hard = np.asarray([i * 1 / sr_hard for i in range(len(w_hard))])

                    start_point = 0
                    end_point = start_point + win_size
                    count = 0
                    while start_point < 300:  # it's 5 minutes of signal

                        hard_idxs = (time_hard <= end_point) & (time_hard >= start_point)
                        w_hard_curr = w_hard[hard_idxs]
                        if len(w_hard_curr) > 0:
                            # print(hard_file)
                            # print(t)
                            # print(hardResultDir + str(count) + '.wav')
                            wavfile.write(hardResultDir + str(count) + '.wav', sr_hard, w_hard_curr)
                        start_point = start_point + step_size
                        end_point = end_point + step_size
                        count = count + 1

                if os.path.exists(easy_file):

                    # easy_signal = np.loadtxt(easy_file ,delimiter = ',')
                    w_easy = []
                    sr_easy, w_easy = wavfile.read(easy_file)
                    time_easy = np.asarray([i * 1 / sr_easy for i in range(len(w_easy))])
                    start_point = 0
                    end_point = start_point + win_size
                    count = 0
                    while start_point < 300:  # it's 5 minutes of signal
                        # print([start_point,end_point])
                        # print(easy_file)
                        easy_idxs = (time_easy <= end_point) & (time_easy >= start_point)
                        w_easy_curr = w_easy[easy_idxs]
                        if len(w_easy_curr) > 0:
                            # print(t)
                            # print(easyResultDir + str(count) + '.wav')
                            wavfile.write(easyResultDir + str(count) + '.wav', sr_easy, w_easy_curr)
                        start_point = start_point + step_size
                        end_point = end_point + step_size
                        count = count + 1
    return


def load_data_features(users, signals, tasks, dataDir='/Users/juanky/Documents/nokia_data/', labels='objective', subj_thre=3, win_size=300,
                       overlap=False):
    xs = []
    ys = []
    for uID in users:
        for t in tasks:
            for s in signals:
                hard_feat_template = dataDir + uID + '/' + t + '_0/features/'
                easy_feat_template = dataDir + uID + '/' + t + '_1/features/'
                if not np.isnan(overlap):
                    if overlap:
                        hard_feat_template = hard_feat_template + 'overlap_'
                        easy_feat_template = easy_feat_template + 'overlap_'
                if not np.isnan(win_size):
                    easy_feat_template = easy_feat_template + str(win_size) + '/' + s
                    hard_feat_template = hard_feat_template + str(win_size) + '/' + s
                else:
                    hard_feat_template = dataDir + uID + '/' + t + '_0/features/' + s
                    easy_feat_template = dataDir + uID + '/' + t + '_1/features/' + s

                if not np.isnan(win_size):
                    count = 0
                else:
                    count = ''

                while True:
                    hard_feat_file = hard_feat_template + str(count) + '.csv'
                    if os.path.exists(hard_feat_file):
                        xs.append(np.loadtxt(hard_feat_file, delimiter=','))
                        if labels == 'objective':
                            ys.append(1)
                        elif labels == 'subjective':
                            y_curr = get_subjective_labels(uID, t, 'hard')
                            ys.append(y_curr)
                    else:
                        break
                    if not np.isnan(win_size):
                        count = count + 1
                    else:
                        break

                if not np.isnan(win_size):
                    count = 0
                else:
                    count = ''

                while True:
                    easy_feat_file = easy_feat_template + str(count) + '.csv'
                    if os.path.exists(easy_feat_file):
                        xs.append(np.loadtxt(easy_feat_file, delimiter=','))
                        if labels == 'objective':
                            ys.append(0)
                        elif labels == 'subjective':
                            y_curr = get_subjective_labels(uID, t, 'easy')
                            ys.append(y_curr)
                    else:
                        break
                    if not np.isnan(win_size):
                        count = count + 1
                    else:
                        break
    ys = np.asarray(ys)
    if labels == 'subjective':
        pos_idxs = (ys > subj_thre)
        neg_idxs = (ys <= subj_thre)
        ys[pos_idxs] = 1
        ys[neg_idxs] = 0
    return np.asarray(xs), ys


def load_features_sequences(users, signals, tasks, expected_len, dataDir='record/data/',
                            labels='objective', subj_thre=3, win_size=300, overlap=0,
                            ignore_buzzers=False, timeSeries=True):
    xs = []
    ys = []
    ts = []
    for uID in users:
        for t in tasks:
            # for s in signals:
            s = signals[0]
            hard_feat_template = dataDir + uID + '/' + t + '_0/features/'
            easy_feat_template = dataDir + uID + '/' + t + '_1/features/'

            hard_feat_template = hard_feat_template + str(win_size) + '_' + str(overlap) + '/' + s + '/'
            easy_feat_template = easy_feat_template + str(win_size) + '_' + str(overlap) + '/' + s + '/'

            count = 0
            missing_count = 0
            x_curr_hard = []
            if ignore_buzzers:

                try:
                    easy_buzzer = pickle.load(open(dataDir + uID + '/' +
                                                   t + '_1/windows/mic_clean/' + str(win_size) + '_' +
                                                   str(overlap) + '/buzzer.p', 'rb'))
                except:
                    easy_buzzer = []
                try:
                    hard_buzzer = pickle.load(open(dataDir + uID + '/' + t +
                                                   '_0/windows/mic_clean/' + str(win_size) +
                                                   '_' + str(overlap) + '/buzzer.p', 'rb'))
                except:
                    hard_buzzer = []

            while count < expected_len:

                if ignore_buzzers:
                    buzzFlag = (count in hard_buzzer)
                else:
                    buzzFlag = False

                hard_feat_file = hard_feat_template + str(count) + '.csv'
                # print(hard_feat_file)
                # stop
                if os.path.exists(hard_feat_file):
                    if buzzFlag:
                        x_curr_hard.append(x_curr_hard[-1])
                    else:
                        x_curr_hard.append(np.loadtxt(hard_feat_file, delimiter=','))
                else:
                    if len(x_curr_hard) > 0:
                        x_curr_hard.append(x_curr_hard[-1])
                    missing_count = missing_count + 1
                    if missing_count > (float(count) / 10.0):
                        break
                        x_curr_hard = []
                count = count + 1

            if len(x_curr_hard) > 0:
                xs.append(np.asarray(x_curr_hard))
                if labels == 'objective':

                    if timeSeries:
                        y_curr = [1]
                        # ys.append(1)
                    else:
                        y_curr = len(x_curr_hard) * [1]
                        # ys = ys + len(x_curr_hard)*[1]
                elif labels == 'subjective':
                    if timeSeries:
                        y_curr = [get_subjective_labels(uID, t, 'hard')]
                    else:
                        #    ys.append(y_curr)
                        # else:
                        # ys = ys + len(x_curr_hard)*[y_curr]
                        y_curr = len(x_curr_hard) * [get_subjective_labels(uID, t, 'hard')]
                ys = ys + y_curr
                ts = ts + len(y_curr) * [t + '_hard']

            count = 0
            missing_count = 0
            x_curr_easy = []
            while count < expected_len:

                if ignore_buzzers:
                    buzzFlag = (count in easy_buzzer)
                else:
                    buzzFlag = False

                easy_feat_file = easy_feat_template + str(count) + '.csv'
                if os.path.exists(easy_feat_file):
                    if buzzFlag:
                        x_curr_easy.append(x_curr_easy[-1])
                    else:
                        x_curr_easy.append(np.loadtxt(easy_feat_file, delimiter=','))
                else:
                    # break
                    if len(x_curr_easy) > 0:
                        x_curr_easy.append(x_curr_easy[-1])
                    missing_count = missing_count + 1
                    if missing_count > (float(count) / 10.0):
                        break
                        x_curr_easy = []
                count = count + 1
            if len(x_curr_easy) > 0:
                xs.append(np.asarray(x_curr_easy))
                if labels == 'objective':
                    if timeSeries:
                        # ys.append(0)
                        y_curr = [0]
                    else:
                        # ys = ys + len(x_curr_easy)*[0]
                        y_curr = len(x_curr_easy) * [0]
                elif labels == 'subjective':
                    if timeSeries:
                        y_curr = [get_subjective_labels(uID, t, 'easy')]
                    # if timeSeries:
                    #    ys.append(y_curr)
                    else:
                        y_curr = len(x_curr_easy) * [get_subjective_labels(uID, t, 'easy')]
                        # ys = ys +  len(x_curr_easy)*[y_curr]
                ys = ys + y_curr
                ts = ts + len(y_curr) * [t + '_easy']
            # END FOR LOOP
    ys = np.asarray(ys)
    if labels == 'subjective':
        pos_idxs = (ys > subj_thre)
        neg_idxs = (ys <= subj_thre)
        ys[pos_idxs] = 1
        ys[neg_idxs] = 0
    xs = np.asarray(xs)
    if not timeSeries:
        xs = np.concatenate(xs)
    return xs, ys, ts
    # return xs, np.asarray(ys)


def normalisation(x, mu=[], std=[]):
    n_feats = x.shape[1]
    if mu == []:
        for i in range(n_feats):
            mu.append(np.mean(x[:, i]))
    if std == []:
        for i in range(n_feats):
            std.append(np.std(x[:, i]))

    for i in range(n_feats):
        x[:, i] = (x[:, i] - mu[i]) / std[i]

    return x, mu, std


def per_channel_signal_normalisation(x_set, mu=[], std=[]):
    n_ch = x_set.shape[2]

    if (len(mu) == 0) or (len(std) == 0):
        mu = [np.mean(x_set[:, :, i]) for i in range(n_ch)]
        std = [np.std(x_set[:, :, i]) for i in range(n_ch)]

    for i in range(n_ch):
        x_set[:, :, i] = (x_set[:, :, i] - mu[i]) / std[i]

    return x_set, mu, std


def signal_normalisation(x_set, mu=np.nan, std=np.nan):
    if np.isnan(mu):
        mu = np.mean(x_set)
        std = np.std(x_set)

    x_set = (x_set - mu) / std

    return x_set, mu, std


def re_sample_signal(t, x, fs, t_end):
    xx = [x[0]]
    tt = [t[0]]
    for i in range(1, len(t)):
        if t[i] == t[i - 1]:
            xx[-1] = 0.5 * (xx[-1] + x[i])
        else:
            tt.append(t[i])
            xx.append(x[i])
    tt = np.asarray(tt)
    xx = np.asarray(xx)

    fcubic = interpolate.interp1d(tt, xx, kind='linear', fill_value="extrapolate")
    dt = 1. / fs
    time_vec = []
    i = 0
    while i * dt < t_end:
        # print(i*dt)
        time_vec.append(i * dt)
        i = i + 1
    time_vec = np.asarray(time_vec)
    x_new = fcubic(time_vec)
    return x_new, time_vec


def re_sample_signal_matrix(signal, fs, t_end):
    t = signal[:, 0]
    n_ch = signal.shape[1] - 1
    signal_new = []
    for j in range(1, n_ch + 1):
        x = signal[:, j]
        x_new, t_new = re_sample_signal(t, x, fs, t_end)
        signal_new.append(x_new)

    return np.asmatrix(signal_new).T, t_new


def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y


def per_task_label_prediction_fusion(vec_of_tasks_idxs, y_hat, there=0.5):
    y_hat_overall = []
    y_hat_overall_median = []
    y_hat_overall_pred_majority = []
    y_hat_overall_majority = []
    y_hat_overall_pred_median = []
    y_hat_overall_pred = []
    for idxs in vec_of_tasks_idxs:
        if len(idxs) > 0:
            y_hat_curr_pred = np.mean(y_hat[idxs])
            y_hat_curr_pred_median = np.median(y_hat[idxs])
            y_hat_curr_majority_pred = np.mean(y_hat[idxs] >= there)
            y_hat_curr = float(y_hat_curr_pred >= there)
            y_hat_curr_median = float(y_hat_curr_pred_median >= there)
            y_hat_curr_majority = float(y_hat_curr_majority_pred >= there)
            y_hat_overall.append(y_hat_curr)
            y_hat_overall_median.append(y_hat_curr_median)
            y_hat_overall_majority.append(y_hat_curr_majority)
            y_hat_overall_pred.append(y_hat_curr_pred)
            y_hat_overall_pred_median.append(y_hat_curr_pred_median)
            y_hat_overall_pred_majority.append(y_hat_curr_majority_pred)

    return (y_hat_overall, y_hat_overall_median, y_hat_overall_pred_majority,
            y_hat_overall_majority, y_hat_overall_pred_median, y_hat_overall_pred)


def per_task_label(vec_of_tasks_idxs, y):
    y_overall = []
    for idxs in vec_of_tasks_idxs:
        if len(idxs) > 0:
            y_curr = np.asarray(y)[idxs][0]
            y_overall.append(y_curr)

    return y_overall


def data_augmentation(x_train, y_train):
    x_train_aug = []
    y_train_aug = []
    gamma = 0.1
    n_rep = 5
    n_interp = 1
    n_extr = 0

    for i in range(len(x_train)):
        curr_x_train = x_train[i]
        curr_y_train = y_train[i]
        x_train_aug.append(curr_x_train)
        y_train_aug.append(curr_y_train)
        for j in range(n_rep):
            # x_mod = curr_x_train + gamma*np.random.normal(scale = np.std(curr_x_train) ,size = curr_x_train.shape)
            x_mod = curr_x_train + gamma * np.random.normal(size=curr_x_train.shape)
            x_train_aug.append(x_mod)
            y_train_aug.append(curr_y_train)
        d = np.asarray([np.linalg.norm(curr_x_train - x_train[j]) for j in range(len(x_train))])
        idxes = np.argsort(d[d > 0])
        count = 0
        j = 0
        while (count < n_interp) and j < (len(idxes)):
            curr_idx = idxes[j]
            if curr_y_train == y_train[curr_idx]:
                x_2_interp = x_train[d > 0][curr_idx]
                x_mod = curr_x_train + 0.5 * (x_2_interp - curr_x_train)
                x_train_aug.append(x_mod)
                y_train_aug.append(curr_y_train)
                count = count + 1
            j = j + 1

        count = 0
        j = 0
        while (count < n_extr) and j < (len(idxes)):
            curr_idx = idxes[j]
            if curr_y_train == y_train[curr_idx]:
                x_2_interp = x_train[d > 0][curr_idx]
                x_mod = curr_x_train - 0.5 * (x_2_interp - curr_x_train)
                x_train_aug.append(x_mod)
                y_train_aug.append(curr_y_train)
                count = count + 1
            j = j + 1

    return np.asarray(x_train_aug), np.asarray(y_train_aug)
