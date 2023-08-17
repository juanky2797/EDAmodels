#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:57:36 2020

@author: apatane
"""

import numpy as np
import os
import sys

sys.path.append('eda-explorer-master/')
# from cvxEDA import cvxEDA

from EDA_Peak_Detection_Script import findPeaks
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

plotFlag = False


def pre_process_signal(eda_in):
    eda_out = eda_in
    return eda_out


def feat_extract_dummy(eda_in, title_plot=''):
    feat = []

    feat.append(np.mean(eda_in[:, 1]))
    feat.append(np.std(eda_in[:, 1]))
    feat.append(np.median(eda_in[:, 1]))
    # feat.append( np.mean(np.diff(eda_in[:,1])) )
    # feat.append( np.std(np.diff(eda_in[:,1])) )

    # params = [2,8,8,0.004]
    params = [2, 8, 8, 0.008]
    (peaks, peak_start, _, peak_end, _,
     amplitude, max_deriv, rise_time, decay_time, SCR_width, half_rise) = findPeaks(
        eda_in, params[0], params[1], params[2], thres=params[3], sampleRate=4)

    if plotFlag:
        plt.figure()
        plt.plot(eda_in[:, 1])
        peak_indexes = [i + 1 for i in range(len(peaks)) if peaks[i]]
        plt.plot(peak_indexes, eda_in[peak_indexes, 1], 'ro')

    peak_start = [i + 1 for i in range(len(peak_start)) if peak_start[i]]
    peak_end = [i + 1 for i in range(len(peak_end)) if peak_end[i]]
    if len(peak_start) > len(peak_end):
        peak_end.append(len(eda_in[:, 1]) - 1)
    amplitude = [a for a in amplitude if a]
    rise_time = [a for a in rise_time if a]
    decay_time = [a for a in decay_time if a]
    SCR_width = [a for a in SCR_width if a]
    half_rise = [a for a in half_rise if a]
    max_deriv = [a for a in max_deriv if a]

    feat.append(len(peak_start))

    # if len(peak_start) > 0:
    if len(amplitude) > 0:
        feat.append(np.nanmean(amplitude))
        feat.append(np.nanstd(amplitude))
    else:
        feat.append(0)
        feat.append(0)

    if len(rise_time) > 0:
        feat.append(np.nanmean(rise_time))
        feat.append(np.nanstd(rise_time))
    else:
        feat.append(0)
        feat.append(0)

    if len(decay_time) > 0:
        feat.append(np.nanmean(decay_time))
        feat.append(np.nanstd(decay_time))
    else:
        feat.append(0)
        feat.append(0)

    if len(SCR_width) > 0:
        feat.append(np.nanmean(SCR_width))
        feat.append(np.nanstd(SCR_width))
    else:
        feat.append(0)
        feat.append(0)

    if len(half_rise) > 0:
        feat.append(np.nanmean(half_rise))
        feat.append(np.nanstd(half_rise))
    else:
        feat.append(0)
        feat.append(0)

    if len(max_deriv) > 0:
        feat.append(np.nanmean(max_deriv))
        feat.append(np.nanstd(max_deriv))
    else:
        feat.append(0)
        feat.append(0)

    # additional features
   # print(eda_in)
    if eda_in.size > 0:
        feat.append(np.min(eda_in[:, 1]))
    else:
        print("Warning: Empty array encountered")
    if eda_in.size > 0:
        feat.append(np.max(eda_in[:, 1]))
    else:
        print("Warning: Empty array encountered")

    regressor = LinearRegression()
    regressor.fit(eda_in[:, 0].reshape(-1, 1), eda_in[:, 1].reshape(-1, 1))
    if plotFlag:
        plt.plot(eda_in[:, 0], eda_in[:, 0] * regressor.coef_[0][0] + regressor.intercept_[0])
    feat.append(regressor.coef_[0][0])

    percentiles2comp = [25, 75, 85, 95]
    for perc in percentiles2comp:
        try:
            feat.append(np.percentile(amplitude, perc))
        except:
            feat.append(np.nan)

    # need to compute instantaneous peak rate

    for i in range(len(feat)):
        if np.isnan(feat[i]):
            feat[i] = .0
    # else:
    #    feat.append(0)
    #    feat.append(0)

    #    feat.append(0)
    #    feat.append(0)

    #    feat.append(0)
    #    feat.append(0)

    #    feat.append(0)
    #    feat.append(0)

    #    feat.append(0)
    #    feat.append(0)

    #    feat.append(0)
    #    feat.append(0)
    if plotFlag:
        plt.title(title_plot)
        for i in range(len(peak_start)):
            plt.plot([peak_start[i], peak_end[i]], [eda_in[peak_start[i], 1], eda_in[peak_end[i], 1]], 'r')

    feature_names = ['mu', 'std', 'm', '#p', 'mu_amp', 'std_amp', 'mu_rise', 'std_rise', 'mu_decay',
                     'std_decay', 'mu_width', 'std_width', 'mu_half', 'std_half', 'mu_deriv', 'std_deriv',
                     'min', 'max', 'lin_coeff', 'perc_25', 'perc_75', 'perc_85', 'perc_95']

    return np.asarray(feat), feature_names


def save_EDA_features2file(userID, task, data_dir='/Users/juanky/Documents/nokia_data/new_data/', offset=0):
    signal = 'eda'
    data_dir = data_dir + userID + '/' + task
    easy_dir = data_dir + '_1/'
    hard_dir = data_dir + '_0/'

    easy_file = easy_dir + signal + '.csv'
    hard_file = hard_dir + signal + '.csv'

    if os.path.exists(easy_file) and os.path.exists(hard_file):
        easy_eda = np.loadtxt(easy_file, delimiter=',')
        hard_eda = np.loadtxt(hard_file, delimiter=',')

        easy_eda = pre_process_signal(easy_eda)
        hard_eda = pre_process_signal(hard_eda)

        easy_eda = easy_eda[offset:-(offset + 1), :]
        hard_eda = hard_eda[offset:-(offset + 1), :]

        print(easy_eda)
        print(hard_eda)


        easy_feats, feature_names = feat_extract_dummy(easy_eda, title_plot='Easy: User: ' + userID + ' Task: ' + task)
        hard_feats, feature_names = feat_extract_dummy(hard_eda, title_plot='Hard: User: ' + userID + ' Task: ' + task)
        # (r, p, t, l, d, e, obj) = cvxEDA(hard_eda[:,1],0.25)

        hard_output = hard_dir + 'features/' + signal + '.csv'
        easy_output = easy_dir + 'features/' + signal + '.csv'

        if not os.path.exists(hard_dir + 'features/'):
            os.makedirs(hard_dir + 'features/')
        if not os.path.exists(easy_dir + 'features/'):
            os.makedirs(easy_dir + 'features/')

        np.savetxt(easy_output, easy_feats, delimiter='')
        np.savetxt(hard_output, hard_feats, delimiter='')
    else:
        feature_names = ''
        print('EDA file missing for user ' + userID + ' task ' + task)

    return feature_names


if __name__ == "__main__":
    userIDs = ['1', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    task = 'reading'
    for userID in userIDs:
        save_EDA_features2file(userID, task, data_dir='/Users/juanky/Documents/nokia_data/new_data/', offset=200)
