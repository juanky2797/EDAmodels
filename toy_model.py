#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:09:11 2020

@author: apatane
"""
import numpy as np
# import os
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFECV, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from my_utils import load_data_features, normalisation, data_augmentation
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(25)


def jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + myNoise


def time_warp(X, sigma=0.2):
    tt = np.arange(X.shape[0])
    tt_new = tt + np.random.normal(loc=0, scale=sigma, size=tt.shape)
    tt_new = np.clip(tt_new, 0, X.shape[0] - 1)
    X_new = np.zeros_like(X)
    for i in range(X.shape[1]):
        X_new[:, i] = np.interp(tt, tt_new, X[:, i])
    return X_new


def perform_loo_validation(signals, tasks, userIDs, classifier, feature_selection, max_num_of_feats, labels, names=[''],
                           win_size=300, overlap=False):
    train_acc = []
    val_acc = []
    train_precision = []
    val_precision = []
    train_recall = []
    val_recall = []
    train_f1 = []
    val_f1 = []
    train_auc = []
    val_auc = []

    for i in range(len(userIDs)):
        train_users = userIDs[:i] + userIDs[(i + 1):]
        val_users = [userIDs[i]]

        x_train, y_train = load_data_features(train_users, signals, tasks, labels=labels, subj_thre=3,
                                              win_size=win_size, overlap=overlap)
        x_val, y_val = load_data_features(val_users, signals, tasks, labels=labels, subj_thre=3, win_size=win_size,
                                          overlap=overlap)

        if classifier == 'MLP':
            x_train, y_train = data_augmentation(x_train, y_train)

        if labels == 'subjective':
            _, task_val = load_data_features(val_users, signals, tasks, labels='objective', subj_thre=3,
                                             win_size=win_size, overlap=overlap)
        else:
            task_val = y_val
        n_features = x_train.shape[1]
        if len(x_val) > 0:
            # print('Currently analysing user ' + val_users)
            x_train, mu, std = normalisation(x_train)
            x_val, _, _ = normalisation(x_val, mu=mu, std=std)

            if classifier == 'NB':

                clf = GaussianNB()

            elif classifier == 'SVM':

                if signals == ['eda']:
                    C = 2.0
                elif signals == ['bvp']:
                    C = 1.0
                elif signals == ['temp']:
                    C = 1.0
                else:
                    C = 1.0  # Default C for other signal types

                clf = svm.SVC(C=C)

            elif classifier == 'RFC':

                if signals == ['eda']:
                    MAX_DEPTH = 3
                    N_ESTIMATORS = 25
                elif signals == ['bvp']:
                    MAX_DEPTH = 3
                    N_ESTIMATORS = 100
                elif signals == ['temp']:
                    MAX_DEPTH = 1
                    N_ESTIMATORS = 200
                else:
                    MAX_DEPTH = 1  # Default MAX_DEPTH for other signal types
                    N_ESTIMATORS = 100  # Default N_ESTIMATORS for other signal types

                clf = RandomForestClassifier(max_depth=MAX_DEPTH,
                                             n_estimators=N_ESTIMATORS)
            elif classifier == 'MLP':

                clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
            else:
                raise ValueError('Classifier not supported')

            if feature_selection == 'none':
                pass
            elif feature_selection == 'MI':
                mi_scores = mutual_info_classif(x_train, y_train, n_neighbors=3)
                featIdxs = mi_scores > 0
                x_train = x_train[:, featIdxs]
                x_val = x_val[:, featIdxs]
            elif feature_selection == 'RFECV':
                selector = RFECV(clf)
                selector = selector.fit(x_train, y_train)
                featIdxs = selector.support_
                x_train = x_train[:, featIdxs]
                x_val = x_val[:, featIdxs]
            elif feature_selection == 'SFS':
                max_num_of_feats = min(max_num_of_feats, n_features)
                # print(max_num_of_feats)
                sfs1 = SFS(clf, k_features=max_num_of_feats, forward=True, verbose=0)
                sfs1 = sfs1.fit(x_train, y_train)
                x_train = x_train[:, sfs1.k_feature_idx_]
                x_val = x_val[:, sfs1.k_feature_idx_]
            elif feature_selection == 'SBS':
                max_num_of_feats = min(max_num_of_feats, n_features)
                sfs1 = SFS(clf, k_features=max_num_of_feats, forward=False)
                sfs1 = sfs1.fit(x_train, y_train)
                x_train = x_train[:, sfs1.k_feature_idx_]
                x_val = x_val[:, sfs1.k_feature_idx_]
            elif feature_selection == 'ANOVA':
                max_num_of_feats = min(max_num_of_feats, n_features)
                selector = SelectKBest(score_func=f_classif, k=max_num_of_feats)
                selector.fit(x_train, y_train)
                feat_idxs = np.nonzero(selector.get_support())
                x_train = x_train[:, feat_idxs[0]]
                x_val = x_val[:, feat_idxs[0]]
            elif feature_selection == 'PCA':
                max_num_of_feats = min(max_num_of_feats, len(x_train))
                pca = PCA(n_components=max_num_of_feats)
                pca.fit(x_train)
                x_train = pca.transform(x_train)
                x_val = pca.transform(x_val)
                # x_train = x_train[:,feat_idxs[0]]
                # x_val = x_val[:,feat_idxs[0]]

            elif feature_selection == 'naming_mic':
                idxs = []
                # mfcs: 0:39   #chroma: 0:11   #mel: 0:127
                if 'mfcc' in names:
                    idxs = idxs + np.arange(0, 40).tolist()
                if 'chroma' in names:
                    idxs = idxs + np.arange(40, 40 + 12).tolist()
                if 'mel' in names:
                    idxs = idxs + np.arange(40 + 12, 40 + 12 + 128).tolist()
                x_train = x_train[:, idxs]
                x_val = x_val[:, idxs]
            elif feature_selection == 'naming_imu':
                idxs = []
                feature_names = pickle.load(open('feat_names_imu.p', 'rb'))
                # currBoolVals = np.asarray([True for _ in range(len(feature_names))] )
                # for curr_name in names:
                currBoolVals = np.asarray([names in el for el in feature_names])
                idxs = np.nonzero(currBoolVals)[0].tolist()

                x_train = x_train[:, idxs]
                x_val = x_val[:, idxs]
            else:
                raise ValueError('Feature selection method not supported')

            clf.fit(x_train, y_train)

            y_train_hat = clf.predict(x_train)
            train_acc.append(np.mean(y_train_hat == y_train))

            y_val_hat = clf.predict(x_val)

            task_one_idx = (task_val == 1)
            task_zero_idx = (task_val == 0)

            y_val_hat_task_one = y_val_hat[task_one_idx]
            y_val_hat_task_zero = y_val_hat[task_zero_idx]

            y_val_hat_task_one_prediction = float(np.mean(y_val_hat_task_one) >= 0.5)
            y_val_hat_task_zero_prediction = float(np.mean(y_val_hat_task_zero) >= 0.5)

            one_idx = np.nonzero(y_val == 1)[0]
            zero_idx = np.nonzero(y_val == 0)[0]
            y_val_task = np.concatenate((y_val[one_idx], y_val[zero_idx]))
            y_val_hat_task = np.asarray([y_val_hat_task_one_prediction, y_val_hat_task_zero_prediction])
            val_acc.append(np.mean(y_val_task == y_val_hat_task))

        else:
            pass
            # print('Failed for user ' + val_users + ' as there are not enough samples ')
            # print(val_users)

    # print('Training Accuracy: ' + str(np.mean(train_acc)))
    # print('Validation Accuracy: ' + str(np.mean(val_acc)))
    # return np.mean(train_acc), np.mean(val_acc), np.std(train_acc), np.std(val_acc)
    return train_acc, val_acc, train_acc, val_acc, n_features


def parametric_analysis_of_methods(signals, tasks, userIDs, classifiers, labels,
                                   feat_selects=['none', 'MI', 'SFS', 'BFS'], max_features=np.inf, step_hop=5,
                                   names_vec=[],
                                   win_size=300, overlap=False):
    mat_val_means = []
    mat_val_stds = []
    for classifier in classifiers:
        print('Currently using ' + classifier)

        curr_val_means = []
        curr_val_stds = []
        if 'none' in feat_selects:
            print('none')
            _, val_acc, _, val_std, max_features = perform_loo_validation(signals,
                                                                          tasks,
                                                                          userIDs,
                                                                          classifier,
                                                                          'none',
                                                                          np.nan,
                                                                          labels,
                                                                          win_size=win_size,
                                                                          overlap=overlap)
            curr_val_means.append(val_acc)
            curr_val_stds.append(val_std)

        if 'MI' in feat_selects:
            print('MI')
            _, val_acc, _, val_std, _ = perform_loo_validation(signals,
                                                               tasks,
                                                               userIDs,
                                                               classifier,
                                                               'MI',
                                                               np.nan,
                                                               labels,
                                                               win_size=win_size,
                                                               overlap=overlap)
            curr_val_means.append(val_acc)
            curr_val_stds.append(val_std)

        # if 'naming' in feat_selects:
        #    for names in names_vec:
        #        _, val_acc, _, val_std, _ = perform_loo_validation(signals,
        #                                                           tasks,
        #                                                           userIDs,
        #                                                           classifier,
        #                                                           'naming',
        #                                                           np.nan,
        #                                                           labels,
        #                                                           names = names,
        #                                                           win_size = win_size,
        #                                                           overlap = overlap)
        #        curr_val_means.append(val_acc)
        #        curr_val_stds.append(val_std)

        # if 'naming_imu' in feat_selects:
        #    #names_vec = []
        #    for names in names_vec:
        #        print(names)
        #        _, val_acc, _, val_std, _ = perform_loo_validation(signals,
        #                                                           tasks,
        #                                                           userIDs,
        #                                                           classifier,
        #                                                           'naming_imu',
        #                                                           np.nan,
        #                                                           labels,
        #                                                           names = names,
        #                                                           win_size = win_size,
        #                                                           overlap = overlap)
        #        curr_val_means.append(val_acc)
        #        curr_val_stds.append(val_std)

        if 'SFS' in feat_selects:
            print('Forward...')
            for i in range(step_hop + 1, max_features + 1, step_hop):
                # print(i)
                _, val_acc, _, val_std, _ = perform_loo_validation(signals,
                                                                   tasks,
                                                                   userIDs,
                                                                   classifier,
                                                                   'SFS',
                                                                   i,
                                                                   labels,
                                                                   win_size=win_size,
                                                                   overlap=overlap)
                curr_val_means.append(val_acc)
                curr_val_stds.append(val_std)

        if 'BFS' in feat_selects:
            print('Backward...')
            for i in range(step_hop + 1, max_features + 1, step_hop):
                # print(i)
                _, val_acc, _, val_std, _ = perform_loo_validation(signals,
                                                                   tasks,
                                                                   userIDs,
                                                                   classifier,
                                                                   'SBS',
                                                                   i,
                                                                   labels,
                                                                   win_size=win_size,
                                                                   overlap=overlap)
                curr_val_means.append(val_acc)
                curr_val_stds.append(val_std)

        if 'ANOVA' in feat_selects:
            print('ANOVA...')
            for i in range(step_hop + 1, max_features + 1, step_hop):
                print(i)
                _, val_acc, _, val_std, _ = perform_loo_validation(signals,
                                                                   tasks,
                                                                   userIDs,
                                                                   classifier,
                                                                   'ANOVA',
                                                                   i,
                                                                   labels,
                                                                   win_size=win_size,
                                                                   overlap=overlap)
                curr_val_means.append(val_acc)
                curr_val_stds.append(val_std)

        if 'PCA' in feat_selects:
            print('PCA...')
            step_hop = 5
            for i in range(step_hop + 1, 2 * (len(userIDs) - 1), step_hop):
                print(i)
                _, val_acc, _, val_std, _ = perform_loo_validation(signals,
                                                                   tasks,
                                                                   userIDs,
                                                                   classifier,
                                                                   'PCA',
                                                                   i,
                                                                   labels,
                                                                   win_size=win_size,
                                                                   overlap=overlap)
                curr_val_means.append(val_acc)
                curr_val_stds.append(val_std)

        mat_val_means.append(curr_val_means)
        mat_val_stds.append(curr_val_stds)

    return mat_val_means, mat_val_stds




if __name__ == '__main__':
    # signals = ['eda']

    # win_sizes_vec = [5,15,30,60,120,300]
    # overlap_vec = [True,True,True,True,True,False]

    additional_res_id = ''
    signals = ['bvp']
    labels = 'subjective'
    userIDs = ['1', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    methods = ['NB', 'SVM', 'RFC']
    #methods = ['MLP']
    feat_selects = ['none', 'MI', 'PCA', 'ANOVA', 'SFS', 'BFS']
    #feat_selects = ['naming_imu']
    #feat_selects = ['none']
    # feat_selects = ['none','MI','naming']
    # names_vec = [['mfcc'],['chroma'],['mel'],['mfcc','chroma'],['mfcc','mel'],['chroma','mel'] ]
    # names_vec = ['time','freq','acc','gyro','gyro_time','acc_time','gyro_freq','acc_freq','_m']
    # task_to_analyse = ['stroop','reading','subtraction','all_tasks']
    task_to_analyse = ['stroop']
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
                                                                         win_size=np.nan,
                                                                         overlap=np.nan)

            results_file_name = ('/Users/juanky/Documents/nokia_data/' +
                                 labels + '_' +
                                 m + '_' +
                                 tt + '_' +
                                 signals[0] + '_' +
                                 additional_res_id +
                                 '.p')
            pickle.dump([mat_val_means, mat_val_stds], open(results_file_name, 'wb'))
            mat_val_means, mat_val_stds = pickle.load(open(results_file_name, 'rb'))
            print(np.mean(mat_val_means, axis=2))
