#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:09:11 2020

@author: apatane
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
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


def perform_loo_validation(signals, tasks, userIDs, classifier, feature_selection, max_num_of_feats, labels, names=[''],
                           win_size=300, overlap=False):

    train_f1 = []
    train_auc = []
    train_acc = []

    val_f1 = []
    val_auc = []
    val_acc = []
    for i in range(len(userIDs)):
        train_users = userIDs[:i] + userIDs[(i + 1):]
        val_users = [userIDs[i]]

        x_train, y_train = load_data_features(train_users, signals, tasks, labels=labels, subj_thre=3,
                                              win_size=win_size, overlap=overlap)
        x_val, y_val = load_data_features(val_users, signals, tasks, labels=labels, subj_thre=3, win_size=win_size,
                                          overlap=overlap)

        if labels == 'subjective':
            _, task_val = load_data_features(val_users, signals, tasks, labels='objective', subj_thre=3,
                                             win_size=win_size, overlap=overlap)
            _, task_train = load_data_features(train_users, signals, tasks, labels='objective', subj_thre=3,
                                             win_size=win_size, overlap=overlap)
        else:
            task_val = y_val
            task_train = y_train
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
                    raise ValueError('Signals specific hyper-parameters not yet implemented')

                clf = svm.SVC(C=C)

            elif classifier == 'RFC':

                if signals == ['eda']:
                    MAX_DEPTH = 3
                    N_ESTIMATORS = 25
                elif signals == ['bvp']:
                    MAX_DEPTH = 3
                    N_ESTIMATORS = 25
                elif signals == ['bvp']:
                    MAX_DEPTH = 3
                    N_ESTIMATORS = 25
                else:
                    raise ValueError('Signals specific hyper-parameters not yet implemented')

                clf = RandomForestClassifier(max_depth=MAX_DEPTH,
                                             n_estimators=N_ESTIMATORS)
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
            else:
                raise ValueError('Feature selection method not supported')

            clf.fit(x_train, y_train)

            y_train_hat = clf.predict(x_train)
            train_acc.append(np.mean(y_train_hat == y_train))

            train_f1.append(f1_score(y_train, y_train_hat, average='macro'))

            if len(np.unique(y_train)) > 1:  # check if there are enough unique values to compute AUC
                train_auc.append(roc_auc_score(y_train, y_train_hat))

            y_val_hat = clf.predict(x_val)
            y_val_scores = clf.predict_proba(x_val)[:, 1]  # probabilities for the positive outcome

            task_one_idx = (task_val == 1)
            task_zero_idx = (task_val == 0)

            y_val_hat_task_one = y_val_hat[task_one_idx]
            y_val_hat_task_zero = y_val_hat[task_zero_idx]

            val_f1.append(f1_score(y_val, y_val_hat, average='macro'))

            if len(np.unique(y_val)) > 1:  # check if there are enough unique values to compute AUC
                val_auc.append(roc_auc_score(y_val, y_val_hat))

            fpr, tpr, thresholds = roc_curve(y_val, y_val_scores)

            y_val_hat_task_one_prediction = float(np.mean(y_val_hat_task_one) >= 0.5)
            y_val_hat_task_zero_prediction = float(np.mean(y_val_hat_task_zero) >= 0.5)

            one_idx = np.nonzero(y_val == 1)[0]
            zero_idx = np.nonzero(y_val == 0)[0]
            y_val_task = np.concatenate((y_val[one_idx], y_val[zero_idx]))
            y_val_hat_task = np.asarray([y_val_hat_task_one_prediction, y_val_hat_task_zero_prediction])
            val_acc.append(np.mean(y_val_task == y_val_hat_task))
        else:
            pass

    return train_acc, val_acc, train_f1, val_f1, train_auc, val_auc, n_features, fpr, tpr, thresholds


def parametric_analysis_of_methods(signals, tasks, userIDs, classifiers, labels, results,
                                   feat_selects=['none', 'MI', 'SFS', 'BFS'], max_features=np.inf, step_hop=5,
                                   names_vec=[],
                                   win_size=300, overlap=False):
    for classifier in classifiers:
        print('Currently using ' + classifier)

        if 'none' in feat_selects:
            print('none')
            train_acc, val_acc, train_f1, val_f1, train_auc, val_auc, max_features, fpr, tpr, thresholds = perform_loo_validation(signals,
                                                                                                          tasks,
                                                                                                          userIDs,
                                                                                                          classifier,
                                                                                                          'none',
                                                                                                          np.nan,
                                                                                                          labels,
                                                                                                          win_size=win_size,
                                                                                                          overlap=overlap)

            results[labels][classifier][tasks[0]]['none'] = {"val_acc": np.mean(val_acc),
                                                             "val_stds": np.std(val_acc),
                                                             "f1_val_means": np.mean(val_f1),
                                                             "f1_val_stds": np.std(val_f1),
                                                             "auc_val_means": np.mean(val_auc),
                                                             "auc_val_stds": np.std(val_auc),
                                                             "train_acc": np.mean(train_acc),
                                                             "train_stds": np.std(train_acc),
                                                             "auc_train_means": np.mean(train_auc),
                                                             "f1_train_means": np.mean(train_f1),
                                                             "fpr": fpr,
                                                             "tpr": tpr,
                                                             "thresholds": thresholds,
                                                             }

        if 'MI' in feat_selects:
            print('MI')
            train_acc, val_acc, train_f1, val_f1, train_auc, val_auc, _, fpr, tpr, thresholds = perform_loo_validation(signals,
                                                                                               tasks,
                                                                                               userIDs,
                                                                                               classifier,
                                                                                               'MI',
                                                                                               np.nan,
                                                                                               labels,
                                                                                               win_size=win_size,
                                                                                               overlap=overlap)

            results[labels][classifier][tasks[0]]['MI'] = { "val_acc": np.mean(val_acc),
                                                            "val_stds": np.std(val_acc),
                                                            "f1_val_means": np.mean(val_f1),
                                                            "f1_val_stds": np.std(val_f1),
                                                            "auc_val_means": np.mean(val_auc),
                                                            "auc_val_stds": np.std(val_auc),
                                                            "train_acc": np.mean(train_acc),
                                                            "train_stds": np.std(train_acc),
                                                            "auc_train_means": np.mean(train_auc),
                                                            "f1_train_means": np.mean(train_f1),
                                                            "fpr": fpr,
                                                            "tpr": tpr,
                                                            "thresholds": thresholds,
                                                             }

        if 'SFS' in feat_selects:
            print('Forward...')
            for i in range(step_hop + 1, max_features + 1, step_hop):
                # print(i)
                train_acc, val_acc, train_f1, val_f1, train_auc, val_auc, _, fpr, tpr, thresholds = perform_loo_validation(signals,
                                                                                                   tasks,
                                                                                                   userIDs,
                                                                                                   classifier,
                                                                                                   'SFS',
                                                                                                   i,
                                                                                                   labels,
                                                                                                   win_size=win_size,
                                                                                                   overlap=overlap)

                results[labels][classifier][tasks[0]]['SFS'] = { "val_acc": np.mean(val_acc),
                                                            "val_stds": np.std(val_acc),
                                                            "f1_val_means": np.mean(val_f1),
                                                            "f1_val_stds": np.std(val_f1),
                                                            "auc_val_means": np.mean(val_auc),
                                                            "auc_val_stds": np.std(val_auc),
                                                            "train_acc": np.mean(train_acc),
                                                            "train_stds": np.std(train_acc),
                                                            "auc_train_means": np.mean(train_auc),
                                                            "f1_train_means": np.mean(train_f1),
                                                                 "fpr": fpr,
                                                                 "tpr": tpr,
                                                                 "thresholds": thresholds,
                                                             }

        if 'BFS' in feat_selects:
            print('Backward...')
            for i in range(step_hop + 1, max_features + 1, step_hop):
                # print(i)
                train_acc, val_acc, train_f1, val_f1, train_auc, val_auc, _, fpr, tpr, thresholds = perform_loo_validation(signals,
                                                                                                   tasks,
                                                                                                   userIDs,
                                                                                                   classifier,
                                                                                                   'SBS',
                                                                                                   i,
                                                                                                   labels,
                                                                                                   win_size=win_size,
                                                                                                   overlap=overlap)

                results[labels][classifier][tasks[0]]['BFS'] = { "val_acc": np.mean(val_acc),
                                                            "val_stds": np.std(val_acc),
                                                            "f1_val_means": np.mean(val_f1),
                                                            "f1_val_stds": np.std(val_f1),
                                                            "auc_val_means": np.mean(val_auc),
                                                            "auc_val_stds": np.std(val_auc),
                                                            "train_acc": np.mean(train_acc),
                                                            "train_stds": np.std(train_acc),
                                                            "auc_train_means": np.mean(train_auc),
                                                            "f1_train_means": np.mean(train_f1),
                                                                 "fpr": fpr,
                                                                 "tpr": tpr,
                                                                 "thresholds": thresholds,
                                                             }

        if 'ANOVA' in feat_selects:
            print('ANOVA...')
            for i in range(step_hop + 1, max_features + 1, step_hop):
                print(i)
                train_acc, val_acc, train_f1, val_f1, train_auc, val_auc, _, fpr, tpr, thresholds = perform_loo_validation(signals,
                                                                                                   tasks,
                                                                                                   userIDs,
                                                                                                   classifier,
                                                                                                   'ANOVA',
                                                                                                   i,
                                                                                                   labels,
                                                                                                   win_size=win_size,
                                                                                                   overlap=overlap)

                results[labels][classifier][tasks[0]]['ANOVA'] = { "val_acc": np.mean(val_acc),
                                                            "val_stds": np.std(val_acc),
                                                            "f1_val_means": np.mean(val_f1),
                                                            "f1_val_stds": np.std(val_f1),
                                                            "auc_val_means": np.mean(val_auc),
                                                            "auc_val_stds": np.std(val_auc),
                                                            "train_acc": np.mean(train_acc),
                                                            "train_stds": np.std(train_acc),
                                                            "auc_train_means": np.mean(train_auc),
                                                            "f1_train_means": np.mean(train_f1),
                                                                   "fpr": fpr,
                                                                   "tpr": tpr,
                                                                   "thresholds": thresholds,
                                                             }

        if 'PCA' in feat_selects:
            print('PCA...')
            step_hop = 5
            for i in range(step_hop + 1, 2 * (len(userIDs) - 1), step_hop):
                print(i)
                train_acc, val_acc, train_f1, val_f1, train_auc, val_auc, _, fpr, tpr, thresholds = perform_loo_validation(signals,
                                                                                                   tasks,
                                                                                                   userIDs,
                                                                                                   classifier,
                                                                                                   'PCA',
                                                                                                   i,
                                                                                                   labels,
                                                                                                   win_size=win_size,
                                                                                                   overlap=overlap)

                results[labels][classifier][tasks[0]]['PCA'] = { "val_acc": np.mean(val_acc),
                                                            "val_stds": np.std(val_acc),
                                                            "f1_val_means": np.mean(val_f1),
                                                            "f1_val_stds": np.std(val_f1),
                                                            "auc_val_means": np.mean(val_auc),
                                                            "auc_val_stds": np.std(val_auc),
                                                            "train_acc": np.mean(train_acc),
                                                            "train_stds": np.std(train_acc),
                                                            "auc_train_means": np.mean(train_auc),
                                                            "f1_train_means": np.mean(train_f1),
                                                                 "fpr": fpr,
                                                                 "tpr": tpr,
                                                                 "thresholds": thresholds,
                                                             }

    return results



if __name__ == '__main__':
    # signals = ['eda']

    # win_sizes_vec = [5,15,30,60,120,300]
    # overlap_vec = [True,True,True,True,True,False]

    additional_res_id = ''
    signals = ['eda']
    labels = 'objective'
    userIDs = ['1', '3', '4', '5', '6', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    #methods = ['NB', 'SVM', 'RFC']
    methods = ['NB']
    #methods = ['SVM']
    #methods = ['RFC']
    #feat_selects = ['none', 'MI', 'PCA', 'ANOVA', 'SFS', 'BFS']
    # feat_selects = ['naming_imu']
    feat_selects = ['none']
    # feat_selects = ['none','MI','naming']
    # names_vec = [['mfcc'],['chroma'],['mel'],['mfcc','chroma'],['mfcc','mel'],['chroma','mel'] ]
    # names_vec = ['time','freq','acc','gyro','gyro_time','acc_time','gyro_freq','acc_freq','_m']
    # task_to_analyse = ['stroop','reading','subtraction','all_tasks']
    task_to_analyse = ['reading']
    step_hop = 5
    # task_to_analyse = ['reading']
    results = {}
    results_final = {}

    for m in methods:
        for tt in task_to_analyse:
            if tt == 'all_tasks':
                tasks = ['stroop', 'subtraction', 'reading']
            else:
                tasks = [tt]
            classifiers = [m]
            print(tasks[0])
            results[labels] = {}
            results[labels][classifiers[0]] = {}
            results[labels][classifiers[0]][tasks[0]] = {}

            results_final = parametric_analysis_of_methods(signals, tasks, userIDs, classifiers, labels, results,
                                                           feat_selects=feat_selects, step_hop=step_hop,
                                                           names_vec=np.nan, win_size=np.nan, overlap=np.nan)


            results_file_name = ('/Users/juanky/Documents/nokia_data/new_results/' +
                     labels + '_' +
                     m + '_' +
                     tt + '_' +
                     signals[0] + '_' +
                     'auc' +
                     '.txt')
            print(results_final)

            pickle.dump([results_final], open(results_file_name, 'wb'))