import os

import numpy as np
import scipy.stats as stats
from scipy.signal import welch

def feat_extract_dummy(eda_in, title_plot=''):
    feat = []

    # Existing features
    feat.append(np.mean(eda_in[:, 1]))  # Mean
    feat.append(np.std(eda_in[:, 1]))  # Standard Deviation
    feat.append(np.median(eda_in[:, 1]))  # Median

    # ... rest of the existing feature extraction code ...

    # Additional features
    skewness = stats.skew(eda_in[:, 1])
    kurtosis = stats.kurtosis(eda_in[:, 1])
    range_temp = np.max(eda_in[:, 1]) - np.min(eda_in[:, 1])
    q1 = np.percentile(eda_in[:, 1], 25)
    q3 = np.percentile(eda_in[:, 1], 75)

    # Add the additional features to the list
    feat += [skewness, kurtosis, range_temp, q1, q3]

    # Updated feature names list
    feature_names = ['mu', 'std', 'm', '#p', 'mu_amp', 'std_amp', 'mu_rise', 'std_rise', 'mu_decay',
                     'std_decay', 'mu_width', 'std_width', 'mu_half', 'std_half', 'mu_deriv', 'std_deriv',
                     'min', 'max', 'lin_coeff', 'perc_25', 'perc_75', 'perc_85', 'perc_95',
                     'skewness', 'kurtosis', 'range_temp', 'q1', 'q3']

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

