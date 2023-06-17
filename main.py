
import matplotlib as matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

classifiers = [
    ("Logistic Regression", LogisticRegression(penalty='l2', C=1.0)),  # C=1.0 is default value
    ("Random Forest", RandomForestClassifier()),
    ("SVM", SVC()),
    ("KNN", KNeighborsClassifier()),
    ("Neural Network", MLPClassifier())
]


# Dummy implementation of signal pre-processing
def pre_process_signal(eda_in):
    eda_out = eda_in
    return eda_out


# Dummy implementation of feature extraction routing
def feat_extract_dummy(eda_in):
    feat = []

    feat.append(np.mean(eda_in))
    feat.append(np.std(eda_in))
    feat.append(np.median(eda_in))
    feat.append(np.max(eda_in))
    feat.append(np.min(eda_in))
    return np.asarray(feat)


# Dummy implementation of feature selection routine
def feature_selection(train_data_in, val_data_in):
    train_data_out = train_data_in
    val_data_out = val_data_in
    return train_data_out, val_data_out



################################################################
####################Description of the data#########333#########
# first column: Subject Index
# Column from 1 to 129: EDA signal (slightly pre-processed)
# Column 130: Class (either 1 for no pain, or -1 for high pain)
################################################################


data_matrix = np.loadtxt('data/pain_dataset_eda_04.csv', dtype=float, delimiter=',')

# Separate the features and the labels
X_raw = data_matrix[:, 1:-1]  # get all columns from 1 to second last as features
y = data_matrix[:, -1]  # get the last column as labels

# Split the data into training and test sets (70-30 split)
X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.30, random_state=42)

X_feat_val = []
for i in range(len(X_raw_test)):
    X_feat_val.append(feat_extract_dummy(X_raw_test[i]))
X_feat_val = np.asarray(X_feat_val)

X_feat_train = []
for i in range(len(X_raw_train)):
    X_feat_train.append(feat_extract_dummy(X_raw_train[i]))
X_feat_train = np.asarray(X_feat_train)

# feature selection
X_feat_train, X_feat_val = feature_selection(X_feat_train, X_feat_val)

cross_val_scores = []
train_accuracies = []
val_accuracies = []

for clf_name, clf in classifiers:
    # Training
    clf.fit(X_feat_train, y_train)

    # Cross Validation
    cv_scores = cross_val_score(clf, X_feat_train, y_train, cv=5)  # 5 folds cross validation
    print(f'Cross Validation Scores for {clf_name}: {cv_scores}')
    avg_cv_score = np.mean(cv_scores)
    print(f'Average Cross Validation Score for {clf_name}: {avg_cv_score}')
    cross_val_scores.append(avg_cv_score)  # Append average CV score to list

    # Evaluating
    y_train_hat = clf.predict(X_feat_train)
    train_accuracy = np.mean(y_train_hat == y_train)
    print(f'Train Accuracy ({clf_name}): {train_accuracy}')
    train_accuracies.append(train_accuracy)  # Append train accuracy to list

    y_val_hat = clf.predict(X_feat_val)
    val_accuracy = np.mean(y_val_hat == y_test)
    print(f'Validation Accuracy ({clf_name}): {val_accuracy}')
    val_accuracies.append(val_accuracy)  # Append validation accuracy to list
# Below am using the model to generate new samples.

names = [clf_name for clf_name, _ in classifiers]

# Plotting cross validation scores
plt.figure(figsize=(10,5))
plt.bar(names, cross_val_scores)
plt.title('Mean Cross Validation Scores')
plt.show()

# Plotting train accuracies
plt.figure(figsize=(10,5))
plt.bar(names, train_accuracies)
plt.title('Train Accuracies')
plt.show()

# Plotting validation accuracies
plt.figure(figsize=(10,5))
plt.bar(names, val_accuracies)
plt.title('Validation Accuracies')
plt.show()