import matplotlib as matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


data_list = []  # We'll store each task's data in this list

parent_dir = "/Users/juanky/Documents/nokia_data"

# Iterate over users
for user in os.listdir(parent_dir):
    user_dir = os.path.join(parent_dir, user)
    # Make sure it's a directory
    if os.path.isdir(user_dir):
        # Iterate over tasks
        for task in os.listdir(user_dir):
            task_dir = os.path.join(user_dir, task)
            if os.path.isdir(task_dir):
                # Define the paths to the csv files
                for measures in os.listdir(task_dir):

                    if measures == "bvp.csv":
                        bvp_path = os.path.join(task_dir, "bvp.csv")
                        bvp = pd.read_csv(bvp_path, names=['Time', 'BVP'])

                    if measures == 'eda.csv':
                        eda_path = os.path.join(task_dir, "eda.csv")
                        eda = pd.read_csv(eda_path, names=['Time', 'EDA'])

                    if measures == 'temp.csv':
                        temp_path = os.path.join(task_dir, "temp.csv")
                        temp = pd.read_csv(temp_path, names=['Time', 'Temp'])

                # Merging all measurements based on 'Time' column
                data = pd.merge(pd.merge(bvp, eda, on='Time'), temp, on='Time')
                data.fillna(data.mean(), inplace=True)

                # Add user, task, and task difficulty information
                data['User'] = user
                data['Task'] = task[:-2]  # assuming task names are like 'reading0', 'reading1', etc.
                data['Difficulty'] = int(task[-1])  # '0' for easy, '1' for hard

                # Append this data to the list
                data_list.append(data)

# Combine all data into a single DataFrame
all_data = pd.concat(data_list, ignore_index=True)

"""
# Descriptive statistics
print(all_data.describe())

# Print the unique tasks and difficulty levels
print(all_data['Task'].unique())
print(all_data['Difficulty'].unique())

# Create box plots for each measure by task and difficulty level
for measure in ['BVP', 'EDA', 'Temp']:
    plt.figure(figsize=(12, 6))

    # Boxplot for task
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Task', y=measure, data=all_data)
    plt.title(f'{measure} by Task')

    # Boxplot for difficulty
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Difficulty', y=measure, data=all_data)
    plt.title(f'{measure} by Difficulty Level')

    plt.tight_layout()
    plt.show()

# Correlation Matrix
corr = all_data[['BVP', 'EDA', 'Temp']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


print(all_data['Task'].unique())
print(all_data['Difficulty'].unique())

# Calculate and print median and quartiles for each measure by task and difficulty level
for measure in ['BVP', 'EDA', 'Temp']:
    print(f'\n{measure} by Task:')
    print(all_data.groupby('Task')[measure].describe())

    print(f'\n{measure} by Difficulty Level:')
    print(all_data.groupby('Difficulty')[measure].describe())

# Calculate and print correlation matrix
corr = all_data[['BVP', 'EDA', 'Temp']].corr()
print("\nCorrelation matrix:\n", corr)
"""




#feature engineering

all_data['BVP_mean'] = all_data.groupby(['User', 'Task', 'Difficulty'])['BVP'].transform('mean')
all_data['BVP_median'] = all_data.groupby(['User', 'Task', 'Difficulty'])['BVP'].transform('median')
all_data['BVP_max'] = all_data.groupby(['User', 'Task', 'Difficulty'])['BVP'].transform('max')
all_data['BVP_min'] = all_data.groupby(['User', 'Task', 'Difficulty'])['BVP'].transform('min')
all_data['BVP_range'] = all_data['BVP_max'] - all_data['BVP_min']
all_data['BVP_std'] = all_data.groupby(['User', 'Task', 'Difficulty'])['BVP'].transform('std')
all_data['BVP_var'] = all_data.groupby(['User', 'Task', 'Difficulty'])['BVP'].transform('var')


# For EDA
all_data['EDA_mean'] = all_data.groupby(['User', 'Task', 'Difficulty'])['EDA'].transform('mean')
all_data['EDA_median'] = all_data.groupby(['User', 'Task', 'Difficulty'])['EDA'].transform('median')
all_data['EDA_max'] = all_data.groupby(['User', 'Task', 'Difficulty'])['EDA'].transform('max')
all_data['EDA_min'] = all_data.groupby(['User', 'Task', 'Difficulty'])['EDA'].transform('min')
all_data['EDA_range'] = all_data['EDA_max'] - all_data['EDA_min']
all_data['EDA_std'] = all_data.groupby(['User', 'Task', 'Difficulty'])['EDA'].transform('std')
all_data['EDA_var'] = all_data.groupby(['User', 'Task', 'Difficulty'])['EDA'].transform('var')

# For Temp
all_data['Temp_mean'] = all_data.groupby(['User', 'Task', 'Difficulty'])['Temp'].transform('mean')
all_data['Temp_median'] = all_data.groupby(['User', 'Task', 'Difficulty'])['Temp'].transform('median')
all_data['Temp_max'] = all_data.groupby(['User', 'Task', 'Difficulty'])['Temp'].transform('max')
all_data['Temp_min'] = all_data.groupby(['User', 'Task', 'Difficulty'])['Temp'].transform('min')
all_data['Temp_range'] = all_data['Temp_max'] - all_data['Temp_min']
all_data['Temp_std'] = all_data.groupby(['User', 'Task', 'Difficulty'])['Temp'].transform('std')
all_data['Temp_var'] = all_data.groupby(['User', 'Task', 'Difficulty'])['Temp'].transform('var')

#Derivative features
all_data['BVP_delta'] = all_data.groupby(['User', 'Task', 'Difficulty'])['BVP'].diff()
all_data['EDA_delta'] = all_data.groupby(['User', 'Task', 'Difficulty'])['EDA'].diff()
all_data['Temp_delta'] = all_data.groupby(['User', 'Task', 'Difficulty'])['Temp'].diff()


#encoding categorical features
all_data = pd.get_dummies(all_data, columns=['Task'])


pd.set_option('display.max_columns', None)
print(all_data)




classifiers = [
    ("Logistic Regression", LogisticRegression(penalty='l2', C=1.0)),  # C=1.0 is default value
    ("Random Forest", RandomForestClassifier()),
    ("SVM", SVC()),
    ("KNN", KNeighborsClassifier()),
    ("Neural Network", MLPClassifier())
]





#data_matrix = np.loadtxt('data/pain_dataset_eda_04.csv', dtype=float, delimiter=',')

X_raw = all_data.drop(['Task_reading', 'Task_stroop', 'Task_subtraction'], axis=1)

conditions = [
    (all_data['Task_reading'] == True),
    (all_data['Task_stroop'] == True),
    (all_data['Task_subtraction'] == True)
    ]
values = ['Reading', 'Stroop', 'Subtraction']
all_data['Task'] = np.select(conditions, values, default='None')

y = all_data['Task']

# Split the data into training and test sets (70-30 split)
X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.30, random_state=42)

# Preprocess data
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_raw_train)

X_raw_train = imputer.transform(X_raw_train)
X_raw_test = imputer.transform(X_raw_test)

cross_val_scores = []
train_accuracies = []
val_accuracies = []

for clf_name, clf in classifiers:
    # Training
    clf.fit(X_raw_train, y_train)

    # Cross Validation
    cv_scores = cross_val_score(clf, X_raw_train, y_train, cv=5)
    print(f'Cross Validation Scores for {clf_name}: {cv_scores}')
    avg_cv_score = np.mean(cv_scores)
    print(f'Average Cross Validation Score for {clf_name}: {avg_cv_score}')
    cross_val_scores.append(avg_cv_score)

    # Evaluating
    y_train_hat = clf.predict(X_raw_train)
    train_accuracy = np.mean(y_train_hat == y_train)
    print(f'Train Accuracy ({clf_name}): {train_accuracy}')
    train_accuracies.append(train_accuracy)

    y_test_hat = clf.predict(X_raw_test)
    test_accuracy = np.mean(y_test_hat == y_test)
    print(f'Test Accuracy ({clf_name}): {test_accuracy}')
    val_accuracies.append(test_accuracy)

names = [clf_name for clf_name, _ in classifiers]

# Plotting cross validation scores
plt.figure(figsize=(10, 5))
plt.bar(names, cross_val_scores)
plt.title('Mean Cross Validation Scores')
plt.show()

# Plotting train accuracies
plt.figure(figsize=(10, 5))
plt.bar(names, train_accuracies)
plt.title('Train Accuracies')
plt.show()

# Plotting validation accuracies
plt.figure(figsize=(10, 5))
plt.bar(names, val_accuracies)
plt.title('Test Accuracies')
plt.show()
