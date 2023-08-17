import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pickle
import seaborn as sns

import os

# Set the path to the files
path = "/Users/juanky/Documents/nokia_data/new_results/"

# Load results
methods = ['NB','SVM','RFC']
tasks = ["stroop","reading","subtraction"]
feature_selections = ['none', 'MI', 'SFS', 'BFS', 'ANOVA', 'PCA']

results = []
for method in methods:
    for task in tasks:
        file_name = f"{path}objective_{method}_{task}_eda_.txt"
        with open(file_name, "rb") as f:
            results.extend(pickle.load(f))

# Extract mean values and standard deviations
mean_values = {method: {fs: {task: [] for task in tasks} for fs in feature_selections} for method in methods}
std_values = {method: {fs: {task: [] for task in tasks} for fs in feature_selections} for method in methods}

for result in results:
    for method in methods:
        if method in result['objective']:
            for fs in feature_selections:
                for task in tasks:
                    if task in result['objective'][method] and fs in result['objective'][method][task]:
                        mean_values[method][fs][task].append(result['objective'][method][task][fs]['val_acc'])
                        std_values[method][fs][task].append(result['objective'][method][task][fs]['val_stds'])


for method in mean_values:
    print(f"Method: {method}")
    for fs in mean_values[method]:
        print(f"  Feature selection: {fs}")
        for task in mean_values[method][fs]:
            avg_mean = np.mean(mean_values[method][fs][task])
            print(f"    Task: {task}, Average Mean Validation Accuracy: {avg_mean}")

# Printing standard deviation values
for method in std_values:
    print(f"Method: {method}")
    for fs in std_values[method]:
        print(f"  Feature selection: {fs}")
        for task in std_values[method][fs]:
            avg_std = np.mean(std_values[method][fs][task])
            print(f"    Task: {task}, Average Standard Deviation: {avg_std}")



'''
# Create plot
x = np.arange(len(tasks))
width = 0.1  # width of the bars

# Use seaborn styles
# Use seaborn styles
# Use seaborn styles
sns.set_style("whitegrid")

# Create a figure with subplots
fig, axs = plt.subplots(nrows=len(methods), figsize=(15, 15))

# Adjust the space between subplots and at the bottom for the legend
fig.subplots_adjust(hspace=0.3, bottom=0.3)

# Define a monochrome color palette
palette = sns.color_palette("Blues", len(feature_selections))

# Add data to the plot
for i, method in enumerate(methods):
    ax = axs[i]
    for j, fs in enumerate(feature_selections):
        means = [np.mean(mean_values[method][fs][task]) for task in tasks]
        stds = [np.mean(std_values[method][fs][task]) for task in tasks]
        ax.bar(x - width/2 + j*width, means, width, label=f"{fs}", yerr=stds, color=palette[j])
    ax.set_ylabel(f'Validation Accuracy ({method})', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=19, ha="left")
    ax.tick_params(axis='x', labelbottom=True)

    # Only show the horizontal grid
    ax.xaxis.grid(False)

# Set common labels

# Add a single legend for the whole figure
handles, labels = axs[-1].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='center right', ncol=1, fontsize=19, title='Feature Selection', title_fontsize='20')

# Show the plot with a tight layout
plt.tight_layout()

# Adjust the layout of the figure taking into account the legend
fig.subplots_adjust(right=0.85)  # Leave space for the legend

plt.show()

'''