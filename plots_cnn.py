import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data
data = {
    'temp': {
        'stroop': {
            'architecture1': {'train_acc': 0.922236025929451, 'val_acc': 0.9442857119441033},
            'architecture2': {'train_acc': 0.8568944096565246, 'val_acc': 0.6742857140302658},
            'architecture3': {'train_acc': 0.8038752402948297, 'val_acc': 0.5265169649020486}
        },
        'reading': {
            'architecture1': {'train_acc': 0.8913812172412873, 'val_acc': 0.8923902463912964},
            'architecture2': {'train_acc': 0.8952209919691085, 'val_acc': 0.7425365847349167},
            'architecture3': {'train_acc': 0.9025024371988634, 'val_acc': 0.6745480611043818}
        },
        'subtraction': {
            'architecture1': {'train_acc': 0.869972373843193, 'val_acc': 0.8472682946920395},
            'architecture2': {'train_acc': 0.8992265152931214, 'val_acc': 0.760829271376133},
            'architecture3': {'train_acc': 0.9392265158891678, 'val_acc': 0.6816585364937783}
        }
    }
}


# Restructuring the data
rows = []
for task, architectures in data['temp'].items():
    for arch, metrics in architectures.items():
        for metric_type, acc in metrics.items():
            rows.append([task, arch, metric_type, acc])

# Creating DataFrame
df = pd.DataFrame(rows, columns=['task', 'architecture', 'type', 'accuracy'])

# Creating a separate subplot for each task
tasks = df['task'].unique()
fig, axes = plt.subplots(nrows=1, ncols=len(tasks), figsize=(18, 6), sharey=True)

# Set up the color palette
palette = sns.color_palette("Blues_r")  # Using the dark ("_d") blues palette

for ax, task in zip(axes, tasks):
    df_task = df[df['task'] == task]
    sns.barplot(data=df_task, x='architecture', y='accuracy', hue='type', errorbar=None, palette=palette, ax=ax)
    ax.set_title(f'Accuracy for {task.capitalize()} Task')
    ax.set_xlabel('Architecture')
    ax.set_ylabel('Accuracy')

    # Display the y-axis labels on all plots
    ax.yaxis.set_tick_params(which='both', labelleft=True)

    # Moving the legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.suptitle('Temperature Signal', fontsize=20)

plt.tight_layout()
plt.show()
