results = {'objective': {'NB': {'reading': {'none': {'val_acc': 0.5384615384615384, 'val_stds': 0.23709284626803753, 'f1_val_means': 0.41025641025641024, 'f1_val_stds': 0.26646935501059654, 'auc_val_means': 0.5384615384615384, 'auc_val_stds': 0.23709284626803753, 'train_acc': 0.6282051282051282, 'train_stds': 0.03452028722522115, 'auc_train_means': 0.6282051282051283, 'f1_train_means': 0.6151554837580245, 'fpr': [0., 0., 1.], 'tpr': [0., 1., 1.], 'thresholds': [1.99559813, 0.99559813, 0.41912083]}}}}}

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Extract fpr, tpr for the 'NB' classifier, 'reading' task, and 'none' feature selection method
fpr = results['objective']['NB']['reading']['none']['fpr']
tpr = results['objective']['NB']['reading']['none']['tpr']

# Plot the ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % results['objective']['NB']['reading']['none']['auc_val_means'])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()