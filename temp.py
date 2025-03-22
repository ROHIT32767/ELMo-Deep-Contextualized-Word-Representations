import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the confusion matrices from the image
train_conf_matrix = np.array([
    [22938, 265, 486, 259],
    [231, 23788, 30, 38],
    [417, 184, 22463, 909],
    [813, 192, 1189, 21798]
])

val_conf_matrix = np.array([
    [5781, 62, 137, 72],
    [54, 5844, 10, 5],
    [113, 48, 5615, 251],
    [191, 62, 290, 5465]
])

test_conf_matrix = np.array([
    [1747, 41, 66, 46],
    [34, 1834, 11, 21],
    [81, 31, 1614, 174],
    [99, 29, 188, 1584]
])

# Function to plot a heatmap
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

# Plot each confusion matrix
plot_confusion_matrix(train_conf_matrix, "Train Confusion Matrix")
plot_confusion_matrix(val_conf_matrix, "Validation Confusion Matrix")
plot_confusion_matrix(test_conf_matrix, "Test Confusion Matrix")
