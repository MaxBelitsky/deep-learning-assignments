import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(confusion_matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt=".0f", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True labels")
    plt.xlabel("Predictions")
    plt.show()


def plot_taining_history(logging_info):
    plt.figure(figsize=(10, 5))
    plt.plot(logging_info['train_losses'], label='train_loss', marker='o')
    plt.plot(logging_info['validation_losses'], label='val_loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
