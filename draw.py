import os

import matplotlib.pyplot as plt

import numpy as np


def draw_curve(train_log, test_log, train_prec,train_rec,train_f1,test_prec,test_rec,test_f1,train_loss, test_loss , args):
    x = list(range(args.num_epoch+1))
    
       
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

    # Plot something on each subplot
    # Subplot 1
    axes[0].plot(x, train_log, c="darkolivegreen", label="Train Accuracy")
    axes[0].plot(x, test_log, c="slateblue", label="Validation Accuracy")
    axes[0].set_ylim((0, 100))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Subplot 2
    axes[1].plot(x, train_loss, c="crimson", label="Training Loss")
    axes[1].plot(x, test_loss, c="darkgoldenrod", label="Validation Loss")
    axes[1].set_ylim((0, 2))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    # Subplot 3
    axes[2].plot(x, train_f1, c="darkcyan", label="Micro-F1 Score")
    axes[2].plot(x, test_f1, c="indigo", label="Validation Micro-F1 Score")
    axes[2].set_ylim((0, 1))
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    #plt.show()
    plt.savefig(args.img_dir+'/graph.png', dpi=400)