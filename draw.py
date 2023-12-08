import os

import matplotlib.pyplot as plt

import numpy as np


def draw_curve(train_log, train_loss, epoch):
    x = list(range(epoch+1))
    
    # show acc
    plt.subplot(2, 1, 1)
    plt.plot(x, train_log, c="blue", label="accuracy")
    plt.ylim((0, 100))
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    #show loss
    plt.subplot(2, 1, 2)
    plt.plot(x[1:], train_loss, c="red", label="loss")
    plt.ylim((0, 4))
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()
