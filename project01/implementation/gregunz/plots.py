# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt

def display_feature(feature, bins=50, name=None):
    plt.title(name, bins)
    plt.hist(feature, bins=bins)

def display_features(x, mask, bins=50, cols=3, figsize=(20, 65)):
    _, axes = plt.subplots(int(np.ceil(x.shape[1] / cols)), cols, figsize=figsize)
    for i in range(x.shape[1]):
        feature = x[:, i][mask[:, i]]
        axes[i // cols, i % cols].hist(feature, bins=bins)
        axes[i // cols, i % cols].set_title("feature {}".format(i))
    plt.show()

def box_plot(data, filename=None):
    if filename != None:
        np.save("data/matrices/{}.npy".format(filename), data)
        
    f, axes = plt.subplots(1, data.shape[2], figsize=(15, 10))
    for i, axis in enumerate(list(axes)):
        axis.boxplot(data[:, :, i].T, 0, '', showmeans=True)
    
    if filename != None:
        plt.savefig("data/plots/{}.png".format(filename))
    
    plt.show()