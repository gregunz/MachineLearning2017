# -*- coding: utf-8 -*-
"""
Data Pre-processing
"""

import numpy as np        

def replace_invalid(x, mask_invalid, replace_by=None):

    # construct the masked array
    masked_x = np.ma.array(x, mask=~mask_invalid)

    # default filled values are zeroes
    fill_value = [0] * x.shape[1]

    # replace with the mean
    if replace_by.lower() == "mean":
        fill_value = masked_x.mean(axis=0)

    # replace with the most frequent value
    if replace_by.lower() == "mf":
        fill_value = []
        for i in range(x.shape[1]):
            uniqw, inverse = np.unique(x[:, i][mask_invalid[:, i]], return_inverse=True)
            idx = np.argmax(np.bincount(inverse))
            fill_value.append(uniqw[idx])

    masked_x.set_fill_value(fill_value)

    return masked_x.filled()

def standardize(x):
    """Standardize the original data set."""
    x = x.copy()
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x