# -*- coding: utf-8 -*-
"""
Split the dataset based on the given ratio
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    
    assert len(y) == len(tx), "x and y must have the same number of rows"
    assert ratio > 0 && ratio < 1, "ratio must inside the interval (0, 1), i.e. 0 > ratio > 1"

    # set seed
    np.random.seed(seed)
    
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te
