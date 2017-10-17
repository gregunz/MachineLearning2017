# -*- coding: utf-8 -*-
"""
Helper functions for cross validation
"""

import numpy as np

def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def build_k_fold_sets(y, x, k_fold,  seed=1):

    # build k_indices
    k_indices = build_k_indices(y, k_fold, seed)

    # get k'th subgroup in validation, others in train
    for k in range(k_fold):
        va_indices = k_indices[k]
        tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
        tr_indices = tr_indices.reshape(-1)

        y_va = y[va_indices]
        y_tr = y[tr_indices]
        x_va = x[va_indices]
        x_tr = x[tr_indices]
        yield x_tr, x_va, y_tr, y_va
