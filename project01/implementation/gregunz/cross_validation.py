# -*- coding: utf-8 -*-
"""
Helper functions for cross validation
"""

import numpy as np
from predictions import predict_labels
from implementations import ridge_regression, logistic_regression

def build_k_indices(y, k_fold, seed=None):
    """build k indices for k-fold."""
    np.random.seed(seed)
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def build_k_fold_sets(y, x, k_fold,  seed=None):
    np.random.seed(seed)

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

def cross_validation_ridge(y_train, x_train, k_fold, lambda_, seed=None):
    np.random.seed(seed)
    scores = []
    for x_tr, x_va, y_tr, y_va in build_k_fold_sets(y_train, x_train, k_fold, seed):
        w, _ = ridge_regression(y_tr, x_tr, lambda_)
        y_te_pred = predict_labels(w, x_va)
        score = (y_te_pred == y_va).mean()
        scores.append(score)
    return np.array(scores)

def cross_validation_logistic(y_train, x_train, k_fold, initial_w, max_iters, gamma, seed=None):
    np.random.seed(seed)
    scores = []
    for x_tr, x_va, y_tr, y_va in build_k_fold_sets(y_train, x_train, k_fold, seed):
        w, _ = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)
        y_te_pred = predict_labels(w, x_va)
        score = (y_te_pred == y_va).mean()
        scores.append(score)
    return np.array(scores)

def cv_with_list(ys, xs, lambdas, k_fold=4, iters=1, seed=None, print_=False):
    if seed == None:
        seed = np.randon.randint(100000)
    xs_scores = []
    train_size = np.sum([x.shape[0] for x in xs])
    for y_tr, x_tr, lambda_ in zip(ys, xs, lambdas):
        scores = np.array([score  for i in range(iters) for score in cross_validation_ridge(y_tr, x_tr, k_fold, lambda_, seed=seed+i)])
        xs_scores.append(scores)
        if(print_):
            print("Test Error Mean = {}".format(scores.mean() * 100))
            print("Test Error St.D = {}".format(scores.std() * 100))
    return np.array(xs_scores).T
