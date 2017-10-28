# -*- coding: utf-8 -*-
"""some helper functions for project 1."""

import numpy as np
from implementations import ridge_regression

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = data @ weights
    
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred

def predict_with_ridge(ys, xs_train, xs_test, lambdas, data_masks):
    train_size = np.sum([x.shape[0] for x in xs_train])
    test_size = np.sum([x.shape[0] for x in xs_test])
    y_sub = np.zeros(test_size)
    for y, x_tr, x_te, lambda_, mask in zip(ys, xs_train, xs_test, lambdas, data_masks):
        w = ridge_regression(y, x_tr, lambda_)
        y_sub[mask[train_size:]] = predict_labels(w, x_te)
    return y_sub