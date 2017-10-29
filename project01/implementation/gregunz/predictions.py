# -*- coding: utf-8 -*-
"""some helper functions for project 1."""

import numpy as np

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = data @ weights
    
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred