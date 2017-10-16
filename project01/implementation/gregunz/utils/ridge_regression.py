# -*- coding: utf-8 -*-
"""
Ridge Regression
"""

import numpy as np

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = len(y)
    lambdaI = (lambda_ * 2 * N) * np.eye(tx.shape[1])
    a = (tx.T @ tx) + lambdaI
    b = tx.T @ y
    return np.linalg.solve(a, b)
