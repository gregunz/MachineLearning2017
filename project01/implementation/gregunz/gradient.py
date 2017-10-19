# -*- coding: utf-8 -*-
"""
Gradient
"""

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = len(y)
    e = y - (tx @ w)
    grad = - (tx.T @ e) / N
    return grad, e