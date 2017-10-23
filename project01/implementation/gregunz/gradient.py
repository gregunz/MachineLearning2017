# -*- coding: utf-8 -*-
"""
Gradient
"""
from helpers import sigmoid

def compute_gradient(y, tx, w, fn="mse"):
    """Compute the (stochastic) gradient given a cost function."""

    if fn.lower() == "mse":
        N = len(y)
        e = y - (tx @ w)
        grad = - (tx.T @ e) / N
        return grad, e

    if fn.lower() == "sig":
        pred = sigmoid(tx @ w)
        grad = tx.T @ (pred - y)
        return grad

    raise NameError("Not such fn exists \"" + fn + "\"")
