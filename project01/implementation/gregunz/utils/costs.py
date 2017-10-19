# -*- coding: utf-8 -*-
"""
A function to compute the cost.
"""

import numpy as np

def compute_loss(y, tx, w, fn="mse"):
    """compute the loss given the function 'fn'"""

    assert len(y) == len(tx), "y and tx must have the same number of rows"

    if fn.lower() == "mse":
        N = len(y)
        e = y - (tx @ w)
        return (e @ e) / (2 * N)

    if fn.lower() == "rmse":
        return np.sqrt(2 * compute_loss(y, tx, w, fn="mse"))

    if fn.lower() == "mae":
        N = len(y)
        e = y - tx @ w
        return np.mean(np.abs(e)) / (2 * len(e))

    raise NameError("Not such fn exists \"" + fn + "\"")

def calculate_loss(e, fn="mse"):
    """Calculate the mse for vector e."""
    if fn.lower() == "mse":
        return (e @ e) / (2 * len(e))

    if fn.lower() == "rmse":
        return np.sqrt(2 * calculate_loss(e, fn="mse"))

    if fn.lower() == "mae":
        return np.mean(np.abs(e)) / (2 * len(e))

    raise NameError("Not such fn exists \"" + fn + "\"")
