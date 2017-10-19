# -*- coding: utf-8 -*-
"""
Least Square
"""

import numpy as np


def least_squares(y, tx):
    """compute the least squares solution using the normal equations"""

    assert len(y) == len(tx), "y and tx must have the same number of rows"

    a = tx.T @ tx
    b = tx.T @ y
    return np.linalg.solve(a, b)
