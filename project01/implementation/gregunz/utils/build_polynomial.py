# -*- coding: utf-8 -*-
"""
Implement a polynomial basis function
"""

import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    assert type(degree) == type(0), "degree must be of type int"
    assert degree >= 0, "degree must be non-negative"

    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly