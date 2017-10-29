# -*- coding: utf-8 -*-
"""
Feature Engineering 
"""

import numpy as np

def build_poly(x, degree, roots=False):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    assert type(degree) == int, "degree must be of type int"
    assert degree >= 1, "degree must non-negative"

    poly = []
    for deg in range(1, degree + 1):
        if roots:
            deg = 1 / deg
        poly.append(np.power(x, deg))
    return np.concatenate(poly, axis=1)
