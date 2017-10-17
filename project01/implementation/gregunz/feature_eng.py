# -*- coding: utf-8 -*-
"""
Feature Engineering 
"""

import numpy as np

def indicator_features(mask_invalid):
    mask_features_with_invalid = np.any(mask_invalid == True, axis=0)
    mask_only_invalid = mask_invalid[:, mask_features_with_invalid]

    return mask_only_invalid.astype(int)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    assert type(degree) == type(0), "degree must be of type int"
    assert degree >= 0, "degree must be non-negative"

    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly
