# -*- coding: utf-8 -*-
"""
Helper Functions
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inv_log(x):
    return np.log(1 / (1 + x))

def abs_dif(x, y):
    return np.abs(x - y)

def mult(x, y, deg=1):
    return np.power(x, deg) * np.power(y, deg)
