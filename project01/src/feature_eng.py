# -*- coding: utf-8 -*-
"""
Feature Engineering 
"""

import numpy as np
from functions import inv_log
from helpers import *
from preprocessing import *

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

def build_x(x_train, x_test, degree, root=None, replace_by="mf", f_mask=None,
            log_degree=None, inv_log_degree=None, tanh_degree=None, fn_log=True, fn_inv_log=True,
            fn_tanh=True, functions=None, invalid_value=-999, print_=False):
    
    if print_:
        print(degree, root, log_degree, inv_log_degree, tanh_degree, fn_log, fn_inv_log, fn_tanh, functions)
    
    assert f_mask == None or len(f_mask) == x.shape[1]
    assert log_degree == None or type(log_degree) == int
    assert inv_log_degree == None or type(inv_log_degree) == int
    
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    
    x = np.concatenate((x_train, x_test))
    
    # Preprocessing
    if print_:
        print("Starting pre-processing")
    
    if f_mask != None:
        x = x[:, f_mask]
    
    x = replace_invalid(x, x != invalid_value, replace_by="mf")

    x_non_negative = x - x.min(axis=0)
    x_std = standardize(x.copy())
    
    # Features Engineering
    # poly
    if print_:
        print("Starting poly")
    x = build_poly(x_std, degree)
    
    if tanh_degree != None:
        x_tanh = standardize(np.tanh(x_std))
        x = np.concatenate((x, build_poly(x_tanh, tanh_degree)), axis=1)
    if log_degree != None:
        x_log = standardize(np.log(1 + x_non_negative))
        x = np.concatenate((x, build_poly(x_log, log_degree)), axis=1)
    if inv_log_degree != None:
        x_inv_log = standardize(inv_log(x_non_negative))
        x = np.concatenate((x, build_poly(x_inv_log, inv_log_degree)), axis=1)
    if root != None:
        x = np.concatenate((x, build_poly(x_non_negative, root, roots=True)[:, x_non_negative.shape[1]:]), axis=1)
    
    # combinations with functions
    if print_:
        print("Starting combinations")
    if functions != None:
        x_comb = x_std.copy()
        if fn_tanh:
            x_tanh = standardize(np.tanh(x_std))
            x_comb = np.concatenate((x_comb, x_tanh), axis=1)
        if fn_log:
            x_log = np.log(1 + x_non_negative)
            x_comb = np.concatenate((x_comb, x_log), axis=1)
        if fn_inv_log:
            x_inv_log = standardize(inv_log(x_non_negative))
            x_comb = np.concatenate((x_comb, x_inv_log), axis=1)
        for fn in functions:
            x = np.concatenate((x, combinations_of(x_comb, fn, create_pairs(x_comb.shape[1], x_comb.shape[1]))), axis=1)

    x = np.concatenate((np.ones(x.shape[0]).reshape((x.shape[0], 1)), x), axis=1)
    
    if print_:
        print("Final shape: {}".format(x.shape))
    
    return x[:train_size], x[train_size:]