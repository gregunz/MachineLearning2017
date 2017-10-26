# -*- coding: utf-8 -*-
"""
Tuning parameters and features to get the best out of them 
"""

import numpy as np
import tqdm

from csv_helpers import list_to_csv
from cross_validation import cross_validation_ridge

def range_mask(length, seq):
    return np.array([i in seq for i in range(length)])

def find_useless_features(ys, xs, lambdas, k_fold=4, filename="useless_features"):
    seed = np.random.randint(0, 10000)
    print(seed)
    for y, x, lambda_ in zip(ys, xs, lambdas):
        x_useless_features = []
        num_features = x.shape[1]
        for i in tqdm(list(range(num_features)), ncols=100):
            mask = ~range_mask(num_features, [i])
            score_with = np.array(cross_validation_ridge(y, x, k_fold, lambda_, seed)).mean()
            score_without = np.array(cross_validation_ridge(y, x[:, mask], k_fold, lambda_, seed)).mean()
            impr = score_without - score_with
            if impr > 0:
                x_useless_features.append(i)
                print(i)
        list_to_csv(x_useless_features, "{f}_{s}.csv".format(f=filename, s=seed))