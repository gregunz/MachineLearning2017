# -*- coding: utf-8 -*-
"""
Tuning parameters and features to get the best out of them 
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from csv_helpers import list_to_csv
from cross_validation import cross_validation_ridge, cv_with_list

def range_mask(length, seq):
    return np.array([i in seq for i in range(length)])

def index_combinations(n, m, with_repetition=False, with_itself=False):
    return [(i, j) for i in range(n) for j in range(m) if (with_repetition or j >= i) and (with_itself or j != i)]

def find_useless_features(ys, xs, lambdas, k_fold=4, filename="useless_features"):
    seed = np.random.randint(0, 10000)
    print(seed)
    for y, x, lambda_, idx in zip(ys, xs, lambdas, range(len(ys))):
        improvements = []
        num_features = x.shape[1]
        for f_idx in tqdm(list(range(num_features)), ncols=100):
            mask = ~range_mask(num_features, [f_idx])
            score_with = np.array(cross_validation_ridge(y, x, k_fold, lambda_, seed)).mean()
            score_without = np.array(cross_validation_ridge(y, x[:, mask], k_fold, lambda_, seed)).mean()
            impr = score_without - score_with
            #if impr > 0:
            improvements.append(impr)
        list_to_csv(improvements, "data/tuning/{f}_{s}_{i}.csv".format(f=filename, s=seed, i=idx))

def find_best_combinations(ys, xs, lambdas, functions, functions_name, k_fold=4):
    seed = np.random.randint(0, 10000)
    print(seed)
    for fn, fn_name in zip(functions, functions_name):
        for y, x, lambda_, x_idx in zip(ys, xs, lambdas, range(len(ys))):

            improvements = []
            score_without = np.array(cross_validation_ridge(y, x, k_fold, lambda_, seed)).mean()
            combs = list(combinations(x.shape[1], x.shape[1]))
            
            for i, j in tqdm(combs, ncols=100):
                    
                comb = fn(x[:, i], x[:, j]).reshape(x.shape[0], 1)
                x_with = np.concatenate((x, comb), axis=1)
                score_with = np.array(cross_validation_ridge(y, x_with, k_fold, lambda_, seed)).mean()

                impr = score_with - score_without
                improvements.append(impr)

            list_to_csv(improvements, "data/tuning/best_{f}_{s}_{i}.csv".format(f=fn_name, s=seed, i=x_idx))
            list_to_csv(combs, "data/tuning/best_{f}_{s}_{i}.csv".format(f=fn_name, s=seed, i=x_idx))

def plot_mult_dif(ys, xs1, xs2, lambdas, k_fold=4, iters=1, filename=None):
    seed = np.random.randint(0, 10000)
    data = np.array([cv_with_list(ys, xs, lambdas, k_fold, iters=iters, seed=seed) for xs in [xs1, xs2]])
    if filename != None:
        np.save("data/matrices/{}.npy".format(filename), data)
        
    f, axes = plt.subplots(1, data.shape[2], figsize=(15, 10))
    for i, axis in enumerate(list(axes)):
        axis.boxplot(data[:, :, i].T, 0, '', showmeans=True)
    
    if filename != None:
        plt.savefig("data/plots/{}.png".format(filename))
    
def plot_mult_impr(ys, xs1, xs2, lambdas, k_fold=4, iters=1, filename=None):
    plot_mult_dif(ys, xs1, [np.concatenate(x, axis=1) for x in zip(xs1, xs2)], lambdas, k_fold, iters, filename)



'''
def check_dif(ys, xs1, xs2, lambdas, k_fold=4):
    seed = np.random.randint(0, 10000)
    scores = np.array([cv_with_list(ys, xs, lambdas, k_fold, seed=seed) for xs in [xs1, xs2]])
    return scores[1] - scores[0]

def check_multiple_dif(ys, xs1, xs2, lambdas, k_fold=4, iters=1):
    dim = len(xs1)
    old, new = np.array([0] * dim), np.array([0] * dim)
    for i in range(iters):
        impr = check_dif(ys, xs1, xs2, lambdas, k_fold)
        old += impr < 0
        new += impr > 0
    return old, new

def check_improvement(ys, xs1, xs2, lambdas, k_fold=4):
    return check_dif(ys, xs1, [np.concatenate(x, axis=1) for x in zip(xs1, xs2)], lambdas, k_fold)

def check_multiple_impr(ys, xs1, xs2, lambdas, k_fold=4, iters=1):
    return check_multiple_dif(ys, xs1, [np.concatenate(x, axis=1) for x in zip(xs1, xs2)], lambdas, k_fold, iters)
'''