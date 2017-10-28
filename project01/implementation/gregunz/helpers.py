# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def range_mask(length, seq):
    return np.array([i in seq for i in range(length)])

def create_pairs(n, m, with_repetition=False, with_itself=False):
    return [(i, j) for i in range(n) for j in range(m) if (with_repetition or j >= i) and (with_itself or j != i)]

def all_combinations_of(xs, fn, combs):
    combinations = []
    for i, pairs in enumerate(combs):
        combinations.append(combinations_of(xs[i], fn, pairs))
    return combinations

def combinations_of(x, fn, pairs):
    if len(pairs) > 0:
        combinations = [fn(x[:, a], x[:, b]).reshape((x.shape[0], 1)) for a, b in pairs]
        return np.concatenate(combinations, axis=1)
    return np.array([])

def separate_train(xs, train_size, data_masks):
    test_size = np.sum([m.sum() for m in data_masks]) - train_size
    train_mask = np.r_[[True] * train_size, [False] * test_size]
    xs_train_size = [(mask & train_mask).sum() for mask in data_masks]

    xs_train = [f[:size] for f, size in zip(xs, xs_train_size)]
    xs_test  = [f[size:] for f, size in zip(xs, xs_train_size)]

    return xs_train, xs_test
