import numpy as np
from costs import compute_loss 
from build_polynomial import build_poly
from ridge_regression import ridge_regression

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    # get k'th subgroup in test, others in train
    te_indices = k_indices[k]
    tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indices = tr_indices.reshape(-1)

    y_te = y[te_indices]
    y_tr = y[tr_indices]
    x_te = x[te_indices]
    x_tr = x[tr_indices]

    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    # ridge regression
    w = ridge_regression(y_tr, tx_tr, lambda_)

    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_loss(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_loss(y_te, tx_te, w))

    return loss_tr, loss_te
