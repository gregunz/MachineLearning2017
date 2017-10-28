# -*- coding: utf-8 -*-
"""
Implementations
"""

import numpy as np
from helpers import batch_iter
from costs import compute_loss, calculate_loss
from gradients import compute_gradient

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    assert y.shape[0] == tx.shape[0], "#rows of y and tx must be equal"
    assert tx.shape[1] == initial_w.shape[0], "#col of tx must be equal to the #rows of initial_w"
    assert max_iters >= 0, "max_iters must be non-negative"

    # Define parameters to store w and loss
    w = initial_w
    loss = compute_loss(y, tx, w)

    for n_iter in range(max_iters):

        # compute loss, gradient
        grad, e = compute_gradient(y, tx, w, fn="mse")
        loss = calculate_loss(e)

        # gradient w by descent update
        w -= gamma * grad

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""
    
    assert y.shape[0] == tx.shape[0], "#rows of y and tx must be equal"
    assert tx.shape[1] == initial_w.shape[0], "#col of tx must be equal to the #rows of initial_w"
    assert max_iters >= 0, "max_iters must be non-negative"

    # Define parameters to store w and loss
    w = initial_w
    loss = compute_loss(y, tx, initial_w)

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            
            # compute a stochastic gradient and loss
            grad, e = compute_gradient(y_batch, tx_batch, w, fn="mse")
            loss = calculate_loss(e, fn="mse")

            # update w through the stochastic gradient update
            w -= gamma * grad

    return w, loss

def least_squares(y, tx):
    """compute the least squares solution using the normal equations"""

    assert y.shape[0] == tx.shape[0], "y and tx must have the same number of rows"

    a = tx.T @ tx
    b = tx.T @ y
    
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w, fn="mse")
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    assert y.shape[0] == tx.shape[0], "y and tx must have the same number of rows"
    assert lambda_ >= 0 and lambda_ <= 1, "incorrect lambda value (0 >= lambda >= 1)"

    N = len(y)
    lambdaI = (lambda_ * 2 * N) * np.eye(tx.shape[1])
    a = (tx.T @ tx) + lambdaI
    b = tx.T @ y

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w, fn="mse")
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):

    assert y.shape[0] == tx.shape[0], "#rows of y and tx must be equal"
    assert tx.shape[1] == initial_w.shape[0], "#col of tx must be equal to the #rows of initial_w"
    assert max_iters >= 0, "max_iters must be non-negative"

    w = initial_w
    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w, fn="sig")
        w -= gamma * grad

    loss = compute_loss(y, tx, w, fn="sig")
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    assert y.shape[0] == tx.shape[0], "#rows of y and tx must be equal"
    assert tx.shape[1] == initial_w.shape[0], "#col of tx must be equal to the #rows of initial_w"
    assert max_iters >= 0, "max_iters must be non-negative"

    w = initial_w
    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w, fn="sig") + 2 * lambda_ * w
        w -= gamma * grad

    loss = compute_loss(y, tx, w, fn="sig") + lambda_ * w.T @ w / 2
    return w, loss
