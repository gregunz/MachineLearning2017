import numpy as np
from costs import compute_loss, calculate_loss

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
        grad, e = compute_gradient(y, tx, w)
        loss = calculate_loss(e)

        # gradient w by descent update
        w = w - gamma * grad

    return w, loss

#TODO: VERIFY THAT 'batch_size' ARGUMENT MUST BE HERE (it's not in the project description)
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    
    assert y.shape[0] == tx.shape[0], "#rows of y and tx must be equal"
    assert tx.shape[1] == initial_w.shape[0], "#col of tx must be equal to the #rows of initial_w"
    assert max_iters >= 0, "max_iters must be non-negative"

    # Define parameters to store w and loss
    w = initial_w
    loss = compute_loss(y, tx, initial_w)

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            
            # compute a stochastic gradient and loss
            grad, e = compute_gradient(y_batch, tx_batch, w)
            loss = calculate_loss(e, fn="mse")

            # update w through the stochastic gradient update
            w = w - gamma * grad

    return w, loss

def least_squares(y, tx):
    """compute the least squares solution using the normal equations"""

    assert len(y) == len(tx), "y and tx must have the same number of rows"

    a = tx.T @ tx
    b = tx.T @ y
    return np.linalg.solve(a, b)

def ridge_regression(y, tx, lambda_): 
    """implement ridge regression."""
    N = len(y)
    lambdaI = (lambda_ * 2 * N) * np.eye(tx.shape[1])
    a = (tx.T @ tx) + lambdaI
    b = tx.T @ y
    return np.linalg.solve(a, b)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    raise NotImplementedError
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    raise NotImplementedError