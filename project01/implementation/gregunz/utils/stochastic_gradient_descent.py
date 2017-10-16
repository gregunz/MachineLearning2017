# -*- coding: utf-8 -*-
"""
Stochastic Gradient Descent
"""

from helpers import batch_iter
from costs import compute_loss, calculate_loss
from gradient_descent import compute_gradient

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            
            # compute a stochastic gradient and loss
            grad, e = compute_gradient(y_batch, tx_batch, w)
            loss = calculate_loss(e, fn="mse")

            # update w through the stochastic gradient update
            w = w - gamma * grad
            
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
