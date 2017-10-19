from helper import *

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.dot(tx, w)
    return (-1/len(y)) * np.dot(tx.T, e)

def compute_loss(y, tx, w):
    """Calculate the loss."""
    e = y - np.dot(tx, w) #compute the error
    return np.dot(e.T,e) / (2*len(y))

### Asked method to implement
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
        print("Gradient Descent({bi}/{ti}): loss={l}, w={w}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w=w))
    return loss, w

def least_squares_MBGD(y, tx, initial_w, max_iters, gamma, batch_size):
    """Linear regression using mini batch gradient descent this function wasn't asked"""
    w = initial_w
    for n_iter, (y_batch, tx_batch) in enumerate(batch_iter(y, tx, batch_size, nb_epochs)):
        loss = compute_loss(y_batch, tx_batch, w)
        gradient = compute_loss(y_batch, tx_batch, w)
        w = w - gamma * gradient
        print("Gradient Descent({bi}/{ti}): loss={l}, w={w}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w=w))

    return losses, ws

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    return least_squares_MBGD(y, tx, initial_w, max_iters, gamma, batch_size = 1)
    
def least_squares(y, tx):
    """Least squares regression using normal equations"""
    square = tx.T.dot(tx)
    w_opt = np.linalg.solve(square, tx.T.dot(y))  
    return compute_loss(y, tx, w_opt), w_opt

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    raise NotImplementedError("Function not implemented")
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    raise NotImplementedError("Function not implemented")

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized  logistic  regression  using  gradient  descent or SGD"""
    raise NotImplementedError("Function not implemented")
    