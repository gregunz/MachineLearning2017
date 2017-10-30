# coding: utf-8

# Imports
import numpy as np

from helpers import *
from implementations import ridge_regression
from functions import abs_dif, mult 
from feature_eng import build_x
from predictions import predict_labels

# Loading the data sets
y_train, x_brute_train, _ = load_csv_data("train.csv")
_, x_brute_test, indices_test = load_csv_data("test.csv")
x_brute = np.concatenate((x_brute_train, x_brute_test))
train_size = x_brute_train.shape[0]
test_size = x_brute_test.shape[0]

# Constants
PHI_features = [15, 18, 20, 25, 28]
invalid_value = -999

# Mask to subdivide in different models
# Mask for the data (rows)
data_masks = [
    x_brute[:, 22] == 0,
    x_brute[:, 22] == 1,
    x_brute[:, 22] > 1
]
num_models = len(data_masks)

# Mask for the features (columns)
features_masks = [(x_brute[m].std(axis=0) != 0) & np.any(x_brute[m] != -999., axis=0) & ~range_mask(30, PHI_features) for m in data_masks]

# Separate X and Y using the masks
ys_train = [y_train[mask[:train_size]] for mask in data_masks]
xs_brute_train = [x_brute_train[d_m[:train_size]][:, f_m] for d_m, f_m in zip(data_masks, features_masks)]
xs_brute_test = [x_brute_test[d_m[train_size:]][:, f_m] for d_m, f_m in zip(data_masks, features_masks)]

# Models variables
degrees = [9, 11, 12]
roots = [3, 4, 3]
tanh_degrees = [3, 4, 3]
log_degrees = [3, 4, 3]
inv_log_degrees = [3, 4, 3]
fn_tanh = [True] * num_models
fn_log = [False] * num_models
fn_inv_log = [True] * num_models
functions = [[mult, abs_dif],] * num_models

# Hyper parameters
lambdas = [1e-03, 3.16e-03, 1e-02]

y_submission = np.zeros(test_size)

# Regression for each models
for i, mask in enumerate(data_masks):
    x_train, x_test = build_x(xs_brute_train[i], xs_brute_test[i], degrees[i], roots[i],
                              log_degree=log_degrees[i], inv_log_degree=inv_log_degrees[i],
                              fn_log=fn_log[i], fn_inv_log=fn_inv_log[i], functions=functions[i])
    
    w, _ = ridge_regression(ys_train[i], x_train, lambdas[i])
    
    mask = mask[train_size:]
    y_submission[mask] = predict_labels(w, x_test)
    
create_csv_submission(indices_test, y_submission, "final_submission.csv")
