# Helper functions

import os

import numpy as np

from helpers_image import load_image, img_to_patches


def ls_rec_path(path):
    return sorted(["{}/{}".format(root, f) for root, _, files in os.walk(path) for f in files])


def path_to_data(path, sample_size=None):
    paths = ls_rec_path(path)
    if sample_size is None:
        sample_size = len(paths)
    return np.array([np.array(load_image(p)) for p in paths[:sample_size]])


def image_pipeline(path, sample_img, rotations, patch_size, stride):
    X = (path_to_data(path, sample_img) / 255).astype('float32')
    if len(X.shape) < 4:
        X = X.reshape(X.shape + (1,))
    h, w, _ = X.shape[1:]
    if h > patch_size:
        X = np.concatenate([img_to_patches(x, patch_size, stride) for x in X])
    return X, h, w