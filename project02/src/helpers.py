# Helper functions

import os

import numpy as np
from natsort import natsorted

from helpers_image import load_image, img_to_patches, img_to_gray, apply_clahe, apply_gamma_correction


def ls_rec_path(path):
    return natsorted(["{}/{}".format(root, f) for root, _, files in os.walk(path) for f in files])


def path_to_data(path, sample_size=None):
    paths = ls_rec_path(path)
    if sample_size is None:
        sample_size = len(paths)
    return np.array([np.array(load_image(p)) for p in paths[:sample_size]])


def image_pipeline(path, sample_img, normalized, grayscale, clahe, gamma, rotations, patch_size, stride):
    X = path_to_data(path, sample_img)

    if len(X.shape) < 4:
        X = X.reshape(X.shape + (1,))
    h, w, _ = X.shape[1:]

    if grayscale:
        X = np.array([img_to_gray(x) for x in X])

    if normalized:
        assert grayscale, 'normalization only if grayscale'
        X = (X - X.mean()) / X.std()
        X = np.array([((x - x.min()) / (x.max() - x.min())) * 255 for x in X])

    if clahe:
        assert grayscale, 'apply clahe only if grayscale'
        X = np.array([apply_clahe(x) for x in X])

    if gamma:
        assert grayscale, 'apply gamma only if grayscale'
        X = np.array([apply_gamma_correction(x) for x in X])

    X = X.astype(np.float32) / 255

    if rotations:
        rotations = rotations.copy()
        rotations.append(0)  # in order to keep the non-rotated image more easily
        for angle in rotations:
            assert angle % 90 == 0, 'atm only 90° angle are supported'
        X = np.array([np.rot90(x, angle // 90) for x in X for angle in rotations])

    if h > patch_size:
        X = np.concatenate([img_to_patches(x, patch_size, stride) for x in X])
    return X, h, w


def new_file_path(path, filename, ext, n):
    assert path[-1] is '/' or path[-1] is '\\', 'directory path should end with (back-)slash'

    file_path = path + filename + "_" + str(n).zfill(3) + ext
    if os.path.exists(file_path):
        return new_file_path(path, filename, ext, n + 1)
    else:
        return file_path
