# Helper functions

import os

import numpy as np
from natsort import natsorted

from helpers_image import load_image, img_to_patches, img_to_gray, apply_clahe, apply_gamma_correction, apply_rotations


def ls_rec_path(path):
    return natsorted(["{}/{}".format(root, f) for root, _, files in os.walk(path) for f in files])


def path_to_data(path, sample_size=None):
    paths = ls_rec_path(path)
    if sample_size is None:
        sample_size = len(paths)
    return np.array([np.array(load_image(p)) for p in paths[:sample_size]])


def image_pipeline(images_or_path, n_sample_img, grayscale, normalized, clahe, gamma, rotations, patch_size, stride,
                   overlapping):
    if type(images_or_path) is str:
        images = path_to_data(images_or_path, n_sample_img)
    else:
        images = images_or_path

    if len(images.shape) < 4:
        images = images.reshape(images.shape + (1,))
    h, w, _ = images.shape[1:]

    assert images.dtype == np.uint8, 'images should be of type uint8 with values between 0 and 255'

    n_channels = 1 if grayscale else 3

    if grayscale:
        images = np.array([img_to_gray(img) for img in images])

    if normalized:
        for channel in range(n_channels):
            images[:, :, :, channel] = \
                np.array([((img - img.min()) / (img.max() - img.min())) * 255 for img in images[:, :, :, channel]])

    if clahe:
        for channel in range(n_channels):
            images[:, :, :, channel] = np.array([apply_clahe(img) for img in images[:, :, :, channel]])

    if gamma:
        for channel in range(n_channels):
            images[:, :, :, channel] = np.array([apply_gamma_correction(img) for img in images[:, :, :, channel]])

    if len(rotations) > 0:
        rotations = [0] + rotations.copy()
        images = apply_rotations(images, rotations)

    if h > patch_size:
        images = np.concatenate([img_to_patches(img, patch_size, stride, overlapping=overlapping) for img in images])

    assert images.dtype == np.uint8, 'images should be of type uint8 with values between 0 and 255'  # sanity check

    images = images.astype(np.float32, copy=False) / 255

    return images, h, w


def new_file_path(path, filename, ext, n):
    assert path[-1] is '/' or path[-1] is '\\', 'directory path should end with (back-)slash'

    file_path = path + filename + "_" + str(n).zfill(3) + ext
    if os.path.exists(file_path):
        return new_file_path(path, filename, ext, n + 1)
    else:
        return file_path
