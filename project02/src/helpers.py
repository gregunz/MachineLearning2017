import json
import os

import numpy as np
from natsort import natsorted

from helpers_image import load_image, img_to_patches, img_to_gray, apply_clahe, apply_gamma_correction, apply_rotations


def ls_rec_path(path):
    """
    List recursively all files (paths) given a directory

    :param path: the directory
    :return: list of paths (str) sorted naturally (e.g.: img_1.jpg before img_12.jpg, note the missing leading zero)
    """
    return natsorted(["{}/{}".format(root, f) for root, _, files in os.walk(path) for f in files])


def path_to_data(path, sample_size=None):
    """
    Given a path load all files inside (assumed to be images) as numpy ndarray
    :param path: the path where images are stored
    :param sample_size: the number of images to load (ordered)
    :return: images (numpy ndarray of dimension n_images x height x width x n_channels)
    """
    paths = ls_rec_path(path)
    if sample_size is None:
        sample_size = len(paths)
    return np.array([np.array(load_image(p)) for p in paths[:sample_size]])


def image_pipeline(images_or_path, n_sample_img, grayscale, normalized, clahe, gamma, rotations, patch_size, stride,
                   overlapping):
    """
    Pre-process input images

    :param images_or_path: images or path to images
    :param n_sample_img: the number of images to load (ordered)
    :param grayscale: whether the images should be loaded as grayscale images (instead of RBG)
    :param normalized: whether the images should be normalized (for pixel this means between going from 0 to 255)
    :param clahe: whether a contrast limited adaptive histogram equalization (CLAHE) should be applied on the images
    :param gamma: whether a gamma correction should be applied on the images
    :param rotations: which rotated version of the images should be added to the dataset (data augmentation)
    :param patch_size: the size of the patches each images is divided into
    :param stride: the stride used for each patches
    :param overlapping: whether patches should be overlapping or not (if not, it's like stride = patch_size)
    :return:
    """
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
    """
    Create a unique file path to a directory given a name, extension and integer

    :param path: directory path
    :param filename: name of the file
    :param ext: extension of the file
    :param n: integer for the file (filename_n.ext)
    :return: a unique file path
    """
    assert path[-1] is '/' or path[-1] is '\\', 'directory path should end with (back-)slash'

    file_path = path + filename + "_" + str(n).zfill(3) + ext
    if os.path.exists(file_path):
        return new_file_path(path, filename, ext, n + 1)
    else:
        return file_path


def load_config(path="default_config.json"):
    """
    Load a JSON dictionary given it's path

    :param path:  where is stored the json file
    :return: the dictionary
    """
    return json.load(open(path))


def save_config(path, config):
    """
    Save a config into a json file

    :param path: where to store a file
    :param config: a dictionary
    :return: None
    """
    assert type(config) is dict
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2, sort_keys=True)
