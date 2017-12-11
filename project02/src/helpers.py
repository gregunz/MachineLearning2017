# Helper functions

import os
import numpy as np
from helpers_image import load_image


def ls_rec_path(path):
    return sorted(["{}/{}".format(root, f) for root, _, files in os.walk(path) for f in files])


def path_to_data(path, sample_size=None):
    paths = ls_rec_path(path)
    if sample_size is None:
        sample_size = len(paths)
    return np.array([np.array(load_image(p)) for p in paths[:sample_size]])
