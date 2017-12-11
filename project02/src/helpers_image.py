import imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(infilename):
    return Image.open(infilename)


def img_to_rgb(img):
    if len(img.shape) < 3 or img.shape[2] == 1:
        return np.repeat(img, 3).reshape(img.shape[0], img.shape[1], 3)
    else:
        return img


def img_to_gray(img):
    if len(img.shape) < 3:
        return img.reshape(img.shape + (1,))
    else:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        return (0.2989 * r + 0.5870 * g + 0.1140 * b).reshape((img.shape[0], img.shape[1], 1))


def show(images, concat=True):
    if concat:
        images = np.concatenate([img_to_rgb(img) for img in images], axis=1)
        show([images], concat=False)
    else:
        for img in images:
            plt.figure(figsize=(15, 7))
            plt.imshow((img * 255).astype(np.uint8))
            plt.show()


def img_to_patches(img, patch_size, stride):
    h, w, _ = img.shape
    assert h == w, 'height should be equal to width ({} != {})'.format(h, w)
    assert (h - patch_size) % stride == 0, 'height - patch_size should be dividable by stride ({} % {} != 0)'.format(
        h - patch_size, stride)

    nb_stride = (h - patch_size) // stride
    patches = []
    for i in range(nb_stride):
        for j in range(nb_stride):
            patch = img[i * stride: i * stride + patch_size, j * stride: j * stride + patch_size]
            patches.append(patch)
    return np.array(patches)


# Rotate an image by a certain degree and extract all possible squares with a certain patch size
def rotate_and_crop(images, angle, patch_size=None, four=False):
    """
    Parameters
    ----------
    images : list of str
        Path to the images that will be rotated.

    angle : int
        the image will be further rotated by this angle

    patch_size : int
        Size of every subsquares of the specified image. Biggest image will be taken if not specified.

    four : boolean
        Whether all four rotations (angle, angle + 90, angle + 180, angle + 270) should be done


    Returns : An array of images of size patch or max size from every rotation
    """
    if angle is 90:
        return np.array([np.rot90(img, i) for img in images for i in range(4)])
    else:
        assert patch_size is not None, 'Need to give a patch_size for other degree than 90'

        if four:
            angles = [angle + i * 90 for i in range(4)]
        else:
            angles = [angle]

        cropped_imgs = []
        for img in images:
            for j in angles:
                img = imutils.rotate_bound(img, j)
                cropped_imgs.append(extract_subsquares(img, patch_size))
        return np.array(cropped_imgs)


def extract_subsquares(image, patch_size):
    h, w = image.shape
    i_jump, j_jump = False, False

    cropped_imgs = []
    if patch_size:
        i, j = 0, 0

        while i < h - patch_size:
            while j < w - patch_size:

                condition = (np.count_nonzero(image[i, j, :]) != 0) and \
                            (np.count_nonzero(image[i, j + patch_size, :]) != 0) and \
                            (np.count_nonzero(image[i + patch_size, j, :]) != 0) and \
                            (np.count_nonzero(image[i, j + patch_size, :]) != 0)

                if condition:
                    cropped = image[i:i + patch_size, j:j + patch_size, :]
                    cropped_imgs.append(cropped)

                if j_jump:
                    j += patch_size
                    j_jump = False
                else:
                    j += 1

            if i_jump:
                i += patch_size
                i_jump = False
            else:
                i += 1
            j = 0
    else:
        side = int(np.floor(h / 4))
        cropped = image[side:3 * side, side:3 * side, :]
        cropped_imgs.append(cropped)

    return cropped_imgs
