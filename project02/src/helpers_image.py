import cv2
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
    if len(img.shape) < 3 or img.shape[2] == 1:
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


def apply_clahe(img):
    if len(img.shape) > 2 and img.shape[2] == 1:
        img = img.reshape(img.shape[0], img.shape[1])
    assert len(img.shape) == 2, 'only on one channel'

    clahe = cv2.createCLAHE()
    return img_to_gray(clahe.apply(np.array(img, dtype=np.uint8)))


def apply_gamma_correction(img, gamma=1.2):
    if len(img.shape) > 2 and img.shape[2] == 1:
        img = img.reshape(img.shape[0], img.shape[1])
    assert len(img.shape) == 2, 'only on one channel'

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return img_to_gray(cv2.LUT(np.array(img, dtype=np.uint8), table))


def img_to_patches(img, patch_size, stride):
    h, w, _ = img.shape
    assert h == w, 'height should be equal to width ({} != {})'.format(h, w)
    assert (h - patch_size) % stride == 0, 'height - patch_size should be dividable by stride ({} % {} != 0)'.format(
        h - patch_size, stride)

    n_stride = (h - patch_size) // stride + 1
    patches = []
    for i in range(n_stride):
        for j in range(n_stride):
            patch = img[i * stride: i * stride + patch_size, j * stride: j * stride + patch_size]
            patches.append(patch)
    return np.array(patches)


def patches_to_img(patches, stride, img_shape):
    if len(img_shape) > 2:
        channels = [patches_to_img(patches[:, :, :, i], stride, img_shape[:2]) for i in range(3)]
        return np.concatenate(channels, axis=2)

    h, w = img_shape
    patch_size = patches.shape[1]
    n_stride = (h - patch_size) // stride + 1

    assert h == w, "only squared image are accepted"
    assert (h - patch_size) % stride == 0, "The stride must be adapted on image and patch size"
    assert len(patches) % n_stride ** 2 == 0, "They must be the right number of patches per image"

    pred_final = np.zeros(img_shape + (1,))  # Accumulator for the final prediction
    pred_normalizer = np.zeros(img_shape + (1,))  # Counter of the patch per prediction per pixel

    for i in range(n_stride):
        for j in range(n_stride):
            x_from, x_to = i * stride, i * stride + patch_size
            y_from, y_to = j * stride, j * stride + patch_size
            idx = i * n_stride + j
            pred_final[x_from: x_to, y_from: y_to] += patches[idx].reshape(patch_size, patch_size, 1)
            pred_normalizer[x_from: x_to, y_from: y_to] += 1
    return pred_final / pred_normalizer


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
