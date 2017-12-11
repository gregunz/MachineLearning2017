import math
import re

import matplotlib.image as mpimg
import numpy as np
from PIL import Image

label_file = 'submission.csv'
h = 16
w = h
imgwidth = int(math.ceil((600.0 / w)) * w)
imgheight = int(math.ceil((600.0 / h)) * h)
nc = 3
foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch


def predictions_to_submission(predictions, path="./"):
    return None


# Convert an array of binary labels to a uint8
def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg


def reconstruct_from_labels(image_id):
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    f = open(label_file)
    lines = f.readlines()
    image_id_str = '%.3d_' % image_id
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j + w, imgwidth)
        ie = min(i + h, imgheight)
        if prediction == 0:
            adata = np.zeros((w, h))
        else:
            adata = np.ones((w, h))

        im[j:je, i:ie] = binary_to_uint8(adata)

    Image.fromarray(im).save('prediction_' + '%.3d' % image_id + '.png')

    return im


# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


def combine_prediction(img, patch_size, stride, predict_fn):
    """ apply per patch prediction to an image and combine the
    prediciton into one final image of same size as img

    WARNING; the size of the image - the patch size must me dividable by the stride
    Params:
        img: numpy Array of shape[n,n]
        patch_size: size of the patch given for prediction to predict_fn
        stride: nb of pixel to be shifted between to patch
        predict_fn: function that given a numpy array of
            shape[patch_size,patch_size] return a prediction of the same shape

    Returns:
        An prediction for img with the same shape
    """
    assert len(img.shape) == 2, "The img must be an image and not a batch of images"
    h = img.shape[0]
    w = img.shape[1]
    pred_final = np.zeros(img.shape)  # Accumulator for the final prediction
    pred_normalizer = np.zeros(img.shape)  # Counter of the patch per prediction per pixel
    assert h == w, "only squared image are accepted"
    assert ((h - patch_size) % stride == 0), "The stride must be adapted on image and patch size"
    nb_stride = (h - patch_size) / stride
    for i in range(nb_stride):
        for j in range(nb_stride):
            pred_final[i * stride: i * stride + patch_size,
            j * stride: j * stride + patch_size] += predict_fn(img[i * stride: i * stride + patch_size,
                                                               j * stride: j * stride + patch_size])
            pred_normalizer[i * stride: i * stride + patch_size,
            j * stride: j * stride + patch_size] += 1
    return pred_final / pred_normalizer
