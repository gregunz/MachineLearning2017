import numpy as np


def combine_prediction(img, patch_size, stride, predict_fn):
    """ apply per patch prediction to an image and combine the
    prediction into one final image of same size as img

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
    assert img.shape[0] == img.shape[1], "only squared image are accepted"
    assert ((img.shape[0] - patch_size) % stride == 0), "The stride must be adapted on image and patch size"

    h, w = img.shape
    pred_final = np.zeros_like(img)  # Accumulator for the final prediction
    pred_normalizer = np.zeros_like(img)  # Counter of the patch per prediction per pixel
    nb_stride = (h - patch_size) / stride
    for i in range(nb_stride):
        for j in range(nb_stride):
            pred_final[i * stride: i * stride + patch_size,
            j * stride: j * stride + patch_size] += predict_fn(img[i * stride: i * stride + patch_size,
                                                               j * stride: j * stride + patch_size])
            pred_normalizer[i * stride: i * stride + patch_size,
            j * stride: j * stride + patch_size] += 1
    return pred_final / pred_normalizer
