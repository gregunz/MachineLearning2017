import os
import numpy as np
import imutils
import cv2
from helpers import *

from sklearn.metrics import f1_score

def load_training_dataset(data_path= "../",sample_size= None, binary_gt= False):
    """ Load the dataset into numpy array
    Value are sorted on there name such that Y[i] correspond to groundtruth of X[i]
    
    Parms:
        sample_size: size of the sample to return (not randomly choosen)
        binary_gt: When True return the groudtruth as a binary image by applying a threshold at 0.5
    Return:
        train_X, train_Y : Training images and they groundtruth
    """
    root_dir = data_path + "data/training/"
    image_dir = root_dir + "images/"
    label_dir = root_dir + "groundtruth"
    files = os.listdir(image_dir)
    files = [image_dir +"/"+ f for f in files]
    files = sorted(files)
    labels_files = os.listdir(label_dir)
    labels_files = [label_dir +"/"+ f for f in labels_files]
    labels_files = sorted(labels_files)

    X = []
    Y = []
    if sample_size == None:
        sample_size = len(files)
    for img in files[:sample_size]:
        X.append(load_image(img))
    for lab in labels_files[:sample_size]:
        Y.append(load_image(lab))
    X = np.array(X)
    Y = np.array(Y)
    if binary_gt:
        Y = np.array(Y>0.5).astype(np.uint8)
    return X, Y

def sk_mean_F1_score(prediction, groundtruth):
    f1s=[]
    for i in range(prediction.shape[0]):
        y_true = np.reshape(groundtruth[i], [-1])
        y_pred = np.reshape(prediction[i], [-1])
        f1s.append(f1_score(y_true, y_pred, average='macro'))
    return np.array(f1s).mean()


# Rotate an image by a certain degree and extract all possible squares with a certain patch size
def rotate_and_crop(image_path, angle, write_path=None, patch=None):
    """
    Parameters
    ----------
    image_path : str
        Path to the image that will be rotated.

    angle : int
        the image will be further rotatated by this angle for each 90 degree rotation

    write_path : str
        If specified will write the images to this path

    patch : int
        Size of every subsquares of the specified image. Biggest image will be taken if not specified.


    Returns: An array of images of size patch or max size from every rotation
    """


    if (patch is None and (angle != 45)):
        print('Need to give a patch for other degree than 45')
        return None
    angles = [angle + x * 90 for x in range(0, 4)]
    image = load_image(image_path)
    cropped_imgs = []

    if (write_path):
        k = image_path.rfind("/")
        l = image_path.rfind('.')
        write_path = write_path + '/' + image_path[k + 1:l]
        os.makedirs(write_path)

    for j in angles:

        img = imutils.rotate_bound(image, j)

        if (write_path):
            path = write_path + '/image_' + str(j) + '/'
            os.makedirs(path)
            cropped_imgs.append(extract_subsquares(img, path, patch))
        else:
            cropped_imgs.append(extract_subsquares(img, patch))

    return cropped_imgs


def extract_subsquares(image, write_path=None, patch=None):
    h = image.shape[0]
    w = image.shape[1]
    cropped_imgs = []
    i_jump = False
    j_jump = False

    if (patch):
        i = 0
        j = 0

        while (i < h - patch):
            while (j < w - patch):

                condition = (np.count_nonzero(image[i, j, :]) != 0) and \
                            (np.count_nonzero(image[i, j + patch, :]) != 0) and \
                            (np.count_nonzero(image[i + patch, j, :]) != 0) and \
                            (np.count_nonzero(image[i, j + patch, :]) != 0)

                if (condition):

                    cropped = image[i:i + patch, j:j + patch, :]
                    cropped_imgs.append(cropped)

                    if (write_path):
                        cv2.imwrite(write_path + str(i) + str(j) + '.png', np.floor(cropped * 255))
                        j_jump = True
                        i_jump = True

                if (j_jump):
                    j += patch
                    j_jump = False
                else:
                    j += 1

            if (i_jump):
                i += patch
                i_jump = False
            else:
                i += 1
            j = 0
    else:
        side = int(np.floor(h / 4))
        cropped = image[side:3 * side, side:3 * side, :]
        cropped_imgs.append(cropped)
        if (write_path):
            cv2.imwrite(write_path + 'image.png', np.floor(cropped * 255))

    return cropped_imgs