import glob

import cv2
import matplotlib.image as mpimg
import tensorflow as tf
from sklearn.metrics import f1_score

from helpers import *


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def load_training_dataset(data_path="../", sample_size=None, binary_gt=False):
    """ Load the dataset into numpy array
    Value are sorted on there name such that Y[i] correspond to groundtruth of X[i]
    
    Parms:
        sample_size: size of the sample to return (not randomly choosen)
        binary_gt: When True return the groudtruth as a binary image by applying a threshold at 0.5
    Return:
        train_X, train_Y : Training images and they groundtruth into a numpy array
    """
    root_dir = data_path + "data/training/"
    image_dir = root_dir + "images/"
    label_dir = root_dir + "groundtruth"
    files = os.listdir(image_dir)
    files = [image_dir + "/" + f for f in files]
    files = sorted(files)
    labels_files = os.listdir(label_dir)
    labels_files = [label_dir + "/" + f for f in labels_files]
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
        Y = np.array(Y > 0.5).astype(np.uint8)
    return X, Y


def load_test_dataset(data_path="../", sample_size=None):
    """ Load the test dataset

    Return:
        Test set images into a numpy array
    """
    root_dir = data_path + "data/test_set_images/"
    test_paths = glob.glob(root_dir + "*/*.png")
    # sort on the folder name (number)
    test_paths = sorted(test_paths, key=lambda path: int(path.split("/")[3].split("_")[1]))
    # print(test_paths)
    X = []
    if sample_size == None:
        sample_size = len(test_paths)
    for img in test_paths[:sample_size]:
        X.append(load_image(img))
    return np.array(X)


# TODO: this methode can be beter code (cleaned)
def save_test_prediction(data_path="../", overlay=False, predictions=None):
    """ Save the prediction on disk (next to the test image)
    Params:
        data_path: root of the data folder
        overlay: Boolean set to true when prediction array is the
            image with an overlay prediction 
        predictions: array of prediction image or overlay Images

    """
    root_dir = data_path + "data/test_set_images/"
    img_name = "overlay" if overlay else "prediction"
    for i in range(len(predictions)):
        img = predictions[i] if overlay else predictions[i] * 255
        if overlay:
            img.save(root_dir + "test_" + str(i + 1) + "/" + img_name + ".png")
        else:
            cv2.imwrite(root_dir + "test_" + str(i + 1) + "/" + img_name + ".png", img)


def sk_mean_F1_score(prediction, groundtruth):
    f1s = []
    for i in range(prediction.shape[0]):
        y_true = np.reshape(groundtruth[i], [-1])
        y_pred = np.reshape(prediction[i], [-1])
        f1s.append(f1_score(y_true, y_pred, average='macro'))
    return np.array(f1s).mean()


class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):
    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.input_fn = input_fn

    def after_save(self, session, global_step):
        print("after_save")
        loss = self.estimator.evaluate(self.input_fn)
        print("loss on test set" + str(loss))
