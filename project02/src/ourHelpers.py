import os
import numpy as np
import tensorflow as tf
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

class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):
    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.input_fn = input_fn

    def after_save(self, session, global_step):
        print("after_save")
        loss = self.estimator.evaluate(self.input_fn)
        print("loss on test set" + str(loss))



def combinePrediction(img, patch_size, stride, predict_fn):
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
    pred_final = np.zeros(img.shape) #Accumulator for the final prediction
    pred_normalizer = np.zeros(img.shape) #Counter of the patch per prediction per pixel
    assert h == w, "only squared image are accepted"
    assert ((h - patch_size) % stride == 0), "The stride must be adaped on image and patch size"
    nb_stride = (h - patch_size) / stride
    for i in range(nb_stride):
        for j in range(nb_stride): 
            pred_final[i*stride: i*stride + patch_size,
                     j*stride: j*stride + patch_size] += predict_fn(img[i*stride: i*stride + patch_size,
                     j*stride: j*stride + patch_size])
            pred_normalizer[i*stride: i*stride + patch_size,
                     j*stride: j*stride + patch_size] += 1
    return pred_final / pred_normalizer


