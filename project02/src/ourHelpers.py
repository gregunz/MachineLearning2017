import os
from helpers import *

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
        Y = np.array(Y>0.5).astype(int)
    return X, Y