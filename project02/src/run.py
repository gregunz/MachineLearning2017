#!/usr/bin/env python

import numpy as np
import tensorflow as tf

from helpers_config import load_config
from unet import UNet

config = load_config(path="config.json")

seed = config['seed']
np.random.seed(seed)
tf.set_random_seed(seed)

save_dir = config['dst_path'] + config['model_name'] + "/"

pipeline = UNet(data_dir=config['data_dir'],
                grayscale=config['grayscale'],
                tr_losses=config['tr_losses'],
                val_losses=config['val_losses'],
                patch_size=config['patch_size'],
                stride=config['stride'],
                telepyth_token=config['telepyth_token'])

X_tr, Y, X_te = pipeline.load_data(overlapping_tr=config['overlapping_tr'],
                                   overlapping_te=config['overlapping_te'],
                                   normalized=config['normalized'],
                                   gamma=config['gamma'],
                                   clahe=config['clahe'],
                                   rotations=config['rotations'],
                                   force_reload=True,
                                   sample_tr_img=config['sample_tr_img'],
                                   sample_te_img=config['sample_te_img'])

validation_data = None
if config['overlapping_tr']:
    split = config['validation_split']
    config['validation_split'] = None
    patches_per_img = (pipeline.n_patches_tr * (1 + len(config['rotations'])))
    assert len(X_tr) % patches_per_img == 0
    n_images = len(X_tr) // patches_per_img
    indices = np.arange(n_images)
    if config['shuffle']:
        indices = np.random.permutation(n_images)
    n_val_images = int(split * n_images)
    X_val = np.concatenate([X_tr[i * patches_per_img:(i + 1) * patches_per_img] for i in indices[:n_val_images]])
    X_tr = np.concatenate([X_tr[i * patches_per_img:(i + 1) * patches_per_img] for i in indices[n_val_images:]])
    Y_val = np.concatenate([Y[i * patches_per_img:(i + 1) * patches_per_img] for i in indices[:n_val_images]])
    Y = np.concatenate([Y[i * patches_per_img:(i + 1) * patches_per_img] for i in indices[n_val_images:]])
    validation_data = (X_val, Y_val)

pipeline.train_model(X_tr=X_tr,
                     Y=Y,
                     epochs=config['epochs'],
                     batch_size=config['batch_size'],
                     verbose=config['verbose'],
                     validation_split=config['validation_split'],
                     shuffle=config['shuffle'],
                     load_checkpoint=config['load_checkpoint'],
                     checkpoint_path=save_dir,
                     save_best_only=config['save_best_only'],
                     sub_epochs=config['sub_epochs'],
                     validation_data=validation_data)

predictions = pipeline.predict(X_te=X_te,
                               batch_size=config['batch_size'],
                               verbose=config['verbose'])

pipeline.save_output(predictions=predictions,
                     path=save_dir,
                     overlapping=config['overlapping_te'],
                     rotations=config['rotations'],
                     config=config.copy())
