#!/usr/bin/env python

import numpy as np
from helpers_config import load_config

config = load_config(path='default_config.json')
save_dir = config['dst_path'] + config['model_name'] + "/"

seed = config['seed']
np.random.seed(seed)

import tensorflow as tf
tf.set_random_seed(seed)

from unet import UNet

LOAD_WEIGHTS = True

pipeline = UNet(data_dir=config['data_dir'],
                grayscale=config['grayscale'],
                tr_losses=config['tr_losses'],
                val_losses=config['val_losses'],
                patch_size=config['patch_size'],
                stride=config['stride'],
                telepyth_token=config['telepyth_token'])

X_tr, Y, X_te = pipeline.load_data(overlapping_tr=config['overlapping_tr'],
                                   overlapping_te=config['overlapping_te'],
                                   patch_size=config['patch_size'],
                                   normalized=config['normalized'],
                                   gamma=config['gamma'],
                                   clahe=config['clahe'],
                                   rotations=config['rotations'],
                                   sample_tr_img=config['sample_tr_img'],
                                   sample_te_img=config['sample_te_img'],
                                   load_train=not LOAD_WEIGHTS)

if LOAD_WEIGHTS:
    pipeline.load_model('weights.hdf5')
else:
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
                         sub_epochs=config['sub_epochs'])

predictions = pipeline.predict(X_te, batch_size=200, check_if_trained=not LOAD_WEIGHTS)

pipeline.save_output(predictions=predictions,
                     path=save_dir,
                     overlapping=config['overlapping_te'],
                     rotations=config['rotations'],
                     config=config.copy())

pipeline.log('DONE')
