#!/usr/bin/env python

from datetime import datetime

from helpers_config import load_config, save_config
from unet import UNet

config = load_config(path="config.json")
# dir_name = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = config['dst_path'] + config['model_name'] + "/"

pipeline = UNet(patch_size=config['patch_size'],
                data_dir=config['data_dir'],
                initial_epoch=config['initial_epoch'],
                tr_losses=config['tr_losses'],
                val_losses=config['val_losses'])

X_tr, Y, X_te = pipeline.load_data(sample_tr_img=config['sample_tr_size'],
                                   rotations=config['rotations'])

pipeline.train_model(X_tr=X_tr,
                     Y=Y,
                     epochs=config['epochs'],
                     sample_img=config['sample_tr_size'],
                     batch_size=config['batch_size'],
                     verbose=config['verbose'],
                     validation_split=config['validation_split'],
                     shuffle=config['shuffle'],
                     load_checkpoint=config['load_checkpoint'],
                     checkpoint_path=save_dir,
                     save_best_only=config['save_best_only'])

predictions = pipeline.predict(X_te=X_te, sample_img=config['sample_te_size'])

pipeline.save_output(predictions=predictions,
                     path=save_dir,
                     config=config.copy())

