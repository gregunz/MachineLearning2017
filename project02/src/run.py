#!/usr/bin/env python

from datetime import datetime

from helpers_config import load_config, save_config
from unet import UNet

config = load_config()
dir_name = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = config['dst_path'] + dir_name + "_" + config['model_name'] + "/"

pipeline = UNet(patch_size=config['patch_size'],
                data_dir=config['data_dir'])

X_tr, Y, X_te = pipeline.load_data(sample_size=config['sample_size'],
                                   rotations=config['rotations'])

pipeline.train_model(X_tr=X_tr,
                     Y=Y,
                     epochs=config['epochs'],
                     sample_size=config['sample_size'],
                     batch_size=config['batch_size'],
                     verbose=config['verbose'],
                     validation_split=config['validation_split'],
                     shuffle=config['shuffle'],
                     load_checkpoint=config['load_checkpoint'],
                     checkpoint_path=save_dir + config['checkpoint_filename'],
                     save_best_only=config['save_best_only'])

predictions = pipeline.predict(X_te=X_te)

pipeline.save_output(predictions=predictions,
                     path=save_dir,
                     checkpoint_path=config['checkpoint_path'])

save_config(save_dir, config)
