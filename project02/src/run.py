#!/usr/bin/env python

from helpers_config import load_config
from unet import UNet

config = load_config(path="config.json")
# dir_name = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = config['dst_path'] + config['model_name'] + "/"

pipeline = UNet(data_dir=config['data_dir'],
                grayscale=config['grayscale'],
                initial_epoch=config['initial_epoch'],
                tr_losses=config['tr_losses'],
                val_losses=config['val_losses'])

X_tr, Y, X_te = pipeline.load_data(patch_size=config['patch_size'],
                                   stride=config['stride'],
                                   normalized=config['normalized'],
                                   gamma=config['gamma'],
                                   clahe=config['clahe'],
                                   rotations=config['rotations'],
                                   sample_tr_img=config['sample_tr_img'],
                                   sample_te_img=config['sample_te_img'])

pipeline.train_model(X_tr=X_tr,
                     Y=Y,
                     epochs=config['epochs'],
                     sample_img=config['sample_tr_img'],
                     batch_size=config['batch_size'],
                     verbose=config['verbose'],
                     validation_split=config['validation_split'],
                     shuffle=config['shuffle'],
                     load_checkpoint=config['load_checkpoint'],
                     checkpoint_path=save_dir,
                     save_best_only=config['save_best_only'])

predictions = pipeline.predict(X_te=X_te, sample_img=config['sample_te_img'])

pipeline.save_output(predictions=predictions,
                     path=save_dir,
                     config=config.copy())
