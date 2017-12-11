import json
import os
from abc import ABC, abstractmethod

import numpy as np
from keras.callbacks import ModelCheckpoint

from helpers import image_pipeline, path_to_data
from helpers_config import save_config
from helpers_submission import predictions_to_submission


class Pipeline(ABC):
    # This method needs to be defined in the chosen pipeline
    @abstractmethod
    def create_model(self):
        pass

    def __init__(self, patch_size, data_dir='../data/', initial_epoch=0, tr_losses=None, val_losses=None, stride=None):
        self.patch_size = patch_size
        self.data_dir = data_dir
        self.initial_epoch = initial_epoch
        self.stride = stride
        if tr_losses is None:
            tr_losses = []
        self.tr_losses = tr_losses
        if val_losses is None:
            val_losses = []
        self.val_losses = val_losses

        self.train_dir = data_dir + 'training/'
        self.test_dir = data_dir + 'test_set_images/'

        self.tr_h = None
        self.tr_w = None
        self.te_h = None
        self.te_w = None
        self.X_tr = None
        self.Y = None
        self.X_te = None
        self.model = None
        super().__init__()

    def load_data(self, sample_tr_img=None, sample_te_img=None, stride=4, rotations=None, force_reload=False):
        if stride is not None:
            self.stride = stride
        if force_reload or self.X_tr is None or self.Y is None or self.X_te is None:
            print('loading data...')

            self.X_tr, self.tr_h, self.tr_w = image_pipeline(
                path=self.train_dir + 'images',
                sample_img=sample_tr_img,
                rotations=rotations,
                patch_size=self.patch_size,
                stride=self.stride)

            Y, y_h, y_w = image_pipeline(
                path=self.train_dir + 'groundtruth',
                sample_img=sample_tr_img,
                rotations=rotations,
                patch_size=self.patch_size,
                stride=self.stride)
            self.Y = (Y > 0.5).astype(np.uint8)
            assert (y_h, y_w) == (self.tr_h, self.tr_w), 'X_tr and Y images should be of the same size'

            self.X_te, self.te_h, self.te_w = image_pipeline(
                path=self.test_dir,
                sample_img=sample_te_img,
                rotations=rotations,
                patch_size=self.patch_size,
                stride=self.stride)

            print('data loaded')

        if sample_tr_img is None:
            sample_tr_img = len(path_to_data(self.train_dir + 'images'))
        if sample_te_img is None:
            sample_te_img = len(path_to_data(self.test_dir))

        sample_tr_patches = sample_tr_img * ((self.tr_h - self.patch_size) // self.stride) ** 2
        sample_te_patches = sample_te_img * ((self.te_h - self.patch_size) // self.stride) ** 2

        if sample_tr_patches > self.X_tr.shape[0] or sample_te_patches > self.X_te.shape[0]:
            return self.load_data(sample_tr_img=sample_tr_img,
                                  sample_te_img=sample_te_img,
                                  stride=stride,
                                  rotations=rotations,
                                  force_reload=True)

        # some assertions on data size
        for data in [self.X_tr, self.Y, self.X_te]:
            for i in range(1, 3):
                assert data.shape[i] == self.patch_size, \
                    'data should have size equal to patch size ({} != {}) (data.shape = {})'.format(data.shape[i],
                                                                                                    self.patch_size,
                                                                                                    data.shape)

        return self.X_tr[:sample_tr_patches], self.Y[:sample_tr_patches], self.X_te[:sample_te_patches]

    def load_model(self, path):
        print("loading weights from " + path)
        if self.model is None:
            self.model = self.create_model()
        self.model.load_weights(path)
        print("weights loaded")

    def train_model(self, X_tr=None, Y=None, epochs=10, sample_img=None, batch_size=4, verbose=1, validation_split=0.2,
                    shuffle=True, load_checkpoint=False, checkpoint_path='./', save_best_only=True):

        if X_tr is None or Y is None:
            X_tr, Y, _ = self.load_data(sample_tr_img=sample_img)

        if sample_img is None:
            n_patches = len(X_tr)
        else:
            n_patches = ((self.tr_h - self.patch_size) // self.stride) ** 2 * sample_img

        os.makedirs(checkpoint_path, exist_ok=load_checkpoint)

        name = 'weights.hdf5'
        if load_checkpoint and self.model is None:
            self.load_model(checkpoint_path + name)

        if self.model is None:
            self.model = self.create_model()

        model_checkpoint = ModelCheckpoint(filepath=checkpoint_path + name,
                                           monitor='val_loss',
                                           verbose=verbose,
                                           save_best_only=save_best_only)

        print('training model...')
        hist = self.model.fit(X_tr[:n_patches],
                              Y[:n_patches],
                              batch_size=batch_size,
                              epochs=epochs + self.initial_epoch,
                              initial_epoch=self.initial_epoch,
                              verbose=verbose,
                              validation_split=validation_split,
                              shuffle=shuffle,
                              callbacks=[model_checkpoint])
        print('model training done')

        self.initial_epoch += epochs
        self.tr_losses.extend(hist.history['loss'])
        self.val_losses.extend(hist.history['val_loss'])

        return self.model

    def predict(self, X_te=None, sample_img=None, batch_size=1, verbose=1):
        assert self.model is not None and self.initial_epoch > 0, 'model should have been defined and trained'

        if X_te is None:
            _, _, X_te = self.load_data(sample_te_img=sample_img)

        if sample_img is None:
            n_patches = len(X_te)
        else:
            n_patches = ((self.te_h - self.patch_size) // self.stride) ** 2 * sample_img

        return self.model.predict(X_te[:n_patches], verbose=verbose, batch_size=batch_size)

    def save_output(self, predictions, path, config=None):
        assert path[-1] is '/' or path[-1] is '\\', 'directory path should end with (back-)slash'
        os.makedirs(path, exist_ok=True)

        predictions_to_submission(predictions, path)

        with open(path + 'model_summary.json', 'w') as outfile:
            json.dump(json.loads(self.model.to_json()), outfile, indent=2, sort_keys=True)

        if config is not None:
            config['initial_epoch'] = self.initial_epoch
            config['tr_losses'] = self.tr_losses
            config['val_losses'] = self.val_losses
            save_config(path + 'config.json', config)
