import os
import shutil
from abc import ABC, abstractmethod

import numpy as np
import yaml
from keras.callbacks import ModelCheckpoint

from helpers import path_to_data
from helpers_submission import predictions_to_submission


class Pipeline(ABC):
    # This method needs to be defined in the chosen pipeline
    @abstractmethod
    def get_model(self):
        pass

    def __init__(self, patch_size, data_dir='../data/'):
        self.path_size = patch_size
        self.data_dir = data_dir
        self.model = None
        self.initial_epoch = 0
        self.val_loss = None
        self.X_tr = None
        self.Y = None
        self.X_te = None
        super().__init__()

    def load_data(self, sample_size=None, rotations=None):
        if self.X_tr is None or self.Y is None or self.X_te is None:
            train_dir = self.data_dir + 'training/'
            test_dir = self.data_dir + 'test_set_images/'

            print('loading data...')

            X_tr = (path_to_data(train_dir + 'images', sample_size) / 255).astype('float32')
            self.X_tr = X_tr
            Y = (path_to_data(train_dir + 'groundtruth', sample_size) > 127)
            self.Y = Y
            X_te = (path_to_data(test_dir) / 255, sample_size).astype('float32')
            self.X_te = X_te
            print('data loaded')

        return self.X_tr, self.Y, self.X_te

    def load_model(self, path='unet.hdf5'):
        self.model.load_weights(path)

    def train_model(self, X_tr=None, Y=None, epochs=10, sample_size=None, batch_size=4, verbose=1, validation_split=0.2,
                    shuffle=True, load_checkpoint=False, checkpoint_path='model_checkpoint.hdf5', save_best_only=True):

        if X_tr is None or Y is None:
            X_tr, Y, _ = self.load_data(sample_size=sample_size)

        if self.model is None:
            self.model = self.get_model()

        if load_checkpoint:
            self.load_model(checkpoint_path)

        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=verbose,
                                           save_best_only=save_best_only)

        print('training model...')
        hist = self.model.fit(X_tr,
                              Y,
                              batch_size=batch_size,
                              epochs=epochs,
                              initial_epoch=self.initial_epoch,
                              verbose=verbose,
                              validation_split=validation_split,
                              shuffle=shuffle,
                              callbacks=[model_checkpoint])
        print('model training done')

        self.initial_epoch += epochs
        self.val_loss = np.min(hist['val_loss'])

        return self.model

    def predict(self, X_te=None):
        assert self.model is not None and self.initial_epoch > 0, 'model should have been defined and trained'

        if X_te is None:
            _, _, X_te = self.load_data()
        return self.model.predict(X_te)

    def save_output(self, predictions, path, checkpoint_path):
        assert path[-1] is '/' or path[-1] is '\\', 'directory path should end with (back-)slash'
        assert os.path.exists(checkpoint_path), 'checkpoint path should point to the .hdf5 file'

        predictions_to_submission(predictions, path)

        os.makedirs(path)
        shutil.move(checkpoint_path, path)

        with open('model_summary.yml', 'w') as outfile:
            yaml.dump(self.model.to_yaml(), outfile)
