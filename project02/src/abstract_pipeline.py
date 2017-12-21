import json
import os
from abc import ABC, abstractmethod

import numpy as np
from keras.callbacks import ModelCheckpoint, LambdaCallback
from telepyth import TelepythClient

from helpers import image_pipeline, path_to_data, new_file_path, save_config
from helpers_image import patches_to_images, revert_rotations
from helpers_submission import masks_to_submission


class Pipeline(ABC):
    """
    This class represents an abstract pipeline it easier to create multiple model and still run the same code,
    thanks to object oriented programming.

    This abstract pipeline implements loading data, loading model, training model, computing predictions,
    creating submission.

    It also features a log function which can log results on your Telegram app using telepyth.
    """

    # This method needs to be defined in the chosen pipeline
    @abstractmethod
    def create_model(self):
        pass

    def __init__(self, data_dir='../data/', grayscale=False, tr_losses=list(), val_losses=list(), patch_size=80,
                 stride=16, telepyth_token=None):
        """
        Initizalize the pipeline

        :param data_dir: where the data (training and testing images) is stored
        :param grayscale: whether the images should be loaded as grayscale images (instead of RBG)
        :param tr_losses: the initial training losses (useful when reloading a previously trained model)
        :param val_losses: the initial validation losses (useful when reloading a previously trained (CV) model)
        :param patch_size: the size of the patches each images is divided into
        :param stride: the stride used for each patches
        :param telepyth_token: the token to be used if we want logs to be sent on your Telegram account
        """

        assert len(tr_losses) == len(val_losses)

        self.n_channels = 1 if grayscale else 3
        self.patch_size = patch_size
        self.stride = stride
        self.data_dir = data_dir
        self.initial_epoch = 0 if tr_losses is None else len(tr_losses)
        self.tr_losses = tr_losses
        self.val_losses = val_losses

        self.train_dir = data_dir + 'training/'
        self.test_dir = data_dir + 'test_set_images/'

        self.tp = None
        if telepyth_token is not None:
            self.tp = TelepythClient(telepyth_token)

        self.tr_h = None
        self.tr_w = None
        self.te_h = None
        self.te_w = None
        self.X_tr = None
        self.n_patches_tr = None,
        self.n_patches_te = None,
        self.Y = None
        self.X_te = None
        self.model = None
        super().__init__()

    def log(self, obj, always_print=True):
        """
        A logging function

        :param obj: the object to log (will be serialized if need)
        :param always_print: whether or not we should print in stdout as well
        :return: None
        """
        try:
            if type(obj) is not str:
                obj = json.dumps(obj, indent=2, sort_keys=True)
            else:
                obj = obj.replace('_', '-')

            if self.tp is not None:
                if always_print:
                    print(obj)
                self.tp.send_text(obj)
            else:
                print(obj)
        except (TypeError, ValueError):
            self.log('An error happen during logging, python object is printed in stdout instead...', always_print=True)
            print(obj)

    def log_plot(self, fig, text=''):
        """
        A logging function for telepyth and plots

        :param fig: a matplotlib plot to log
        :param text: a legend for the plot
        :return: None
        """
        if self.tp is not None:
            self.tp.send_figure(fig, text)

    def load_data(self, patch_size=None, stride=None, overlapping_tr=False, overlapping_te=True, normalized=False,
                  clahe=False, gamma=False, rotations=list(), force_reload=False, sample_tr_img=None,
                  sample_te_img=None, load_train=True, load_test=True):
        """
        Load the data inside the pipeline

        :param patch_size: the size of the patches each images is divided into (overrides the one in init)
        :param stride: the stride used for each patches (overrides the one in init)
        :param overlapping_tr: whether training patches should be overlapping or not (if not, it's like stride = patch_size)
        :param overlapping_te: whether testing patches should be overlapping or not (if not, it's like stride = patch_size)
        :param normalized: whether the images should be normalized (for pixel this means between going from 0 to 255)
        :param clahe: whether a contrast limited adaptive histogram equalization (CLAHE) should be applied on the images
        :param gamma: whether a gamma correction should be applied on the images
        :param rotations: which rotated version of the images should be added to the dataset (data augmentation)
        :param force_reload: whether we should reload data from disk
        :param sample_tr_img: number of training images to be loaded from disk in this pipeline
        :param sample_te_img: number of testing images to be loaded from disk in this pipeline
        :param load_train: whether the training data should be loaded from disk
        :param load_test: wheter the testing data should be loaded from disk
        :return: X_tr (training data), Y (training groundtruth), X_te (testing data)
        """
        if stride is not None:
            self.stride = stride
        if patch_size is not None:
            self.patch_size = patch_size

        if force_reload or self.X_tr is None or self.Y is None or self.X_te is None:
            self.log('loading data...')

            if load_train:
                self.X_tr, self.tr_h, self.tr_w = image_pipeline(
                    images_or_path=self.train_dir + 'images',
                    n_sample_img=sample_tr_img,
                    rotations=rotations,
                    patch_size=self.patch_size,
                    stride=self.stride,
                    grayscale=self.n_channels == 1,
                    normalized=normalized,
                    clahe=clahe,
                    gamma=gamma,
                    overlapping=overlapping_tr,
                )
                assert overlapping_tr or self.tr_w % self.patch_size == 0

                self.Y, y_h, y_w = image_pipeline(
                    images_or_path=self.train_dir + 'groundtruth',
                    n_sample_img=sample_tr_img,
                    rotations=rotations,
                    patch_size=self.patch_size,
                    stride=self.stride,
                    grayscale=False,
                    normalized=False,
                    clahe=False,
                    gamma=False,
                    overlapping=overlapping_tr
                )
                self.Y = (self.Y > 0.5).astype(np.float32)
                assert (y_h, y_w) == (self.tr_h, self.tr_w), 'X_tr and Y images should be of the same size'
                self.log('train data loaded')

                if sample_tr_img is None:
                    sample_tr_img = len(path_to_data(self.train_dir + 'images'))

                self.n_patches_tr = ((self.tr_h - self.patch_size) // self.stride + 1) ** 2 if overlapping_tr else \
                    (self.tr_h // self.patch_size) ** 2
                sample_tr_patches = sample_tr_img * self.n_patches_tr * (len(rotations) + 1)

                self.X_tr = self.X_tr[:sample_tr_patches]
                self.Y = self.Y[:sample_tr_patches]

                if sample_tr_patches > self.X_tr.shape[0]:
                    return self.load_data(overlapping_tr=overlapping_tr,
                                          overlapping_te=overlapping_te,
                                          normalized=normalized,
                                          gamma=gamma,
                                          clahe=clahe,
                                          sample_tr_img=sample_tr_img,
                                          sample_te_img=sample_te_img,
                                          rotations=rotations,
                                          force_reload=True,
                                          load_train=load_train,
                                          load_test=load_test)

                # some assertions on data size
                for data in [self.X_tr, self.Y]:
                    for i in range(1, 3):
                        assert data.shape[i] == self.patch_size, \
                            'data should have size equal to patch size ({} != {}) (data.shape = {})'.format(
                                data.shape[i],
                                self.patch_size,
                                data.shape)

            if load_test:
                self.X_te, self.te_h, self.te_w = image_pipeline(
                    images_or_path=self.test_dir,
                    n_sample_img=sample_te_img,
                    rotations=rotations,
                    patch_size=self.patch_size,
                    stride=self.stride,
                    grayscale=self.n_channels == 1,
                    normalized=normalized,
                    clahe=clahe,
                    gamma=gamma,
                    overlapping=overlapping_te,
                )
                assert overlapping_te or self.te_w % self.patch_size == 0
                self.log('test data loaded')

                if sample_te_img is None:
                    sample_te_img = len(path_to_data(self.test_dir))

                self.n_patches_te = ((self.te_h - self.patch_size) // self.stride + 1) ** 2 if overlapping_te else \
                    (self.te_h // self.patch_size) ** 2

                sample_te_patches = sample_te_img * self.n_patches_te * (len(rotations) + 1)

                for i in range(1, 3):
                    assert self.X_te.shape[i] == self.patch_size, \
                        'data should have size equal to patch size ({} != {}) (data.shape = {})' \
                            .format(self.X_te.shape[i], self.patch_size, self.X_te.shape)

                if load_test and sample_te_patches > self.X_te.shape[0]:
                    return self.load_data(overlapping_tr=overlapping_tr,
                                          overlapping_te=overlapping_te,
                                          normalized=normalized,
                                          gamma=gamma,
                                          clahe=clahe,
                                          sample_tr_img=sample_tr_img,
                                          sample_te_img=sample_te_img,
                                          rotations=rotations,
                                          force_reload=True,
                                          load_train=load_train,
                                          load_test=load_test)

        return self.X_tr, self.Y, self.X_te

    def load_model(self, path):
        """
        Load the weights in the pipeline given a path to its weights

        :param path: path to the stored weights
        :return: None
        """
        self.log('loading weights from ' + path)
        if self.model is None:
            self.log('model was not created, creating the default model...')
            self.create_model()
        self.model.load_weights(path)
        self.log('weights loaded')

    def train_model(self, X_tr=None, Y=None, epochs=5, batch_size=4, verbose=1, validation_split=0.2,
                    shuffle=True, load_checkpoint=False, checkpoint_path='./', save_best_only=True, sub_epochs=1,
                    validation_data=None):
        """
        Train the model described in this pipeline

        :param X_tr: training data (numpy ndarray of dimension n_images x height x width x n_channels)
        :param Y: groundtruth data (numpy ndarray of dimension n_images x height x width x n_channels)
        :param epochs: number of epoch to compute
        :param batch_size: batch size of data used for the stochastic descent computations
        :param verbose: 0 if we should not print computations details, 1 if we want
        :param validation_split: by how much to split the data in order to perform cross validation
        :param shuffle: whether the data should be shuffled
        :param load_checkpoint: whether we should load checkpoint (weights) before training
        :param checkpoint_path: path to store checkpoints (weights) during training
        :param save_best_only: whether only the best weights should be saved (regarding loss if no splitting,
                               and val_loss otherwise)
        :param sub_epochs: number of which the epoch is divided by in order to have a higher granularity
        :param validation_data: if validation split is set to None, this can be used to give validation data directly
        :return: None
        """

        assert validation_split is None or 0 <= validation_split < 1
        assert validation_data is not None or validation_split is not None, 'only one validation technique at a time'

        if X_tr is None or Y is None:
            self.log('data was not loaded in the pipeline...')
            X_tr, Y, _ = self.load_data()

        if sub_epochs > 1:
            self.log('subdividing epochs into {} smaller epochs'.format(epochs * sub_epochs))
            indices = np.arange(X_tr.shape[0])
            if shuffle:
                indices = np.random.permutation(X_tr.shape[0])
            if validation_data is None:
                until = int(indices.shape[0] * validation_split)
                validation_data = (X_tr[indices[:until]], Y[indices[:until]])
                indices = indices[until:]
                validation_split = None

            for _ in range(epochs):
                for i in range(sub_epochs):
                    ind_part = np.split(indices, sub_epochs)[i]
                    self.train_model(X_tr=X_tr[ind_part], Y=Y[ind_part], epochs=1,
                                     batch_size=batch_size, verbose=verbose, validation_split=validation_split,
                                     shuffle=shuffle, load_checkpoint=load_checkpoint, checkpoint_path=checkpoint_path,
                                     save_best_only=save_best_only, sub_epochs=1, validation_data=validation_data)
            return self.model

        os.makedirs(checkpoint_path, exist_ok=True)

        if load_checkpoint and self.model is None and self.initial_epoch > 0:
            self.load_model(checkpoint_path + 'weights_' + str(self.initial_epoch).zfill(3) + '.hdf5')

        if self.model is None:
            self.create_model()

        monitor = 'val_loss' if validation_data is not None or validation_split > 0 else 'loss'

        if save_best_only:
            file_path = new_file_path(path=checkpoint_path,
                                      filename='weights',
                                      ext='.hdf5',
                                      n=epochs + self.initial_epoch)
        else:
            file_path = checkpoint_path + 'weights.{epoch:03d}-{' + monitor + ':.5f}.hdf5'

        model_checkpoint = ModelCheckpoint(filepath=file_path,
                                           monitor=monitor,
                                           verbose=verbose,
                                           save_best_only=save_best_only)

        log_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: self.log((epoch + 1, logs)))

        self.log('training model...')
        hist = self.model.fit(X_tr,
                              Y,
                              batch_size=batch_size,
                              epochs=epochs + self.initial_epoch,
                              initial_epoch=self.initial_epoch,
                              verbose=verbose,
                              validation_split=validation_split,
                              validation_data=validation_data,
                              shuffle=shuffle,
                              callbacks=[model_checkpoint, log_callback])
        self.log('model training done')

        self.initial_epoch += epochs
        self.tr_losses.extend(hist.history['loss'])
        if validation_split is not None and validation_split > 0:
            self.val_losses.extend(hist.history['val_loss'])

    def predict(self, X_te=None, batch_size=1, verbose=1, check_if_trained=True):
        """
        Compute predictions (a pixel wise probability of a road) for each testing data

        :param X_te: testing data (np.ndarray of dimension n_images x height x width x n_channels)
        :param batch_size: batch size used to compute the prections (if it fits in memory)
        :param verbose: 0 if we should not print computations details, 1 if we want
        :param check_if_trained: whether an assertions checks if the model has been trained already
        :return: the predictions
        """
        assert self.model is not None and (
                not check_if_trained or self.initial_epoch > 0), 'model should have been defined and trained'

        if X_te is None:
            _, _, X_te = self.load_data()

        return self.model.predict(X_te, verbose=verbose, batch_size=batch_size)

    def create_submission(self, masks, path, overlapping, rotations):
        """
        Create a submission given masks

        :param masks: masks where each pixel represents a probability of being foreground (road)
        :param path: path to store the submission
        :param overlapping: whether overlapping data was used (for the data augmentation)
        :param rotations: which rotations where used (for the data augmentation)
        :return: None
        """
        assert path[-1] is '/' or path[-1] is '\\', 'directory path should end with (back-)slash'

        file_path = new_file_path(path, 'submission', '.csv', 0)

        if overlapping:
            masks = patches_to_images(masks, self.stride, (self.te_h, self.te_w))

        if len(rotations) > 0:
            rotations = [0] + rotations.copy()
            masks = revert_rotations(masks, rotations)

        masks_to_submission(file_path, masks)

    def save_output(self, predictions, path, overlapping, rotations, config):
        """
        Save a submission given some predictions, save the model description (json), save the config used in the pipeline.

        :param predictions: predictions where each pixel represents a probability of being foreground (road)
        :param path: path to store the submission, model description and config
        :param overlapping: whether overlapping data was used (for the data augmentation)
        :param rotations: which rotations where used (for the data augmentation)
        :param config: a dictionary containing pipeline configuration details to save
        :return: None
        """
        assert path[-1] is '/' or path[-1] is '\\', 'directory path should end with (back-)slash'
        os.makedirs(path, exist_ok=True)

        with open(path + 'model_summary.json', 'w') as outfile:
            json.dump(json.loads(self.model.to_json()), outfile, indent=2, sort_keys=True)

        if config is not None:
            config['initial_epoch'] = self.initial_epoch
            config['tr_losses'] = self.tr_losses
            config['val_losses'] = self.val_losses
            save_config(path + 'config.json', config)

        self.create_submission(predictions, path, overlapping, rotations)
