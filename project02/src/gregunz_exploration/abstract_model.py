from abc import ABC, abstractmethod

import numpy as np
from keras.callbacks import ModelCheckpoint

from helpers import load_image, ls_rec_path, img_to_rgb, img_to_gray


class AModel(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def __init__(self, img_rows=608, img_cols=608, n_channels=3):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.n_channels = n_channels
        super().__init__()

    def load_data(self, data_dir="../../data/", sample_size=np.inf):
        train_dir = data_dir + "training/"
        test_dir = data_dir + "test_set_images/"

        two_d_shape = (self.img_rows, self.img_cols)
        three_d_shape = two_d_shape + (self.n_channels,)

        print("loading data...")
        print('>>> all images will be resized to shape: ' + str(three_d_shape))

        def to_channel(img, is_y=False):
            return img_to_gray(img) if self.n_channels == 1 or is_y else img_to_rgb(img)

        def to_img(path, is_y=False):
            return to_channel(np.array(load_image(path).resize(two_d_shape)), is_y)

        def path_to_data(path, is_y=False):
            paths = ls_rec_path(path)
            n_samples = np.min([sample_size, len(paths)])
            return np.array([to_img(p, is_y) for p in paths[:n_samples]])

        X_tr = path_to_data(train_dir + "images").astype(np.uint8)
        Y = (path_to_data(train_dir + "groundtruth", True) > 127).astype(np.uint8)
        X_te = path_to_data(test_dir).astype(np.uint8)
        print("data loaded")
        return X_tr, Y, X_te

    def train(self, data_dir="../data", deepness=4, kernel_size=5, pool_size=(2, 2), sample_size=np.inf, batch_size=4, epochs=10,
              verbose=1, validation_split=0.2, shuffle=True, callbacks=None):

        X_tr, Y, X_te = self.load_data(data_dir=data_dir, sample_size=sample_size)

        model = self.get_model(deepness=deepness, kernel_size=kernel_size, pool_size=pool_size)

        print('creating model checkpoint')
        model_checkpoint = ModelCheckpoint('u_net.hdf5', monitor='loss', verbose=1, save_best_only=True)
        if callbacks is None:
            callbacks = [model_checkpoint]
        else:
            callbacks.append(model_checkpoint)

        print('fitting model...')
        model.fit(X_tr,
                  Y,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose,
                  validation_split=validation_split,
                  shuffle=shuffle,
                  callbacks=callbacks)
        print('model fit done')

        print('predicting test data...')
        pred = model.predict(X_te, batch_size=batch_size, verbose=verbose)
        print('predict done')

        path = '../results/predictions.npy'
        np.save(path, pred)
        print('predictions saved in {}'.format(path))

        return model, pred
