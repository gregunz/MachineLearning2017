from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.models import Input, Model
from keras.optimizers import Adam

from abstract_pipeline import Pipeline
from helpers_keras import f1


class UNet(Pipeline):
    def create_model(self, from_=64, deepness=4):
        """
        Create the U-net model

        (model architecture from: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

        :param from_: inital number of filters for the convolution
        :param deepness: deepness of the U-net
        :return: None
        """

        def loop(input, deep_idx, deeper, convs):

            assert 0 <= deep_idx <= deepness

            filters = from_ * 2 ** deep_idx

            if deeper:
                conv = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input)
                conv = Dropout(0.2)(conv)
                conv = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv)

                if deep_idx == deepness:
                    self.log('deepest layer - number of filters = {}'.format(filters))
                    return loop(conv, deep_idx - 1, not deeper, convs)

                pool = MaxPooling2D(pool_size=(2, 2))(conv)
                convs.append(conv)
                return loop(pool, deep_idx + 1, deeper, convs)

            else:
                assert deep_idx < deepness
                up = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal') \
                    (input)
                up = concatenate([up, convs[deep_idx]], axis=-1)
                conv = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up)
                conv = Dropout(0.2)(conv)
                conv = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv)

                if deep_idx == 0:
                    return Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal')(conv)

                return loop(conv, deep_idx - 1, deeper, convs)

        self.log("creating model...")
        inputs = Input((self.patch_size, self.patch_size, self.n_channels))
        outputs = loop(inputs, deep_idx=0, deeper=True, convs=list())
        model = Model(inputs=[inputs], outputs=[outputs])

        model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy', f1])

        self.log("model created")
        self.model = model
        return model
