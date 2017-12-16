from keras.layers import concatenate, Activation, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.models import Input, Model
from keras.optimizers import Adam

from abstract_pipeline import Pipeline
from helpers_keras import f1


class UNet(Pipeline):
    def create_model(self):
        if self.model is not None:
            return self.model
        else:
            print("creating model...")
            inputs = Input((self.patch_size, self.patch_size, self.n_channels))

            conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = Dropout(0.2)(conv1)
            conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = Dropout(0.2)(conv2)
            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = Dropout(0.2)(conv3)
            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = Dropout(0.2)(conv4)
            conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = Dropout(0.2)(conv5)
            conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

            up6 = concatenate(
                [Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv5),
                 conv4], axis=3)
            conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
            conv6 = Dropout(0.2)(conv6)
            conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = concatenate(
                [Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6),
                 conv3], axis=3)
            conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
            conv7 = Dropout(0.2)(conv7)
            conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            up8 = concatenate(
                [Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7),
                 conv2], axis=3)
            conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
            conv8 = Dropout(0.2)(conv8)
            conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            up9 = concatenate(
                [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8),
                 conv1], axis=3)
            conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
            conv9 = Dropout(0.2)(conv9)
            conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

            conv10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal')(conv9)

            model = Model(inputs=[inputs], outputs=[conv10])

            model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy', f1])

            print("model created")
            self.model = model
            return model
