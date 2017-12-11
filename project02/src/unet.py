from keras.layers import merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import *
from keras.optimizers import *

from abstract_pipeline import Pipeline


class UNet(Pipeline):
    def create_model(self, deepness=4, kernel_size=3, pool_size=(2, 2)):
        if self.model is not None:
            return self.model
        else:
            print("loading model...")
            inputs = Input((self.path_size, self.path_size, 3))

            '''
            def loop(deep_ind, going_deep, convs, pools):
                if going_deep:
                    if deep_ind < deepness:
                        layers = 2 ** (5 + deep_ind)
                        conv = Conv2D(layers, kernel_size, activation='relu', padding='same')(pools[deep_ind])
                        conv = Conv2D(layers, kernel_size, activation='relu', padding='same')(conv)
                        pool = MaxPooling2D(pool_size)(conv)
                        convs.append(conv)
                        pools.append(pool)
                        return loop(deep_ind=deep_ind + 1, going_deep=going_deep, convs=convs, pools=pools)
                    if deep_ind >= deepness:
                        layers = 2 ** (5 + deep_ind)
                        conv = Conv2D(layers, kernel_size, activation='relu', padding='same')(pools[deep_ind])
                        conv = Conv2D(layers, kernel_size, activation='relu', padding='same')(conv)
                        convs.append(conv)
                        return loop(deep_ind=deep_ind - 1, going_deep=not going_deep, convs=convs, pools=pools)
                if not going_deep:
                    if deep_ind >= 0:
                        layers = 2 ** (5 + deep_ind)
                        up = concatenate([Conv2DTranspose(layers, (2, 2), strides=(2, 2), padding='same')(
                            convs[2 * deepness - deep_ind - 1]), convs[deep_ind]], axis=3)
                        conv = Conv2D(layers, kernel_size, activation='relu', padding='same')(up)
                        conv = Conv2D(layers, kernel_size, activation='relu', padding='same')(conv)
                        convs.append(conv)
                        return loop(deep_ind=deep_ind - 1, going_deep=going_deep, convs=convs, pools=pools)
                    if deep_ind < 0:
                        conv = Conv2D(1, 1, activation='sigmoid')(convs[-1])
                        model = Model(inputs=[inputs], outputs=[conv])
                        model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy',
                                      metrics=[precision, recall, f1, 'accuracy'])
                        return model
                        
            model = loop(deep_ind=0, going_deep=True, convs=[], pools=[inputs])
                        
            '''

            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

            conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)

            up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(drop5))
            merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
            conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
            conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv6))
            merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
            conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
            conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv7))
            merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
            conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv8))
            merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
            conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
            conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

            model = Model(input=inputs, output=conv10)

            model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

            print("model loaded")
            self.model = model
            return model
