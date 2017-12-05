# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from keras.models import *
from keras.optimizers import *

from abstract_model import AModel
from helpers import f1, precision, recall


class UNet(AModel):
    def get_model(self, deepness=4, kernel_size=3, pool_size=(2, 2)):

        inputs = Input((self.img_rows, self.img_cols, self.n_channels))

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

        '''
        conv1 = Conv2D(32, kernel_size, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, kernel_size, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size)(conv1)

        conv2 = Conv2D(64, kernel_size, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, kernel_size, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size)(conv2)

        conv3 = Conv2D(128, kernel_size, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, kernel_size, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size)(conv3)

        conv4 = Conv2D(256, kernel_size, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, kernel_size, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size)(conv4)

        conv5 = Conv2D(512, kernel_size, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, kernel_size, activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, kernel_size, activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, kernel_size, activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, kernel_size, activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, kernel_size, activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, kernel_size, activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, kernel_size, activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, kernel_size, activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, kernel_size, activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=[precision, recall, f1, 'accuracy'])

        return model
        '''
        print("loading model...")
        model = loop(deep_ind=0, going_deep=True, convs=[], pools=[inputs])
        print("model loaded")
        return model

    '''
    def get_model(self):
        inputs = Input((self.img_rows, self.img_cols, self.n_channels))

        unet with crop(because padding = valid) 

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
        print("conv1 shape:",conv1.shape)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
        print("conv1 shape:",conv1.shape)
        crop1 = Cropping2D(cropping=((90,90),(90,90)))(conv1)
        print("crop1 shape:",crop1.shape)
        pool1 = MaxPooling2D(pool_size)(conv1)
        print("pool1 shape:",pool1.shape)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
        print("conv2 shape:",conv2.shape)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
        print("conv2 shape:",conv2.shape
        crop2 = Cropping2D(cropping=((41,41),(41,41)))(conv2)
        print("crop2 shape:",crop2.shape)
        pool2 = MaxPooling2D(pool_size)(conv2)
        print("pool2 shape:",pool2.shape)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
        print("conv3 shape:",conv3.shape)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
        print("conv3 shape:",conv3.shape)
        crop3 = Cropping2D(cropping=((16,17),(16,17)))(conv3)
        print("crop3 shape:",crop3.shape)
        pool3 = MaxPooling2D(pool_size)(conv3)
        print("pool3 shape:",pool3.shape)

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        crop4 = Cropping2D(cropping=((4,4),(4,4)))(drop4)
        pool4 = MaxPooling2D(pool_size)(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([crop4,up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([crop3,up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([crop2,up8], mode = 'concat', concat_axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([crop1,up9], mode = 'concat', concat_axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
        
        '#''

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print(">>> conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print(">>> conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size)(conv1)
        print(">>> pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print(">>> conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print(">>> conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size)(conv2)
        print(">>> pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print(">>> conv3 shape:", conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print(">>> conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size)(conv3)
        print(">>> pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size)(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = layers.concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = layers.concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = layers.concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = layers.concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        print("model loaded")

        return model

'''


'''
if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    myunet.save_img()
'''
