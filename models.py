from keras import models, layers
from keras.utils import multi_gpu_model
from layers import Length, Mask, FashionCaps
from utils import squash, manhattan_dist


class MultiGPUNet(models.Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(MultiGPUNet, self).__getattribute__(attrname)


def FashionSiameseCapsNet(input_shape):
    x1 = layers.Input(shape=input_shape)
    x2 = layers.Input(shape=input_shape)

    siamese_caps_net = models.Sequential()
    siamese_caps_net.add(layers.Conv2D(64, 9, strides=1, padding="valid", name="conv_block_1"))
    siamese_caps_net.add(layers.BatchNormalization())
    siamese_caps_net.add(layers.LeakyReLU())
    siamese_caps_net.add(layers.Conv2D(128, 7, strides=2, padding="valid", name="conv_block_2"))
    siamese_caps_net.add(layers.BatchNormalization())
    siamese_caps_net.add(layers.LeakyReLU())
    siamese_caps_net.add(layers.Conv2D(128, 5, strides=2, padding="valid", name="conv_block_3"))
    siamese_caps_net.add(layers.BatchNormalization())
    siamese_caps_net.add(layers.LeakyReLU())
    siamese_caps_net.add(layers.Conv2D(filters=16 * 32, kernel_size=9, strides=2, padding="same",
                                       name='primarycap_conv2d'))
    siamese_caps_net.add(layers.Reshape(target_shape=[-1, 16], name='primarycap_reshape'))
    siamese_caps_net.add(layers.Lambda(squash, name='primarycap_squash'))
    siamese_caps_net.add(FashionCaps(num_capsule=32, dim_capsule=8,
                                     routings=3, name='fashioncaps'))

    encoded_1 = siamese_caps_net(x1)
    encoded_2 = siamese_caps_net(x2)

    # out_caps_1 = Length(name='out_caps_1')(encoded_1)
    # out_caps_2 = Length(name='out_caps_2')(encoded_2)

    # Mask the output of FashionCapsNet
    y = layers.Input(shape=(1,))
    masked_by_y_1 = Mask()([encoded_1, y])  # The true label is used to mask the output of capsule layer. For training
    masked_1 = Mask()(encoded_1)  # Mask using the capsule with maximal length. For prediction
    masked_by_y_2 = Mask()([encoded_2, y])  # The true label is used to mask the output of capsule layer. For training
    masked_2 = Mask()(encoded_2)  # Mask using the capsule with maximal length. For prediction

    # Transpose-convolutional decoder network for reconstruction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(8*8*256, activation='relu', input_dim=256))
    decoder.add(layers.Reshape((8, 8, 256)))
    decoder.add(layers.Conv2DTranspose(256, kernel_size=9, strides=1, padding='same'))
    decoder.add(layers.BatchNormalization(axis=-1))
    decoder.add(layers.LeakyReLU())
    decoder.add(layers.Conv2DTranspose(128, kernel_size=9, strides=2, padding='same'))
    decoder.add(layers.BatchNormalization(axis=-1))
    decoder.add(layers.LeakyReLU())
    decoder.add(layers.Conv2DTranspose(64, kernel_size=7, strides=2, padding='same'))
    decoder.add(layers.BatchNormalization(axis=-1))
    decoder.add(layers.LeakyReLU())
    decoder.add(layers.Conv2DTranspose(32, kernel_size=7, strides=2, padding='same'))
    decoder.add(layers.BatchNormalization(axis=-1))
    decoder.add(layers.LeakyReLU())
    # decoder.add(layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same'))
    # decoder.add(layers.BatchNormalization(axis=-1))
    # decoder.add(layers.LeakyReLU())
    decoder.add(layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                                       activation='sigmoid'))

    decoded_train_1 = decoder(masked_by_y_1)
    decoded_train_2 = decoder(masked_by_y_2)
    decoded_eval_1 = decoder(masked_1)
    decoded_eval_2 = decoder(masked_2)

    length_layer = Length(name='out_caps')
    fc = layers.Dense(units=128, name='fc')

    out_1 = length_layer(encoded_1)
    out_2 = length_layer(encoded_2)
    out_1 = fc(out_1)
    out_2 = fc(out_2)
    out = layers.Lambda(manhattan_dist, name="capsnet")([out_1, out_2])

    train_model = models.Model(inputs=[x1, x2, y], outputs=[out, decoded_train_1, decoded_train_2])
    eval_model = models.Model(inputs=[x1, x2], outputs=[out, decoded_eval_1, decoded_eval_2])
    # Model for training

    # train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])

    # Model for evaluation
    # eval_model = models.Model([x1, x2], [out_caps, decoder(masked)])
    return train_model, eval_model
    # return train_model, eval_model
