from keras import backend as K
from keras import models, layers
from keras.utils import multi_gpu_model
from layers import siamese_capsule_model, l2_norm


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
    x = layers.Input(shape=input_shape)
    siamese_capsule = models.Model(x, siamese_capsule_model(x))

    siamese_capsule.summary()

    x1 = layers.Input(shape=input_shape)
    x2 = layers.Input(shape=input_shape)
    x3 = layers.Input(shape=input_shape)

    anchor_encoding = siamese_capsule(x1)
    positive_encoding = siamese_capsule(x2)
    negative_encoding = siamese_capsule(x3)

    l2_anchor_encoding = l2_norm(anchor_encoding)
    l2_positive_encoding = l2_norm(positive_encoding)
    l2_negative_encoding = l2_norm(negative_encoding)

    out = layers.Concatenate()([l2_anchor_encoding, l2_positive_encoding, l2_negative_encoding])
    # out_pred = layers.Concatenate()([l2_anchor_encoding])

    model = models.Model(inputs=[x1, x2, x3], outputs=[out, l2_anchor_encoding])

    return model

    # out_caps_1 = Length(name='out_caps_1')(encoded_1)
    # out_caps_2 = Length(name='out_caps_2')(encoded_2)

    # # Mask the output of FashionCapsNet
    # # y = layers.Input(shape=(2,))
    # # masked_by_y_1 = Mask()([encoded_1, y])  # The true label is used to mask the output of capsule layer. For training
    # masked_1 = Mask()(encoded_1)  # Mask using the capsule with maximal length. For prediction
    # # masked_by_y_2 = Mask()([encoded_2, y])  # The true label is used to mask the output of capsule layer. For training
    # masked_2 = Mask()(encoded_2)  # Mask using the capsule with maximal length. For prediction
    #
    # # Transpose-convolutional decoder network for reconstruction
    # decoder = models.Sequential(name='decoder')
    # decoder.add(layers.Dense(8*8*256, activation='relu', input_dim=256))
    # decoder.add(layers.Reshape((8, 8, 256)))
    # decoder.add(layers.Conv2DTranspose(256, kernel_size=9, strides=1, padding='same'))
    # decoder.add(layers.BatchNormalization(axis=-1))
    # decoder.add(layers.LeakyReLU())
    # decoder.add(layers.Conv2DTranspose(128, kernel_size=9, strides=2, padding='same'))
    # decoder.add(layers.BatchNormalization(axis=-1))
    # decoder.add(layers.LeakyReLU())
    # decoder.add(layers.Conv2DTranspose(64, kernel_size=7, strides=2, padding='same'))
    # decoder.add(layers.BatchNormalization(axis=-1))
    # decoder.add(layers.LeakyReLU())
    # decoder.add(layers.Conv2DTranspose(32, kernel_size=7, strides=2, padding='same'))
    # decoder.add(layers.BatchNormalization(axis=-1))
    # decoder.add(layers.LeakyReLU())
    # # decoder.add(layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding='same'))
    # # decoder.add(layers.BatchNormalization(axis=-1))
    # # decoder.add(layers.LeakyReLU())
    # decoder.add(layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
    #                                    activation='sigmoid'))
    #
    # decoded_train_1 = decoder(masked_1)
    # decoded_train_2 = decoder(masked_2)
    # decoded_dist_train = layers.Lambda(cosine_distance, output_shape=cos_dist_output_shape)([decoded_train_1,
    #                                                                                          decoded_train_2])
    #
    # # decoded_dist_train = layers.Lambda(manhattan_dist, name="decoded_distance")([decoded_train_1, decoded_train_2])
    #
    # decoded_eval_1 = decoder(masked_1)
    # decoded_eval_2 = decoder(masked_2)
    # decoded_dist_eval = layers.Lambda(cosine_distance, output_shape=cos_dist_output_shape)([decoded_eval_1,
    #                                                                                         decoded_eval_2])
    #
    # # decoded_dist_eval = layers.Lambda(manhattan_dist, name="decoded_distance")([decoded_eval_1, decoded_eval_2])

    # fc = layers.Dense(units=128, name='fc1')
    #
    # out_1 = length_layer(encoded_1)
    # out_2 = length_layer(encoded_2)
    #
    # # out_1 = fc(out_1)
    # # out_2 = fc(out_2)
    # # out = layers.Lambda(manhattan_dist, name="distance")([out_1, out_2])
    #
    # out = layers.Concatenate()([out_1, out_2])
    # out = layers.Dense(units=64, activation="relu", name="fc")(out)
    # out = layers.Dense(units=2, activation="softmax", name="capsnet")(out)
    #
    # train_model = models.Model(inputs=[x1, x2], outputs=[out])#, decoded_dist_train])
    # eval_model = models.Model(inputs=[x1, x2], outputs=[out])#, decoded_dist_eval])
    # # Model for training
    #
    # # train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    #
    # # Model for evaluation
    # # eval_model = models.Model([x1, x2], [out_caps, decoder(masked)])
    # return train_model, eval_model
    # # return train_model, eval_model
