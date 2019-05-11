from keras import backend as K
from keras import models, layers
from keras.utils import multi_gpu_model
from layers import siamese_capsule_model, decoder_model, Mask, Length


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


def FashionSiameseCapsNet(input_shape, args):
    x = layers.Input(shape=input_shape)

    siamese_capsule = models.Model(x, siamese_capsule_model(x, args))
    siamese_capsule.summary()

    x1 = layers.Input(shape=input_shape)
    x2 = layers.Input(shape=input_shape)
    x3 = layers.Input(shape=input_shape)

    anchor_encoding = siamese_capsule(x1)
    positive_encoding = siamese_capsule(x2)
    negative_encoding = siamese_capsule(x3)

    # shape: (None, NUM_CLASS, DIM_CAPSULE)

    l2_norm = layers.Lambda(lambda enc: K.l2_normalize(enc, axis=-1))
    l2_anchor_encoding = l2_norm(anchor_encoding)
    l2_positive_encoding = l2_norm(positive_encoding)
    l2_negative_encoding = l2_norm(negative_encoding)

    y = layers.Input(shape=(args.num_class,))

    masked_anchor_encoding = Mask()([l2_anchor_encoding, y])
    masked_positive_encoding = Mask()([l2_positive_encoding, y])
    masked_negative_encoding = Mask()([l2_negative_encoding, y])

    # shape: (None, NUM_CLASS*DIM_CAPSULE)

    out = layers.Concatenate()([masked_anchor_encoding, masked_positive_encoding, masked_negative_encoding])

    # z = layers.Input(shape=(args.num_class*args.dim_capsule,))
    # decoder = models.Model(z, decoder_model(z))
    # decoder.summary()
    #
    # anchor_decoding = decoder(masked_anchor_encoding)

    model = models.Model(inputs=[x1, x2, x3, y], outputs=[out])
    eval_model = models.Model(inputs=[x1, y], outputs=masked_anchor_encoding)

    return model, eval_model
