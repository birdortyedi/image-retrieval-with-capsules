from keras import backend as K
from keras import models, layers
from keras.utils import multi_gpu_model
from layers import siamese_capsule_model, decoder_model, encoder_model


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

    out = layers.Concatenate()([anchor_encoding, positive_encoding, negative_encoding])

    # y = layers.Input(shape=(128,))
    # encoder = models.Model(y, encoder_model(y))
    # encoder.summary()
    #
    # anchor_latent_vector = encoder(anchor_encoding)
    # positive_latent_vector = encoder(positive_encoding)
    # negative_latent_vector = encoder(negative_encoding)

    y = layers.Input(shape=(128,))
    decoder = models.Model(y, decoder_model(y))
    decoder.summary()

    anchor_decoding = decoder(anchor_encoding)
    positive_decoding = decoder(positive_encoding)
    negative_decoding = decoder(negative_encoding)

    model = models.Model(inputs=[x1, x2, x3], outputs=[out, anchor_decoding, positive_decoding, negative_decoding])
    eval_model = models.Model(inputs=x1, outputs=anchor_encoding)

    return model, eval_model
