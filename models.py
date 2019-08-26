from keras import backend as K
from keras import models, layers
from keras.utils import multi_gpu_model
from layers import Mask, Length
from blocks import capsule_model


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


def FashionTripletCapsNet(input_shape, args):
    x = layers.Input(shape=input_shape)

    caps_model = models.Model(x, capsule_model(x, args))
    caps_model.summary()

    x1 = layers.Input(shape=input_shape)
    x2 = layers.Input(shape=input_shape)
    x3 = layers.Input(shape=input_shape)

    anchor_encoding = caps_model(x1)
    positive_encoding = caps_model(x2)
    negative_encoding = caps_model(x3)

    # shape: (None, NUM_CLASS, DIM_CAPSULE)

    l2_norm = layers.Lambda(lambda enc: K.l2_normalize(enc, axis=-1) + K.epsilon())
    l2_anchor_encoding = l2_norm(anchor_encoding)
    l2_positive_encoding = l2_norm(positive_encoding)
    l2_negative_encoding = l2_norm(negative_encoding)

    y1 = layers.Input(shape=(args.num_class,))

    masked_anchor_encoding = Mask(name="anchor_mask")([l2_anchor_encoding, y1])
    masked_positive_encoding = Mask(name="positive_mask")([l2_positive_encoding, y1])
    masked_negative_encoding = Mask(name="negative_mask")([l2_negative_encoding, y1])

    # shape: (None, NUM_CLASS*DIM_CAPSULE)

    out = layers.Concatenate()([masked_anchor_encoding, masked_positive_encoding, masked_negative_encoding])

    cls_out_anchor = Length(name="anchor_class")(anchor_encoding)
    cls_out_positive = Length(name="positive_class")(positive_encoding)
    cls_out_negative = Length(name="negative_class")(negative_encoding)

    model = models.Model(inputs=[x1, x2, x3, y1],
                         outputs=[out, cls_out_anchor, cls_out_positive, cls_out_negative])
    eval_model = models.Model(inputs=[x1, y1], outputs=masked_anchor_encoding)

    return model, eval_model
