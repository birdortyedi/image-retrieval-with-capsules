from keras import backend as K
from keras import layers, initializers
from utils import squash, sampling
import tensorflow as tf


def l2_norm(x):
    return layers.Lambda(lambda data: K.l2_normalize(data, axis=-1), name="l2_norm")(x)


def conv_bn_block(inputs, filters, k_size, stride, padding, name):
    out = layers.Conv2D(filters=filters, kernel_size=k_size, strides=stride, padding=padding, name=name)(inputs)
    out = layers.BatchNormalization()(out)
    return layers.LeakyReLU()(out)


def transpose_conv_bn_block(inputs, filters, k_size, stride, padding, name):
    out = layers.Conv2DTranspose(filters=filters, kernel_size=k_size, strides=stride, padding=padding,
                                 name=name)(inputs)
    out = layers.BatchNormalization(axis=-1)(out)
    return layers.LeakyReLU()(out)


def primary_capsule(inputs, dim_capsule, name, n_channels=32, kernel_size=9, strides=2, padding="same"):
    inputs = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name=name+'_conv')(inputs)
    inputs = layers.Reshape(target_shape=[-1, dim_capsule], name=name+'_reshape')(inputs)
    return layers.Lambda(squash, name=name+'_squash')(inputs)


def siamese_capsule_model(inputs):
    out = conv_bn_block(inputs, filters=64, k_size=9, stride=2, padding="same", name="conv_block_1")
    out = conv_bn_block(out, filters=128, k_size=7, stride=2, padding="same", name="conv_block_2")
    out = conv_bn_block(out, filters=256, k_size=5, stride=2, padding="same", name="conv_block_3")
    out = primary_capsule(out, dim_capsule=16, name="primarycaps")
    out = FashionCaps(num_capsule=16, dim_capsule=8, routings=3, name="fashioncaps")(out)
    out = layers.Flatten()(out)
    return l2_norm(out)


def decoder_model(inputs):
    out = layers.Dense(8*8*256, activation='relu')(inputs)
    out = layers.Reshape((8, 8, 256))(out)
    out = transpose_conv_bn_block(out, filters=128, k_size=9, stride=1, padding='same', name="t_conv_block_1")
    out = transpose_conv_bn_block(out, filters=64, k_size=7, stride=2, padding='same', name="t_conv_block_2")
    out = transpose_conv_bn_block(out, filters=32, k_size=7, stride=2, padding='same', name="t_conv_block_3")
    out = transpose_conv_bn_block(out, filters=16, k_size=5, stride=2, padding='same', name="t_conv_block_4")
    out = transpose_conv_bn_block(out, filters=16, k_size=5, stride=2, padding='same', name="t_conv_block_5")
    return layers.Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same', activation='sigmoid',
                                  name="decoder_out")(out)


def encoder_model(inputs):
    out = layers.Dense(64, activation='relu')(inputs)
    z_mu = layers.Dense(2, name='z_mean')(out)
    z_log_var = layers.Dense(2, name='z_log_var')(out)
    z = layers.Lambda(sampling, name='z')([z_mu, z_log_var])
    return z, z_mu, z_log_var


class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        return super(Length, self).get_config()


class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            mask = K.one_hot(indices=K.argmax(K.sqrt(K.sum(K.square(inputs), -1)), 1),
                             num_classes=K.sqrt(K.sum(K.square(inputs), -1)).get_shape().as_list()[1])

        return K.batch_flatten(inputs * K.expand_dims(mask, -1))

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        return super(Mask, self).get_config()


class FashionCaps(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_initializer='glorot_uniform', **kwargs):
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(FashionCaps, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')
        self.built = True

    def call(self, inputs, training=None):
        inputs = K.expand_dims(inputs, 1)
        inputs = K.tile(inputs, [1, self.num_capsule, 1, 1])
        inputs = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs)

        # Dynamic routing
        b = tf.zeros(shape=[K.shape(inputs)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0
        for i in range(self.routings):
            outputs = squash(K.batch_dot(tf.nn.softmax(b, dim=1), inputs, [2, 2]))

            if i < self.routings - 1:
                b += K.batch_dot(outputs, inputs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(FashionCaps, self).get_config()
        new_config = list(base_config.items()) + list(config.items())
        return dict(new_config)
