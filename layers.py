from keras import backend as K
from keras import layers, initializers, activations
from utils import squash
import tensorflow as tf


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
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 share_weights=True, activation='squash', kernel_initializer='glorot_uniform', **kwargs):
        super(FashionCaps, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        self.kernel_initializer = initializers.get(kernel_initializer)
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        input_num_capsule = input_shape[1]
        input_dim_capsule = input_shape[2]
        if self.share_weights:
            self.kernel = self.add_weight(name='capsule_kernel',
                                          shape=(1, input_dim_capsule, self.num_capsule * self.dim_capsule),
                                          initializer=self.kernel_initializer,
                                          trainable=True)
        else:
            self.kernel = self.add_weight(name='capsule_kernel',
                                          shape=(input_num_capsule, input_dim_capsule,
                                                 self.num_capsule * self.dim_capsule),
                                          initializer=self.kernel_initializer,
                                          trainable=True)

    def call(self, inputs, training=None):
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = tf.nn.softmax(b, dim=1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])

        # # # AVARAJ ROUT # # #
        # norm_hat_inputs = tf.norm(hat_inputs, axis=-1)
        # weighted_hat_inputs = hat_inputs * tf.expand_dims(norm_hat_inputs, axis=-1)
        # o = K.sum(weighted_hat_inputs, axis=2) / self.dim_capsule
        # o = self.activation(o)

        return o

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {'num_capsule': self.num_capsule,
                  'dim_capsule': self.dim_capsule,
                  'routings': self.routings}
        base_config = super(FashionCaps, self).get_config()
        new_config = list(base_config.items()) + list(config.items())
        return dict(new_config)
