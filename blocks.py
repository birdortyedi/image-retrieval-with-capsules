from keras import layers
from utils import squash
from layers import FashionCaps


def conv_bn_block(inputs, filters, k_size, stride, padding, name):
    out = layers.Conv2D(filters=filters, kernel_size=k_size, strides=stride, padding=padding, name=name)(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    return layers.SpatialDropout2D(rate=0.3)(out)


def transpose_conv_bn_block(inputs, filters, k_size, stride, padding, name):
    out = layers.Conv2DTranspose(filters=filters, kernel_size=k_size, strides=stride, padding=padding,
                                 name=name)(inputs)
    out = layers.BatchNormalization(axis=-1)(out)
    out = layers.ReLU()(out)
    return layers.SpatialDropout2D(rate=0.3)(out)


def primary_capsule(inputs, dim_capsule, name, n_channels=32, kernel_size=7, strides=2, padding="same"):
    inputs = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name=name+'_conv')(inputs)
    inputs = layers.Reshape(target_shape=[-1, dim_capsule], name=name+'_reshape')(inputs)
    return layers.Lambda(squash, name=name+'_squash')(inputs)


def capsule_model(inputs, args):
    out = conv_bn_block(inputs, filters=64, k_size=7, stride=3, padding="valid", name="conv_block_1")
    out = conv_bn_block(out, filters=128, k_size=5, stride=2, padding="valid", name="conv_block_2")
    out = primary_capsule(out, dim_capsule=16, name="primarycaps")
    out = FashionCaps(num_capsule=args.num_class, dim_capsule=args.dim_capsule, routings=3, name="fashioncaps")(out)
    return out


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
