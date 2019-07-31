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


def residual_block(y, nb_channels, _strides=(2, 2), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y


def inception_block(y, nb_channels, k_size=3, name=None):
    br_0 = layers.Conv2D(nb_channels, k_size, padding="same", name=name + "_br_0")(y)
    br_0 = layers.BatchNormalization()(br_0)
    br_1 = layers.Conv2D(nb_channels, k_size, padding="same", name=name + "_br_1_0")(y)
    br_1 = layers.Conv2D(2 * nb_channels, k_size, padding="same", name=name + "_br_1_1")(br_1)
    br_1 = layers.BatchNormalization()(br_1)
    br_2 = layers.Conv2D(nb_channels, k_size, padding="same", name=name + "_br_2_0")(y)
    br_2 = layers.Conv2D(int(3 * nb_channels / 2), k_size, padding="same", name=name + "_br_2_1")(br_2)
    br_2 = layers.Conv2D(2 * nb_channels, k_size, padding="same", name=name + "_br_2_2")(br_2)
    br_2 = layers.BatchNormalization()(br_2)
    mix = layers.concatenate([br_0, br_1, br_2], axis=-1, name=name+"_concat")
    mix = layers.Conv2D(2 * nb_channels, kernel_size=1, strides=2, name=name+"_1_by_1")(mix)
    return layers.LeakyReLU()(mix)


def primary_capsule(inputs, dim_capsule, name, args, n_channels=32, kernel_size=7, strides=2, padding="same"):
    # inputs = inception_block(inputs, nb_channels=int(dim_capsule*n_channels/2), name=name+"_primary_conv")
    if args.model_type == "rc":
        inputs = residual_block(inputs, nb_channels=dim_capsule*n_channels, _project_shortcut=True)
    else:
        inputs = layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size, strides=strides,
                               padding=padding,
                               name=name + '_conv')(inputs)
    inputs = layers.Reshape(target_shape=[-1, dim_capsule], name=name+'_reshape')(inputs)
    return layers.Lambda(squash, name=name+'_squash')(inputs)


def capsule_model(inputs, args):
    out = conv_bn_block(inputs, filters=64, k_size=7, stride=2, padding="same", name="conv_block_1")

    # out = inception_block(out, nb_channels=64, name="inception0")
    # out = layers.SpatialDropout2D(rate=0.3)(out)
    # out = inception_block(out, nb_channels=128, name="inception1")
    # out = layers.SpatialDropout2D(rate=0.3)(out)

    if args.model_type == "rc":
        out = residual_block(out, nb_channels=128, _project_shortcut=True)
        out = layers.SpatialDropout2D(rate=0.3)(out)
        out = residual_block(out, nb_channels=256, _project_shortcut=True)
        out = layers.SpatialDropout2D(rate=0.3)(out)
    else:
        out = conv_bn_block(out, filters=128, k_size=7, stride=2, padding="same", name="conv_block_2")
        out = conv_bn_block(out, filters=64, k_size=7, stride=2, padding="same", name="conv_block_3")

    out = primary_capsule(out, dim_capsule=16, name="primarycaps", args=args)
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
