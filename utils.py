from keras import backend as K
from keras.preprocessing import image
from SiameseDirectoryIterator import SiameseDirectoryIterator


def triplet_loss(y_true, y_pred):
    enc_size = int(K.get_variable_shape(y_pred)[1]/3)
    anchor_encoding = y_pred[:, :enc_size]
    positive_encoding = y_pred[:, enc_size:2 * enc_size]
    negative_encoding = y_pred[:, 2 * enc_size:]
    margin = K.constant(2.0)

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor_encoding - positive_encoding), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor_encoding - negative_encoding), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + margin

    return K.mean(K.maximum(basic_loss, 0.0))


def squash(activations, axis=-1):
    scale = K.sum(K.square(activations), axis, keepdims=True) / \
            (1 + K.sum(K.square(activations), axis, keepdims=True)) / \
            K.sqrt(K.sum(K.square(activations), axis, keepdims=True) + K.epsilon())
    return scale * activations


def decay_lr(lr, rate):
    return lr * rate


def custom_generator(iterator):
    while True:
        pairs_batch, y_batch = iterator.next()
        yield ([pairs_batch[0], pairs_batch[1], pairs_batch[2]], [y_batch])


def get_iterator(file_path, input_size=256,
                 shift_fraction=0., h_flip=False, zca_whit=False, rot_range=0.,
                 bright_range=0., shear_range=0., zoom_range=0.):
    data_gen = image.ImageDataGenerator(width_shift_range=shift_fraction,
                                        height_shift_range=shift_fraction,
                                        horizontal_flip=h_flip,
                                        zca_whitening=zca_whit,
                                        rotation_range=rot_range,
                                        brightness_range=bright_range,
                                        shear_range=shear_range,
                                        zoom_range=zoom_range,
                                        rescale=1./255)
    t_iterator = SiameseDirectoryIterator(directory=file_path, image_data_generator=data_gen,
                                          target_size=(input_size, input_size))

    return t_iterator
