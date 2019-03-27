import os
from keras import backend as K
from keras.preprocessing import image
from SiameseDirectoryIterator import SiameseDirectoryIterator


def siamese_accuracy(y_true, y_pred):
    margin = 0.5
    return K.mean(K.not_equal(y_true, K.cast(K.greater_equal(y_pred, margin), dtype=K.floatx())))


def contrasive_margin_loss(y_true, y_pred):
    margin = 0.5
    return K.mean((1 - y_true) * 0.5 * y_pred + 0.5 * y_true * K.maximum(margin - y_pred, 0))


def squash(activations, axis=-1):
    scale = K.sum(K.square(activations), axis, keepdims=True) / \
            (1 + K.sum(K.square(activations), axis, keepdims=True)) / \
            K.sqrt(K.sum(K.square(activations), axis, keepdims=True) + K.epsilon())
    return scale * activations


def manhattan_dist(x):
    return K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)


def custom_generator(iterator, testing=True):
    if testing:
        while True:
            pairs_batch, y_batch = iterator.next()
            yield ([pairs_batch[0], pairs_batch[1]],
                   [y_batch, pairs_batch[0], pairs_batch[1]])
    else:
        while True:
            pairs_batch, y_batch = iterator.next()
            yield ([pairs_batch[0], pairs_batch[1], y_batch],
                   [y_batch, pairs_batch[0], pairs_batch[1]])


def get_iterator(file_path, input_size=256,
                 shift_fraction=0., h_flip=False, zca_whit=False, rot_range=0.,
                 bright_range=0., shear_range=0., zoom_range=0., subset="train"):

    file_path = os.path.join(file_path, subset)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if subset == "train":
        data_gen = image.ImageDataGenerator(width_shift_range=shift_fraction,
                                            height_shift_range=shift_fraction,
                                            horizontal_flip=h_flip,
                                            zca_whitening=zca_whit,
                                            rotation_range=rot_range,
                                            brightness_range=bright_range,
                                            shear_range=shear_range,
                                            zoom_range=zoom_range,
                                            rescale=1./255)
    else:
        data_gen = image.ImageDataGenerator(rescale=1./255)

    t_iterator = SiameseDirectoryIterator(file_path, data_gen, target_size=(input_size, input_size))
    return t_iterator
