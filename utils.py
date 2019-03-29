import os
import numpy as np
from random import shuffle
from keras import backend as K
from keras.preprocessing import image
from SiameseDirectoryIterator import SiameseDirectoryIterator


def siamese_accuracy(y_true, y_pred):
    margin = 0.5
    return K.mean(K.not_equal(y_true, K.cast(K.greater_equal(y_pred, margin), dtype=K.floatx())))


def contrastive_margin_loss(y_true, y_pred):
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


def create_one_shot_task(it_1, it_2, input_size, N):
    pairs = [np.zeros((N, input_size, input_size, 3)) for _ in range(2)]
    targets = np.zeros((N,))
    targets[0] = 1

    i = 0
    while True:
        if i == N:
            break

        if i % N == 0:
            a, b = next(it_1)

        pairs[0][i, :, :, :] = np.asarray(a, dtype="float")

        c, d = next(it_2)

        if i % N == 0:
            while np.argmax(d) != np.argmax(b):
                c, d = next(it_2)
        else:
            while np.argmax(d) == np.argmax(b):
                c, d = next(it_2)

        pairs[1][i, :, :, :] = np.asarray(c, dtype="float")
        i += 1

    negs = pairs[0]
    poss = pairs[1]
    tmp = list(zip(negs, poss, targets))
    shuffle(tmp)
    negs, poss, targets = zip(*tmp)

    pairs[0] = np.asarray(negs)
    pairs[1] = np.asarray(poss)
    pairs = np.asarray(pairs)
    targets = np.asarray(targets)

    return pairs, targets


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
