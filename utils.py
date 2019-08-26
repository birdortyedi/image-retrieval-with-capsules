from keras import backend as K
from keras.metrics import kullback_leibler_divergence
from keras.preprocessing import image
from TripletDirectoryIterator import TripletDirectoryIterator
from config import get_arguments

args = get_arguments()


def triplet_eucliden_loss(y_true, y_pred):
    enc_size = int(K.get_variable_shape(y_pred)[1]/3)
    anchor_encoding = y_pred[:, :enc_size]
    positive_encoding = y_pred[:, enc_size:2 * enc_size]
    negative_encoding = y_pred[:, 2 * enc_size:]
    margin = K.constant(2.0)

    def euclidean_dist(a, e):
        return K.sum(K.square(a - e), axis=-1)  # squared euclidean distance
        # return K.sqrt(K.sum(K.square(a - e), axis=-1))  # original euclidean distance

    pos_dist = euclidean_dist(anchor_encoding, positive_encoding)
    neg_dist = euclidean_dist(anchor_encoding, negative_encoding)
    basic_loss = pos_dist - neg_dist + margin

    return K.mean(K.maximum(basic_loss, 0.0))


def triplet_cosine_loss(y_true, y_pred):
    enc_size = int(K.get_variable_shape(y_pred)[1] / 3)
    anchor_encoding = y_pred[:, :enc_size]
    positive_encoding = y_pred[:, enc_size:2 * enc_size]
    negative_encoding = y_pred[:, 2 * enc_size:]

    def cosine_similarity(a, e):
        return K.batch_dot(a, e, axes=-1)  # simple dot product since vectors are l2_normed and pdf

    pos_sim = cosine_similarity(anchor_encoding, positive_encoding)
    neg_sim = cosine_similarity(anchor_encoding, negative_encoding)

    return K.mean(K.sum(K.log(1 + K.exp(-(pos_sim - neg_sim))), axis=-1))


def margin_loss(y_true, y_pred):
    m_plus = 0.9
    m_minus = 1 - m_plus
    lamb = 0.5

    loss = y_true * K.square(K.relu(m_plus - y_pred)) + \
        lamb * (1 - y_true) * K.square(K.relu(y_pred - m_minus))

    return K.sum(loss, axis=-1)


def kl_divergence(y_true, y_pred):
    alpha = 0.9
    beta = 0.1
    enc_size = int(K.get_variable_shape(y_pred)[1] / 3)
    anchor_encoding = y_pred[:, :enc_size]
    positive_encoding = y_pred[:, enc_size:2 * enc_size]
    negative_encoding = y_pred[:, 2 * enc_size:]

    return alpha * kullback_leibler_divergence(anchor_encoding, positive_encoding) + \
        beta * kullback_leibler_divergence(anchor_encoding, (K.reverse(negative_encoding, axes=-1)))


def squash(activations, axis=-1):
    scale = K.sum(K.square(activations), axis, keepdims=True) / \
            (1 + K.sum(K.square(activations), axis, keepdims=True)) / \
            K.sqrt(K.sum(K.square(activations), axis, keepdims=True) + K.epsilon())
    return scale * activations


def decay_lr(lr, rate):
    return lr * rate


def custom_generator(it):
    while True:
        pairs_batch, y_batch = it.next()
        yield ([pairs_batch[0], pairs_batch[1], pairs_batch[2], y_batch[0]],
               [y_batch[0], y_batch[0], y_batch[1], y_batch[2]])


def get_iterator(file_path, input_size=256, batch_size=32,
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
    t_iterator = TripletDirectoryIterator(directory=file_path, image_data_generator=data_gen,
                                          batch_size=batch_size, target_size=(input_size, input_size))

    return t_iterator

