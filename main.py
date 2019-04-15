import os
import time
import numpy as np
from tqdm import tqdm
from keras import optimizers
from keras import backend as K
from keras.preprocessing import image
from config import get_arguments
from models import FashionSiameseCapsNet, MultiGPUNet
from utils import custom_generator, get_iterator, triplet_loss, decay_lr


def train(model, args):
    best = -1

    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr, amsgrad=True), loss=[triplet_loss])

    train_iterator = get_iterator(os.path.join(args.filepath, "train"), args.input_size,
                                               args.shift_fraction, args.hor_flip, args.whitening,
                                               args.rotation_range, args.brightness_range, args.shear_range,
                                               args.zoom_range)
    train_generator = custom_generator(train_iterator)
    losses = list()
    for i in range(args.epochs):
        total_loss = 0
        retr_acc = 0  # TODO
        print("Epoch (" + str(i+1) + "/" + str(args.epochs) + "):")
        t_start = time.time()
        for j in tqdm(range(len(train_iterator)), ncols=50):
            x, y = next(train_generator)
            loss = model.train_on_batch(x, y)
            total_loss += loss
            # p = model.predict(x)

            print("Loss: {:.4f} \t"
                  "Loss at particular batch: {:.4f}".format(total_loss/(j+1), loss) + "\r", end="")
            # print("Total loss: {:.4f} \t"
            #       "Binary Cross-Entropy Loss: {:.4f} \t"
            #       "Reconstruction Loss: {:.4f} \t".format(loss[0], loss[1], loss[2]) + "\r", end="")
        losses.append(total_loss)

        # To see the encodings, make prediction.
        x, y = next(train_generator)
        p = model.predict(x)
        print(p)

        print("Epoch (" + str(i+1) + "/" + str(args.epochs) + ") completed in " + str(time.time()-t_start) + " secs.")

        if retr_acc > best:
            print("\tCurrent best: {:2.4f}, previous best: {:2.4f}".format(retr_acc, best))
            print("\tSaving weights to {} \n".format(args.save_dir))
            model.save_weights(os.path.join(args.save_dir, "weights-" + str(i) + ".h5"))
            best = retr_acc
        else:
            print("\tNot improved the best accuracy ({:2.4f})".format(best))

        # LR scheduling
        K.set_value(model.optimizer.lr, decay_lr(K.get_value(model.optimizer.lr), 0.9))

    # Model saving
    model_path = 't_model.h5'
    model.save(os.path.join(args.save_dir, model_path))
    print('The model file saved to \'%s' + model_path + '\'' % args.save_dir)


def test(model, args):
    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[triplet_loss],
                  # loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    # TODO
    # Decide that do we need that method?

    # # Evaluate the model using custom generator
    # scores = model.evaluate_generator(generator=custom_generator(get_iterator(args.filepath,
    #                                                                           subset="test")),
    #                                   steps=int(40000 / args.batch_size))
    # print(scores)

    # TODO
    # Reconstruct batch of images
    # if args.recon:
        # x_test_batch, y_test_batch = get_iterator(args.filepath, subset="test").next()
        # y_pred, x_recon = model.predict(x_test_batch)
        #
        # # Save reconstructed and original images
        # save_recons(x_recon, x_test_batch, y_pred, y_test_batch, args.save_dir)


if __name__ == '__main__':
    K.clear_session()
    args = get_arguments()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = FashionSiameseCapsNet(input_shape=(args.input_size, args.input_size, 3))

    if args.weights is not None:
        model.load_weights(args.weights)

    model.summary()

    if args.multi_gpu:
        p_model = MultiGPUNet(model, args.multi_gpu)

    if not args.testing:
        if args.multi_gpu:
            train(model=p_model, args=args)
            # implicitly sure that p_model defined
        else:
            train(model=model, args=args)
    else:
        if args.weights is None:
            print('Random initialization of weights.')
        test(model=model, args=args)
