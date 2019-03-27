import os
import time
import numpy as np
from tqdm import tqdm
from keras import optimizers, callbacks
from keras import backend as K
from config import get_arguments
from models import FashionSiameseCapsNet, MultiGPUNet
from utils import custom_generator, get_iterator, contrasive_margin_loss, siamese_accuracy


def make_one_shot(model, N, k):
    one_shot_generator = custom_generator(get_iterator(args.filepath, args.input_size, args.shift_fraction,
                                                       args.hor_flip, args.whitening, args.rotation_range,
                                                       args.brightness_range, args.shear_range, args.zoom_range,
                                                       subset="query"),
                                          testing=args.testing)
    correct_preds = 0
    print("Evaluating model on {} random {} way one-shot learning tasks...\n".format(k, N))

    for i in range(k):
        inputs, targets = next(one_shot_generator)
        probs = model.predict(inputs)  # TODO HERE
        # if np.argmax(probs) == np.argmax(targets):
        #     correct_preds += 1

    print("Got an average of {:2.2f}% {} way one-shot learning accuracy \n".format((100.0 * correct_preds / k), N))
    return correct_preds / k


def train(model, args):
    best = -1
    step = int(25882 / args.batch_size)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr, amsgrad=True),
                  loss=[contrasive_margin_loss, 'mse', 'mse'],
                  loss_weights=[1., args.lam_recon, args.lam_recon])

    train_generator = custom_generator(get_iterator(args.filepath, args.input_size, args.shift_fraction, args.hor_flip,
                                                    args.whitening, args.rotation_range, args.brightness_range,
                                                    args.shear_range, args.zoom_range, subset="train"),
                                       testing=args.testing)

    for i in range(args.epochs):
        print("Epoch (" + str(i) + "/" + str(args.epochs) + "):")
        t_start = time.time()
        for _ in tqdm(range(step), ncols=50):
            x, y = next(train_generator)
            loss = model.train_on_batch(x, y)

            print("Total loss: {:.4f} \t"
                  "Contrasive Margin Loss: {:.4f} \t"
                  "Reconstruction Loss for Pair 1: {:.4f} \t"
                  "Reconstruction Loss for Pair 2: {:.4f} \t".format(loss[0], loss[1], loss[2], loss[3]) + "\r", end="")

        print("Epoch (" + str(i) + "/" + str(args.epochs) + ") completed in " + str(time.time()-t_start) + " secs.")
        val_acc = make_one_shot(model, N=25, k=21311)
        if val_acc >= best:
            print("\tCurrent best: {:2.4f}, previous best: {:2.4f}".format(val_acc, best))
            print("\tSaving weights to {} \n".format(args.save_dir))
            model.save_weights(args.save_dir)
            best = val_acc

    model_path = '/t_model.h5'
    model.save(args.save_dir + model_path)
    print('The model file saved to \'%s' + model_path + '\'' % args.save_dir)


def test(model, args):
    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[contrasive_margin_loss],
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

    model, eval_model = FashionSiameseCapsNet(input_shape=(args.input_size, args.input_size, 3))

    if args.weights is not None:
        model.load_weights(args.weights)

    model.summary()

    if args.multi_gpu:
        p_model = MultiGPUNet(model, args.multi_gpu)
        # p_eval_model = MultiGPUNet(eval_model, args.multi_gpu)

    if not args.testing:
        if args.multi_gpu:
            train(model=p_model, args=args)
            # implicitly sure that p_model defined
        else:
            train(model=model, args=args)
    else:
        if args.weights is None:
            print('Random initialization of weights.')
        test(model=eval_model, args=args)
