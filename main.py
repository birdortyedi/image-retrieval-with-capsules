import os
import time
import numpy as np
from tqdm import tqdm
from keras import optimizers
from keras import backend as K
from keras.preprocessing import image
from config import get_arguments
from models import FashionSiameseCapsNet, MultiGPUNet
from utils import custom_generator, get_iterator, contrastive_margin_loss, create_one_shot_task


# TODO
def make_one_shot(model, file_path, subset, input_size, N, k):
    file_path_1 = os.path.join(file_path, subset[0])
    file_path_2 = os.path.join(file_path, subset[1])

    data_gen = image.ImageDataGenerator(rescale=1. / 255)

    it_1 = image.DirectoryIterator(directory=file_path_1, image_data_generator=data_gen,
                                   target_size=(input_size, input_size), batch_size=1)
    it_2 = image.DirectoryIterator(directory=file_path_2, image_data_generator=data_gen,
                                   target_size=(input_size, input_size), batch_size=1)
    correct_preds = 0
    print("Evaluating model on {} random {} way one-shot learning tasks...\n".format(k, N))

    for i in range(k):
        pairs, targets = create_one_shot_task(it_1=it_1, it_2=it_2, input_size=input_size, N=N)
        probs = model.predict([pairs[0], pairs[1]])
        if np.argmax(np.asarray(probs[0])) == np.argmax(targets):
            correct_preds += 1

    print("Got an average of {:2.2f}% {} way one-shot learning accuracy \n".format((100.0 * correct_preds / k), N))
    return correct_preds / k


def train(model, args):
    best = -1

    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr, amsgrad=True), loss=[contrastive_margin_loss])

    train_iterator = get_iterator(os.path.join(args.filepath, "train"), args.input_size,
                                                    args.shift_fraction, args.hor_flip, args.whitening,
                                                    args.rotation_range, args.brightness_range, args.shear_range,
                                                    args.zoom_range)
    train_generator = custom_generator(train_iterator)

    for i in range(args.epochs):
        total_loss = 0
        print("Epoch (" + str(i) + "/" + str(args.epochs) + "):")
        t_start = time.time()
        for j in tqdm(range(len(train_iterator)), ncols=50):
            x, y = next(train_generator)
            loss = model.train_on_batch(x, y)
            total_loss += loss

            print("Loss: {:.4f} \t"
                  "Loss at particular batch: {:.4f}".format(total_loss/(j+1), loss) + "\r", end="")
            # print("Total loss: {:.4f} \t"
            #       "Binary Cross-Entropy Loss: {:.4f} \t"
            #       "Reconstruction Loss: {:.4f} \t".format(loss[0], loss[1], loss[2]) + "\r", end="")

        print("Epoch (" + str(i) + "/" + str(args.epochs) + ") completed in " + str(time.time()-t_start) + " secs.")
        val_acc = make_one_shot(model, file_path=args.filepath, subset=["query", "gallery"],
                                input_size=args.input_size, N=9, k=50)
        if val_acc >= best:
            print("\tCurrent best: {:2.4f}, previous best: {:2.4f}".format(val_acc, best))
            print("\tSaving weights to {} \n".format(args.save_dir))
            model.save_weights(os.path.join(args.save_dir, "weights-" + str(i) + ".h5"))
            best = val_acc

    model_path = 't_model.h5'
    model.save(os.path.join(args.save_dir, model_path))
    print('The model file saved to \'%s' + model_path + '\'' % args.save_dir)


def test(model, args):
    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[contrastive_margin_loss],
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
