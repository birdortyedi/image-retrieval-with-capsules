import os
import time
import numpy as np
from tqdm import tqdm
from keras import optimizers, callbacks
from keras import backend as K
from keras.preprocessing import image
from config import get_arguments
from models import FashionSiameseCapsNet, MultiGPUNet
from utils import custom_generator, get_iterator, triplet_loss


def train(model, args):
    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr, amsgrad=True), loss=[triplet_loss, None])

    if not os.path.isdir(os.path.join(args.save_dir, "tensorboard-logs")):
        os.mkdir(os.path.join(args.save_dir, "tensorboard-logs"))

    tensorboard = callbacks.TensorBoard(log_dir=os.path.join(args.save_dir, "tensorboard-logs"),
                                        histogram_freq=0, batch_size=args.batch_size,
                                        write_graph=True, write_grads=True)
    tensorboard.set_model(model)

    # logger = callbacks.CSVLogger(os.path.join(args.save_dir, "log.csv"), append=True)
    # logger.set_model(model)
    # logger.on_train_begin()

    lr_scheduler = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    lr_scheduler.set_model(model)

    train_iterator = get_iterator(os.path.join(args.filepath, "train"), args.input_size,
                                               args.shift_fraction, args.hor_flip, args.whitening,
                                               args.rotation_range, args.brightness_range, args.shear_range,
                                               args.zoom_range)
    train_generator = custom_generator(train_iterator)

    losses = list()
    for i in range(args.epochs):
        total_loss = 0

        print("Epoch (" + str(i+1) + "/" + str(args.epochs) + "):")
        t_start = time.time()
        lr_scheduler.on_epoch_begin(i)

        for j in tqdm(range(len(train_iterator)), ncols=50):
            x, y = next(train_generator)
            loss, _ = model.train_on_batch(x, y)
            total_loss += loss

            print("Loss: {:.4f} \t"
                  "Loss at particular batch: {:.4f}".format(total_loss/(j+1), loss) + "\r", end="")
            # print("Total loss: {:.4f} \t"
            #       "Binary Cross-Entropy Loss: {:.4f} \t"
            #       "Reconstruction Loss: {:.4f} \t".format(loss[0], loss[1], loss[2]) + "\r", end="")

        # To see the encodings, make prediction.
        # x, y = next(train_generator)
        # p, _ = model.predict(x)
        # print(p)

        print("\nEpoch (" + str(i+1) + "/" + str(args.epochs) + ") completed in " + str(time.time()-t_start) + " secs.")

        # On epoch end loss and improved or not
        on_epoch_end_loss = total_loss/len(train_iterator)
        print("On epoch end loss: {}".format(on_epoch_end_loss))
        if len(losses) > 0:
            if np.min(losses) > on_epoch_end_loss:
                print("\nSaving weights to {}".format(os.path.join(args.save_dir, "weights-" + str(i+1) + ".h5")))
                if os.path.exists(os.path.join(args.save_dir, "weights-" + str(np.argmin(losses)) + ".h5")):
                    os.remove(os.path.join(args.save_dir, "weights-" + str(np.argmin(losses)) + ".h5"))
                model.save_weights(os.path.join(args.save_dir, "weights-" + str(i+1) + ".h5"))
            else:
                print("\nLoss value not improved from ({:.6f})".format(on_epoch_end_loss))
        else:
            print("\nSaving weights to {}".format(os.path.join(args.save_dir, "weights-" + str(i+1) + ".h5")))
            model.save_weights(os.path.join(args.save_dir, "weights-" + str(i+1) + ".h5"))

        losses.append(on_epoch_end_loss)

        # LR scheduling
        print("\nLearning rate is reduced to {}.".format(K.get_value(model.optimizer.lr)))
        lr_scheduler.on_epoch_end(i)

        # Tensorboard
        tensorboard.on_epoch_end(i, {"Loss": on_epoch_end_loss})

        # CSVLogger
        # logger.on_epoch_end(i, {"Learning rate": K.get_value(model.optimizer.lr),
        #                         "Loss": on_epoch_end_loss})

    tensorboard.on_train_end(None)
    # logger.on_train_end()

    # Model saving
    model_path = 't_model.h5'
    model.save(os.path.join(args.save_dir, model_path))
    print('The model file saved to \'%s' + model_path + '\'' % args.save_dir)


def test(model, args, query_len=None, gallery_len=None):
    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr, amsgrad=True), loss=[triplet_loss, None])

    data_gen = image.ImageDataGenerator(rescale=1./255)

    query_generator = data_gen.flow_from_directory(os.path.join(args.filepath, "query"),
                                                   target_size=(args.input_size, args.input_size),
                                                   shuffle=True,
                                                   batch_size=1)

    gallery_generator = data_gen.flow_from_directory(os.path.join(args.filepath, "gallery"),
                                                     target_size=(args.input_size, args.input_size),
                                                     shuffle=True,
                                                     batch_size=args.batch_size)

    if query_len is None:
        query_len = len(query_generator)
    else:
        assert query_len <= len(query_generator)

    if gallery_len is None:
        gallery_len = len(gallery_generator)
    else:
        assert gallery_len <= len(gallery_generator)

    retrieved = 0
    for i in range(query_len):
        query_x, query_y = next(query_generator)
        query_xs = np.repeat(query_x, args.batch_size, axis=0)

        results = list()
        for j in range(gallery_len):
            gallery_xs, gallery_ys = next(gallery_generator)
            _, y_pred = model.predict_on_batch([query_xs, gallery_xs, gallery_xs])

            for k, (e, y) in enumerate(zip(y_pred, gallery_ys)):
                dist = np.sum(np.square(e[:int(len(e)/2)] - e[int(len(e)/2):]))
                results.append({"distance": dist, "label": y})

        results = sorted(results, key=lambda r: r["distance"])
        results = [results[i] for i in range(len(results)) if i < args.k]
        results = np.array([True if r["label"] == query_y else False for r in results])

        if results.any():
            retrieved += 1

        print("{} of {} query images have been completed.".format(i+1, len(query_generator)))

    acc = retrieved / len(query_generator)
    print("The model has successfully retrieved {} images of {} query images"
          "\nThe top-{} retrieval accuracy:\t{}\n".format(retrieved, len(query_generator), args.top_k, acc))

    # TODO
    # # Reconstruct batch of images
    # if args.recon:
    #   x_test_batch, y_test_batch = get_iterator(args.filepath, subset="test").next()
    #   y_pred, x_recon = model.predict(x_test_batch)
    #
    #   # Save reconstructed and original images
    #   save_recons(x_recon, x_test_batch, y_pred, y_test_batch, args.save_dir)


if __name__ == '__main__':
    K.clear_session()
    args = get_arguments()
    print(args)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

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
