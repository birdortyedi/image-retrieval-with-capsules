import os
import time
import numpy as np
from tqdm import tqdm
from keras import optimizers, callbacks
from keras import backend as K
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

    lr_scheduler = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    lr_scheduler.set_model(model)

    train_iterator = get_iterator(os.path.join(args.filepath, "train"), args.input_size, args.batch_size,
                                  args.shift_fraction, args.hor_flip, args.whitening, args.rotation_range,
                                  args.brightness_range, args.shear_range, args.zoom_range)
    train_generator = custom_generator(train_iterator)

    losses = list()
    for i in range(args.epochs):
        total_loss = 0

        print("Epoch (" + str(i+1) + "/" + str(args.epochs) + "):")
        t_start = time.time()
        lr_scheduler.on_epoch_begin(i)

        for j in tqdm(range(len(train_iterator)), ncols=100):
            x, y = next(train_generator)
            loss, _ = model.train_on_batch(x, y)
            total_loss += loss

            print("\tLoss: {:.4f}"
                  "\tLoss at particular batch: {:.4f}".format(total_loss/(j+1), loss) + "\r", end="")
            # print("Total loss: {:.4f} \t"
            #       "Binary Cross-Entropy Loss: {:.4f} \t"
            #       "Reconstruction Loss: {:.4f} \t".format(loss[0], loss[1], loss[2]) + "\r", end="")

        # To see the encodings, make prediction.
        # x, y = next(train_generator)
        # p, _ = model.predict(x)
        # print(p)

        print("\nEpoch ({}/{}) completed in {:5.6f} secs.".format(i+1, args.epochs, time.time()-t_start))

        # On epoch end loss and improved or not
        on_epoch_end_loss = total_loss/len(train_iterator)
        print("On epoch end loss: {:.6f}".format(on_epoch_end_loss))
        if len(losses) > 0:
            if np.min(losses) > on_epoch_end_loss:
                print("\nSaving weights to {}".format(os.path.join(args.save_dir, "weights-" + str(i) + ".h5")))
                if os.path.isfile(os.path.join(args.save_dir, "weights-" + str(np.argmin(losses)) + ".h5")):
                    os.remove(os.path.join(args.save_dir, "weights-" + str(np.argmin(losses)) + ".h5"))
                model.save_weights(os.path.join(args.save_dir, "weights-" + str(i) + ".h5"))
            else:
                print("\nLoss value {:.6f} not improved from ({:.6f})".format(on_epoch_end_loss, np.min(losses)))
        else:
            print("\nSaving weights to {}".format(os.path.join(args.save_dir, "weights-" + str(i) + ".h5")))
            model.save_weights(os.path.join(args.save_dir, "weights-" + str(i) + ".h5"))

        losses.append(on_epoch_end_loss)

        # LR scheduling
        lr_scheduler.on_epoch_end(i)
        print("\nLearning rate is reduced to {:.8f}.".format(K.get_value(model.optimizer.lr)))

        # Tensorboard
        tensorboard.on_epoch_end(i, {"Loss": on_epoch_end_loss,
                                     "Learning rate": K.get_value(model.optimizer.lr)})

    tensorboard.on_train_end(None)

    # Model saving
    model_path = 't_model.h5'
    model.save(os.path.join(args.save_dir, model_path))
    print("The model file saved to \"{}\"".format(os.path.join(args.save_dir, model_path)))


def test(model, args, query_len=None, gallery_len=None):
    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr, amsgrad=True), loss=[triplet_loss, None])

    query_iterator = get_iterator(os.path.join(args.filepath, "query"), args.input_size, 1,
                                  0., False, False, 0, None, 0., 0., False)
    query_generator = custom_generator(query_iterator, False)

    gallery_iterator = get_iterator(os.path.join(args.filepath, "gallery"), args.input_size, args.batch_size,
                                    0., False, False, 0, None, 0., 0., False)
    gallery_generator = custom_generator(gallery_iterator, False)

    if query_len is None:
        query_len = len(query_iterator)
    else:
        assert query_len <= len(query_iterator)

    if gallery_len is None:
        gallery_len = len(gallery_iterator)
    else:
        assert gallery_len <= len(gallery_iterator)

    relevant, retrieved = 0, 0
    for i in range(query_len):
        query_x, query_y = next(query_generator)
        query_x = np.array(query_x).reshape((np.array(query_x).shape[1:]))

        results = list()
        for j in range(gallery_len):
            gallery_xs, gallery_ys = next(gallery_generator)
            gallery_xs = np.array(gallery_xs).reshape((np.array(gallery_xs).shape[1:]))

            query_xs = np.repeat(query_x, len(gallery_xs), axis=0)

            _, y_pred = model.predict_on_batch([query_xs, gallery_xs, gallery_xs])

            for k, (e, y) in enumerate(zip(y_pred, gallery_ys[0])):
                dist = np.sum(np.square(e[:int(len(e)/2)] - e[int(len(e)/2):]))
                results.append({"distance": dist, "label": y["class_idx"], "item_idx": y["item_idx"]})

        results = sorted(results, key=lambda r: r["distance"])
        results = results[:args.top_k]

        relevance_results = np.array([True if r["label"] == query_y[0][0]["class_idx"] else False
                                      for r in results])
        retrieval_results = np.array([True if r["item_idx"] == query_y[0][0]["item_idx"] else False
                                      for r in results])

        if relevance_results.any():
            relevant += 1

        if retrieval_results.any():
            retrieved += 1

        print("{} of {} query images have been completed with the "
              "relevance accuracy of {:2.2f}% and "
              "the retrieval accuracy of {:2.2f}%.".format(i+1, query_len,
                                                           np.round(100*relevant/(i+1), 2),
                                                           np.round(100*retrieved/(i+1), 2)))

    retrieval_acc = 100 * retrieved / query_len
    relevance_acc = 100 * relevant / query_len

    print("The model has successfully retrieved {} images of {} relevant images in {} query images"
          "\nThe top-{} relevance accuracy:\t{:2.2f}"
          "\nThe top-{} retrieval accuracy:\t{:2.2f}\n".format(retrieved, relevant, query_len,
                                                               args.top_k, relevance_acc,
                                                               args.top_k, retrieval_acc))

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

    if args.multi_gpu and args.multi_gpu >= 2:
        p_model = MultiGPUNet(model, args.multi_gpu)

    if not args.testing:
        if args.multi_gpu and args.multi_gpu >= 2:
            train(model=p_model, args=args)
            # implicitly sure that p_model defined
        else:
            train(model=model, args=args)
    else:
        if args.weights is None:
            print('Random initialization of weights.')
        test(model=model, args=args)
