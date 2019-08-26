import os
import time
import numpy as np
from tqdm import tqdm
from colorama import Fore
from keras import optimizers, callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from config import get_arguments
from models import FashionTripletCapsNet, MultiGPUNet
from utils import custom_generator, get_iterator, triplet_eucliden_loss, triplet_cosine_loss, margin_loss


def train(model, eval_model, args):
    # Compile the model
    if args.metric_type == "euclidean":
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[triplet_eucliden_loss, "mse", "mse", "mse"],
                      loss_weights=[1., 0., 0., 0.])
    elif args.metric_type == "cosine":
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[triplet_cosine_loss, margin_loss,
                            margin_loss, margin_loss],
                      loss_weights=[1., 0.2, 0.2, 0.2])
    else:
        raise Exception("Wrong metric type. Available: ['euclidean', 'cosine']")

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
    for i in range(args.initial_epoch, args.epochs):
        total_loss, total_triplet_loss = 0, 0
        total_anchor_xentr, total_positive_xentr, total_negative_xentr = 0, 0, 0

        print("Epoch (" + str(i+1) + "/" + str(args.epochs) + "):")
        t_start = time.time()
        lr_scheduler.on_epoch_begin(i)
        if i > 0:
            print("\nLearning rate is reduced to {:.8f}.".format(K.get_value(model.optimizer.lr)))

        for j in tqdm(range(len(train_iterator)), ncols=100, desc="Training",
                      bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
            x, y = next(train_generator)

            loss, triplet_loss_, anchor_xentr, positive_xentr, negative_xentr = model.train_on_batch(x, y)
            total_loss += loss
            total_triplet_loss += triplet_loss_
            total_anchor_xentr += anchor_xentr
            total_positive_xentr += positive_xentr
            total_negative_xentr += negative_xentr

            if args.metric_type == "euclidean":
                print("\tTotal Loss: {:.4f}"
                      "\tTriplet Loss: {:.4f}".format(total_loss / (j + 1),
                                                      total_triplet_loss / (j + 1)),  "\r", end="")
            elif args.metric_type == "cosine":
                print("\tTotal Loss: {:.4f}"
                      "\tTriplet: {:.4f}"
                      "\tA X-Ent: {:.4f}"
                      "\tP X-Ent: {:.4f}"
                      "\tN X-Ent: {:.4f}".format(total_loss / (j + 1),
                                                 total_triplet_loss / (j + 1),
                                                 total_anchor_xentr / (j + 1),
                                                 total_positive_xentr / (j + 1),
                                                 total_negative_xentr / (j + 1)), "\r", end="")
            else:
                Exception("Wrong metric type. Available: ['euclidean', 'cosine']")

        print("\nEpoch ({}/{}) completed in {:5.6f} secs.".format(i+1, args.epochs, time.time()-t_start))

        if i % 5 == 0:
            print("\nEvaluating the model...")
            test(model=eval_model, args=args)

        # On epoch end loss and improved or not
        on_epoch_end_loss = total_loss/len(train_iterator)
        on_epoch_end_triplet = total_triplet_loss/len(train_iterator)
        on_epoch_end_a_xentr = total_anchor_xentr / len(train_iterator)
        on_epoch_end_p_xentr = total_positive_xentr / len(train_iterator)
        on_epoch_end_n_xentr = total_negative_xentr / len(train_iterator)
        print("On epoch end loss: {:.6f}".format(on_epoch_end_loss))
        if len(losses) > 0:
            if np.min(losses) > on_epoch_end_loss:
                print("\nSaving weights to {}".format(os.path.join(args.save_dir, "weights-" + str(i+1) + ".h5")))
                # if os.path.isfile(os.path.join(args.save_dir, "weights-" + str(np.argmin(losses)) + ".h5")):
                #     os.remove(os.path.join(args.save_dir, "weights-" + str(np.argmin(losses)) + ".h5"))
                model.save_weights(os.path.join(args.save_dir, "weights-" + str(i+1) + ".h5"))
            else:
                print("\nLoss value {:.6f} not improved from ({:.6f})".format(on_epoch_end_loss, np.min(losses)))
        else:
            print("\nSaving weights to {}".format(os.path.join(args.save_dir, "weights-" + str(i+1) + ".h5")))
            model.save_weights(os.path.join(args.save_dir, "weights-" + str(i+1) + ".h5"))

        losses.append(on_epoch_end_loss)

        # LR scheduling
        lr_scheduler.on_epoch_end(i)

        # Tensorboard
        tensorboard.on_epoch_end(i, {"Total Loss": on_epoch_end_loss,
                                     "Triplet Loss": on_epoch_end_triplet,
                                     "Anchor X-Entropy Loss": on_epoch_end_a_xentr,
                                     "Positive X-Entropy Loss": on_epoch_end_p_xentr,
                                     "Negative X-Entropy Loss": on_epoch_end_n_xentr,
                                     "Learning rate": K.get_value(model.optimizer.lr)})

    tensorboard.on_train_end(None)

    # Model saving
    model_path = 't_model.h5'
    model.save(os.path.join(args.save_dir, model_path))
    print("The model file saved to \"{}\"".format(os.path.join(args.save_dir, model_path)))


def test(model, args):
    query_dict = extract_embeddings(model, args)
    gallery_dict = extract_embeddings(model, args, subset="gallery")

    results = list()
    print("Finding k closest images of gallery set to the query image...")
    t_start = time.time()
    for i in tqdm(range(len(query_dict["out"])), ncols=100, desc="Distance Calc",
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        q_result = list()
        # if np.argmax(query_dict["cls"][i]) == args.category or args.category == -1:
        for j in range(len(gallery_dict["out"])):
            if args.metric_type == "euclidean":
                q_result.append({"is_same_cls": (np.argmax(query_dict["cls"][i]) == np.argmax(gallery_dict["cls"][j])),
                                 "is_same_item": (query_dict["fname"][i].split("/")[-2] ==
                                                  gallery_dict["fname"][j].split("/")[-2]),
                                 "distance": np.sum(np.square(query_dict["out"][i] - gallery_dict["out"][j]), axis=-1)})
            elif args.metric_type == "cosine":
                q_result.append({"is_same_cls": (np.argmax(query_dict["cls"][i]) == np.argmax(gallery_dict["cls"][j])),
                                 "is_same_item": (query_dict["fname"][i].split("/")[-2] ==
                                                  gallery_dict["fname"][j].split("/")[-2]),
                                 "distance": 1 - np.sum(query_dict["out"][i] * gallery_dict["out"][j], axis=-1)})
            else:
                raise Exception("Wrong metric type. Available: ['euclidean', 'cosine']")

        q_result = sorted(q_result, key=lambda r: r["distance"])

        results.append(q_result[:50])

    retr_acc_1 = eval_results(results, k=1)
    retr_acc_5 = eval_results(results, k=5)
    retr_acc_10 = eval_results(results, k=10)
    retr_acc_20 = eval_results(results, k=20)
    retr_acc_30 = eval_results(results, k=30)
    retr_acc_40 = eval_results(results, k=40)
    retr_acc_50 = eval_results(results)

    print("Testing is completed.\tTime Elapsed: {:5.2f}\n"
          "The retrieval accuracies:\n"
          "\tTop-1: {:2.2f}\n"
          "\tTop-5: {:2.2f}\n"
          "\tTop-10: {:2.2f}\n"
          "\tTop-20: {:2.2f}\n"
          "\tTop-30: {:2.2f}\n"
          "\tTop-40: {:2.2f}\n"
          "\tTop-50: {:2.2f}\n".format(time.time() - t_start, retr_acc_1, retr_acc_5,  retr_acc_10,
                                       retr_acc_20, retr_acc_30, retr_acc_40, retr_acc_50))

    # TODO
    # # Reconstruct batch of images
    # if args.recon:
    #   x_test_batch, y_test_batch = get_iterator(args.filepath, subset="test").next()
    #   y_pred, x_recon = model.predict(x_test_batch)
    #
    #   # Save reconstructed and original images
    #   save_recons(x_recon, x_test_batch, y_pred, y_test_batch, args.save_dir)


def extract_embeddings(model, args, subset="query"):
    print("Extracting features for each image in {} set...".format(subset))
    data_gen = ImageDataGenerator(rescale=1/255.)

    data_iterator = data_gen.flow_from_directory(directory=os.path.join(args.filepath, subset),
                                                 batch_size=args.batch_size,
                                                 shuffle=False)

    if args.category != -1:
        print(list(data_iterator.class_indices.keys())[list(data_iterator.class_indices.values()).index(args.category)])

    for i in tqdm(range(len(data_iterator)), ncols=100, desc=subset,
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        xs, ys = next(data_iterator)

        y_pred = model.predict([xs, ys])

        if i > 0:
            embedings = np.vstack((embedings, y_pred))
            clss = np.vstack((clss, ys))
        else:
            embedings = np.array(y_pred)
            clss = np.array(ys)

    return {"out": embedings, "cls": clss, "fname": data_iterator.filenames}


def eval_results(x, k=50):
    retrievals = list()
    for result in x:
        retrieved = False
        for r in result[:k]:
            if r["is_same_item"]:
                retrieved = True
                break
        retrievals.append(retrieved)

    return 100 * np.mean(retrievals)


if __name__ == '__main__':
    K.clear_session()
    args = get_arguments()
    print(args)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    model, eval_model = FashionTripletCapsNet(input_shape=(args.input_size, args.input_size, 3), args=args)

    if args.weights is not None:
        model.load_weights(args.weights)
        eval_model.load_weights(args.weights)

    if args.multi_gpu and args.multi_gpu >= 2:
        p_model = MultiGPUNet(model, args.multi_gpu)
        p_eval_model = MultiGPUNet(eval_model, args.multi_gpu)

    if not args.testing:
        model.summary()
        if args.multi_gpu and args.multi_gpu >= 2:
            train(model=p_model, eval_model=p_eval_model, args=args)
            # implicitly sure that p_model defined
        else:
            train(model=model, eval_model=eval_model, args=args)
    else:
        eval_model.summary()
        if args.weights is None:
            print('Random initialization of weights.')
        test(model=p_eval_model, args=args)
