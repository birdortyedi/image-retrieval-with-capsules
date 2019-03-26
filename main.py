import os
from keras import optimizers, callbacks
from keras import backend as K
from config import get_arguments
from models import FashionSiameseCapsNet, MultiGPUNet
from utils import custom_generator, get_iterator, contrasive_margin_loss, siamese_accuracy


def train(model, args):
    # Define callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv', append=True)
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                           monitor='val_capsnet_siamese_accuracy', save_best_only=True,
                                           save_weights_only=True, verbose=args.verbose)
    early_stopper = callbacks.EarlyStopping(monitor='val_capsnet_loss', patience=args.patience, verbose=args.verbose)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr, amsgrad=True),
                  loss=[contrasive_margin_loss, 'mse', 'mse'],
                  loss_weights=[1., args.lam_recon, args.lam_recon],
                  metrics={'capsnet': [siamese_accuracy]})

    # Start training using custom generator
    model.fit_generator(generator=custom_generator(get_iterator(args.filepath,
                                                                args.input_size,
                                                                args.shift_fraction,
                                                                args.hor_flip,
                                                                args.whitening,
                                                                args.rotation_range,
                                                                args.brightness_range,
                                                                args.shear_range,
                                                                args.zoom_range,
                                                                subset="train"),
                                                   testing=args.testing),
                        steps_per_epoch=int(25882 / args.batch_size),
                        epochs=args.epochs,
                        validation_data=custom_generator(get_iterator(args.filepath, args.input_size,
                                                                      subset="gallery"),
                                                         testing=args.testing),
                        validation_steps=int(12612 / args.batch_size),
                        callbacks=[log, tb, checkpoint, lr_decay, early_stopper],
                        initial_epoch=args.initial_epoch)

    # Save the model
    model_path = '/t_model.h5'
    model.save(args.save_dir + model_path)
    print('The model saved to \'%s' + model_path + '\'' % args.save_dir)


def test(model, args):
    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[contrasive_margin_loss],
                  # loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    # Evaluate the model using custom generator
    scores = model.evaluate_generator(generator=custom_generator(get_iterator(args.filepath,
                                                                              subset="test")),
                                      steps=int(40000 / args.batch_size))
    print(scores)

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
