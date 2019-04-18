import argparse


def get_arguments():
    # Define all hyper-parameters
    parser = argparse.ArgumentParser(description="FashionCapsNet")

    # INPUT & OUTPUT
    parser.add_argument('--filepath', default='./data/img/BOTH', type=str)
    parser.add_argument('--save_dir', default='./results')

    # MODEL ARCHITECTURE
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('-k', '--top_k', default=20, type=int)
    parser.add_argument('--conv_filters', default=256, type=int)
    parser.add_argument('--conv_kernel_size', default=9, type=int)
    parser.add_argument('--dim_capsule', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.995, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=10, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('--patience', default=20, type=int,
                        help="The number of patience epochs for early stopping")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing data set")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--multi_gpu', default=2, type=int,
                        help="The number of gpu available as >1, if =1, then leave default as None")
    parser.add_argument('--initial_epoch', default=0, type=int,
                        help="The initial epoch for beginning of the training")
    parser.add_argument('--recon', default=False, type=bool,
                        help="Saving the reconstructed images during testing")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--verbose', default=1, type=int,
                        help="Verbose or not")

    # DATA AUGMENTATION
    parser.add_argument('--shift_fraction', default=0.25, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--hor_flip', default=True, type=bool,
                        help="Flipping the images randomly on horizontal line.")
    parser.add_argument('--whitening', default=False, type=bool,
                        help="Applies ZCA Whitening randomly.")
    parser.add_argument('--rotation_range', default=30, type=int,
                        help="The range of rotation degree for the images.")
    parser.add_argument('--brightness_range', default=None, type=list,
                        help="The range of brightness degree for the images.")
    parser.add_argument('--shear_range', default=0., type=float,
                        help="Shear angle in counter-clockwise direction in degrees.")
    parser.add_argument('--zoom_range', default=0., type=float,
                        help="Range for random zoom for the images.")

    args = parser.parse_args()
    print(args)
    return args
