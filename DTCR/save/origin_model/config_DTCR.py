import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument(
        "--hidden_structs",
        required=False,
        default=[100,50,50],
        help="pooling hyperparameter. Refer to the paper for each dataset's corresponding value",
    )
    parser.add_argument(
        "--dilations",
        required=False,
        default=[1,2,4],
        help="number of timesteps (can be None for variable length sequences)",
    )
    parser.add_argument(
        "--input_dims",
        required=False,
        default=1,
        help="number of filters in convolutional layer",
    )
    parser.add_argument(
        "--cell_type",
        required=False,
        default='GRU',
        help="size of kernel in convolutional layer",
    )

    parser.add_argument(
        "--cluster_num",
        required=False,
        default=2,
        help="numbers of units in the two BiLSTM layers",
    )
    parser.add_argument(
        "--batch_first",
        required=False,
        default=True,
        help="The similarity type",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        help="models weights",
    )
    parser.add_argument(
        "--train_decoder",
        default=True,
        help="Number of clusters , corresponding to the labels number",
    )

    parser.add_argument(
        "--lambda1",
        type=float,
        default=0,
        help="weight of kl loss",
    )
    # training args
    parser.add_argument("--batch_size", default=256, help="batch size")
    parser.add_argument(
        "--epochs_ae",
        type=int,
        default=10,
        help="Epochs number of the autoencoder training",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,#50,
        help="Maximum epochs numer of the full model training",
    )

    parser.add_argument(
        "--max_patience",
        type=int,
        default=5,#10,
        help="The maximum patience for DTC training , above which we stop training.",
    )

    parser.add_argument(
        "--lr_ae",
        type=float,
        default=1e-2,
        help="Learning rate of the autoencoder training",
    )
    parser.add_argument(
        "--lr_cluster",
        type=float,
        default=1e-2,
        help="Learning rate of the full model training",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum for the full model training",
    )

    return parser
