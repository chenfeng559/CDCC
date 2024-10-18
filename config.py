import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument(
        "--dataset_name", default="SAD+NATOPS", help="dataset name"
    )
    parser.add_argument(
        "--path_weights",
        default="models_weights_DTC/{}/",
        help="models weights",
    )
    parser.add_argument("--path_data", default="data/{}", help="dataset name")
    parser.add_argument("--dataset_num", default=1, help="number of dataset")
    parser.add_argument("--inchannels", default=[15], help="feature_dim of dataset[i]")
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=2,
        help="Number of clusters , corresponding to the labels number",
    )
    parser.add_argument(
        "--timesteps",
        required=False,
        help="Number of timesteps (can be None for variable length sequences)",
    )

# anomaly inject args
    parser.add_argument(
        "--ano_sample_rate",
        required=False,
        default=0.5,
        help="sample inject rate",
    )
    parser.add_argument(
        "--ano_type_num",
        required=False,
        default=6,
        help="num of inject anomaly type",
    )
    parser.add_argument(
        "--ano_col_rate",
        required=False,
        default=0.3,
        help="feature col inject rate",
    )
    parser.add_argument(
        "--ano_time_rate_max",
        required=False,
        default=1,
        help="max time inject rate",
    )
    parser.add_argument(
        "--ano_time_rate_min",
        required=False,
        default=0.5,
        help="min time inject rate",
    )

    # model args
    parser.add_argument(
        "--pool_size",
        required=False,
        default=5,
        help="pooling hyperparameter. Refer to the paper for each dataset's corresponding value",
    )

    parser.add_argument(
        "--n_filters",
        required=False,
        default=50,
        help="number of filters in convolutional layer",
    )
    parser.add_argument(
        "--kernel_size",
        required=False,
        default=9,#15，20,
        help="size of kernel in convolutional layer",
    )
    parser.add_argument(
        "--strides",
        required=False,
        default=1,
        help="strides in convolutional layer",
    )
    parser.add_argument(
        "--n_units",
        required=False,
        default=[50, 1],
        help="numbers of units in the two BiLSTM layers",
    )
    parser.add_argument(
        "--similarity",
        required=False,
        choices=["COR", "EUC", "CID"],
        default="EUC",
        help="The similarity type",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=1,
        help="alpha hyperparameter for DTC model",
    )
    parser.add_argument(
        "--input_embeding_units",
        type=int,
        default=64,
        help="input_embeding_units",
    )
    parser.add_argument(
        "--gamma1",
        type=float,
        default=0.2,
        help="weight of kl loss",
    )
    parser.add_argument(
        "--gamma2",
        type=float,
        default=0,#(1e-1)/2,
        help="weight of l2 regularization",
    )
    parser.add_argument(
        "--gamma3",
        type=float,
        default=1,
        help="weight of cls loss",
    )
    parser.add_argument(
        "--classfier_hidden_dim",
        type=int,
        default=64,
        help="classfier_hidden_dim",
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
        "--window_size",
        type=int,
        default=100,
        help="Epochs number of the autoencoder training",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,#50,
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
    # parser.add_argument(
    #     '--ratio',
    #     default=[0.2,0.1,0.3,0.2,0.1,0.1],
    #     help='different ratio for injected anomalies',
    # )
    return parser
