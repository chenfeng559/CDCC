import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument(
        "--dataset_name", default="SAD+NATOPS", help="dataset name"
    )
    parser.add_argument(
        "--path_weights",
        default="model_weights_DTCR/{}/",
        help="models weights",
    )
    parser.add_argument(
        "--dataset_num",
        required=False,
        default=1,
        help="dataset num",
    )
    parser.add_argument(
        "--cluster_num",
        required=False,
        default=2,
        help="cluster num",
    )
    parser.add_argument(
        "--batch_first",
        required=False,
        default=True,
        help="input form",
    )
    parser.add_argument(
        "--drop_last",
        required=False,
        default=True,
        help="if loader drop last", #必须是True，因为DTCR的聚类矩阵F,是根据batch_size初始化的。
    )

    # anomaly inject args
    parser.add_argument(
        "--ano_sample_rate",
        required=False,
        default=1,
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
        help="max time step inject rate",
    )
    parser.add_argument(
        "--ano_time_rate_min",
        required=False,
        default=0.5,
        help="min time step inject rate",
    )


    # model args
    parser.add_argument(
        "--hidden_structs",
        required=False,
        default=[100,50,50],
        help="dilated rnn hidden dims",
    )
    parser.add_argument(
        "--dilations",
        required=False,
        default=[1,2,4],
        help="dilation of per dilated rnn",
    )
    parser.add_argument(
        "--inchannels",
        required=False,
        default=[1],
        help="number of per dataset feature num",
    )
    parser.add_argument(
        "--cell_type",
        required=False,
        default='GRU',
        help="rnn cell type",
    )

    #!!!!!!!!!!!!!!!!!!!!!!!! multi dataset parameter
    parser.add_argument(
        "--input_embeding_units",
        required=False,
        default=64,
        help="hidden units of input/output mebeding layer",
    )
    parser.add_argument(
        "--gamma1",
        type=float,
        default=(1e-1)/2,
        help="weight of kmeans loss",
    )
    parser.add_argument(
        "--gamma2",
        type=float,
        default=0,
        help="weight of l2 regularization",
    )
    parser.add_argument(
        "--gamma3",
        type=float,
        default=1,
        help="weight of cls loss",
    )


    # training args
    parser.add_argument(
        "--device",
        default="cpu"
    )
    parser.add_argument(
        "--train_decoder",
        default=True,
        help="train decoder or test decoder; decide the decoder input"
    )
    parser.add_argument(
        "--batch_size", 
        default=16, 
        help="batch size"
        )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum epochs numer of the  model training",
    )
    parser.add_argument(
        "--max_patience",
        type=int,
        default=5,
        help="The maximum patience for DTCR training , above which we stop training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate of the model training",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum for the full model training",
    )

    return parser
