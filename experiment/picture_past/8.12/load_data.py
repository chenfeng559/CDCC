import numpy as np
from sklearn.preprocessing import LabelEncoder
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.datasets import UCR_UEA_datasets
from tslearn.datasets import CachedDatasets
import torch
from torch.utils.data import Dataset, DataLoader, dataloader


def load_ucr(dataset, scale):
    """
    Load ucr dataset.
    Taken from https://github.com/FlorentF9/DeepTemporalClustering/blob/4f70d6b24722bd9f8331502d9cae00d35686a4d2/datasets.py#L18
    """
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    if dataset == "HandMovementDirection":  # this one has special labels
        y = [yy[0] for yy in y]
    y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
    assert y.min() == 0  # assert labels are integers and start from 0
    # preprocess data (standardization)
    if scale:
        X = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X, y


def load_cac(dataset, scale):
    """
    Load cac dataset.
    """
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset(dataset)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
    assert y.min() == 0  # assert labels are integers and start from 0
    # preprocess data (standardization)
    if scale:
        X = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X, y


def load_data(dataset_name, all_datasets, data_type, scale=True):
    """
    args :
        dataset_name : a string representing the dataset name.
        all_ucr_datasets : a list of all ucr datasets present in tslearn UCR_UEA_datasets
        scale : a boolean that represents whether to scale the time series or not.
    return :
        X : time series , scaled or not.
        y : the labels ( binary in our case) . s
    """
    if dataset_name in all_datasets:
        if data_type=='ucr':
            return load_ucr(dataset_name, scale)
        else:
            return load_cac(dataset_name, scale)
    else:
        print(
            "Dataset {} not available! Available datasets are UCR/UEA univariate and multivariate datasets.".format(
                dataset_name
            )
        )
        exit(0)


class CustomDataset(Dataset):
    def __init__(self, time_series, labels):
        """
        This class creates a torch dataset.

        """
        self.time_series = time_series
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        time_serie = torch.tensor(self.time_series[idx])
        label = torch.tensor(self.labels[idx])

        return (time_serie, label)


def get_loader(args,data_type='cac'):

    if data_type == 'ucr':
        datasets = UCR_UEA_datasets()
    else:
        datasets = CachedDatasets()

    all_datasets = datasets.list_datasets()
    X_scaled, y = load_data(args.dataset_name, all_datasets, data_type)
    #print(len(np.unique(y)))
    # create dataset
    trainset = CustomDataset(X_scaled, y)

    ## create dataloader
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    return trainloader, X_scaled


if __name__ == "__main__":
    from config import get_arguments
    parser = get_arguments()
    args = parser.parse_args()
    args.dataset_name = 'Trace'
    args.batch_size = 20
    trainloader, X_scaled = get_loader(args)
    for data,label in trainloader:
        print(data.shape,label.shape)
    #print(len(trainloader))
    #print(X_scaled.shape)