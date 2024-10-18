import numpy as np
from sklearn.preprocessing import LabelEncoder
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import train_test_split

from tslearn.datasets import UCR_UEA_datasets
from tslearn.datasets import CachedDatasets
import torch
from torch.utils.data import Dataset, DataLoader, dataloader
import pandas as pd
from scipy.io import arff
from inject import InjectAnomalies


def inject_anomaly(dataset: np.ndarray, y: np.ndarray, ano_sample_rate: float, ano_col_rate: float = 0.3,
                   ano_time_rate_max: float = 0.3, ano_type_num: int = 6, random_state=42):
    dataset_copy = dataset.copy()
    y_copy = y.copy()

    anomalies = ['peaks', 'time_noise',
                 'flip', 'amplitude_scaling',
                 'uniform_replacing', 'average']
    inj = InjectAnomalies(ano_col_rate=ano_col_rate, ano_time_rate_max=ano_time_rate_max, random_state=random_state)
    # 随机要注入异常的样本索引（不重复）,靠main函数中seed函数来固定随机性
    anomaly_index = np.random.choice(dataset.shape[0], size=int(ano_sample_rate * dataset.shape[0]), replace=False)
    # 存储异常标签，无异常0，1~6：peaks~average
    anomaly_label = np.zeros(dataset.shape[0], dtype=int)
    # 存储每个样本的每个时间步是否存在异常
    seq_anomaly_label = np.zeros((dataset.shape[0], dataset.shape[1]), dtype=int)
    # 均匀注入每种异常
    every_anomaly_type_num = len(anomaly_index) / ano_type_num
    anomaly_type_index = 0
    # 存储真正被注入异常的样本的index
    true_anomaly_index = []

    
    for i in range(len(anomaly_index)):
        index = anomaly_index[i]
        dataset[index], la = inj.inject_anomalies(dataset[index],
                                                  anomaly_type=anomalies[anomaly_type_index],
                                                  use_global_config=True)
        if 1 in la:
            anomaly_label[index] = anomaly_type_index + 1
            true_anomaly_index.append(index)
            seq_anomaly_label[index] = la
        if i % every_anomaly_type_num == 0 and i != 0:
            anomaly_type_index += 1
    # 将注入异常前的原始样本重新拼接回来
    dataset_selected = dataset_copy[true_anomaly_index]
    y_selected = y_copy[true_anomaly_index]
    dataset_final = np.concatenate((dataset, dataset_selected), axis=0)
    y_final = np.concatenate((y,y_selected), axis=0)
    anomaly_label_final = np.concatenate((anomaly_label,np.zeros(dataset_selected.shape[0])), axis=0)
    seq_anomaly_label_final = np.concatenate((seq_anomaly_label,np.zeros((dataset_selected.shape[0], dataset_selected.shape[1]), dtype=int)))
    return dataset_final, y_final, anomaly_label_final, seq_anomaly_label_final


# 读arrf文件，注意arrf文件的格式要符合要求
def read_arrf(file):
    with open(file) as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        df = pd.read_csv(f, header=None, na_values=['?'])
        df.columns = header
    return df


def load_arrf_data(dataset, scale, clamp_length=-1, pad_length=0):
    """
    Load classfication dataset.
    """
    np_data_list = []
    if dataset == 'SAD':
        for i in range(13):
            df = read_arrf('datasets/SpokenArabicDigits/SpokenArabicDigitsDimension{}_TRAIN.arff'.format(i + 1))
            df = df.sample(n=400, random_state=42)
            if i == 0:
                label = df.iloc[:, -1].to_numpy()
            df = df.iloc[:, :clamp_length]
            np_data_list.append(df.to_numpy())
    if dataset == 'NATOPS':
        for i in range(24):
            df = read_arrf('datasets/NATOPS/NATOPSDimension{}_TRAIN.arff'.format(i + 1))
            if i == 0:
                label = df.iloc[:, -1].to_numpy()
            df = df.iloc[:, :clamp_length]
            np_data_list.append(df.to_numpy())
    if dataset == 'NATOPS_Test':
        for i in range(24):
            df = read_arrf('datasets/NATOPS/NATOPSDimension{}_TEST.arff'.format(i + 1))
            if i == 0:
                label = df.iloc[:, -1].to_numpy()
            df = df.iloc[:, :clamp_length]
            np_data_list.append(df.to_numpy())
    if dataset == 'FM':
        for i in range(28):
            df = read_arrf('datasets/FingerMovements/FingerMovementsDimension{}_TRAIN.arff'.format(i + 1))
            if i == 0:
                label = df.iloc[:, -1].to_numpy()
            df = df.iloc[:, :clamp_length]
            np_data_list.append(df.to_numpy())
    if dataset == 'FD':
        df = read_arrf('datasets/FaceDetection/FaceDetection_TRAIN.arff')
        label = df.iloc[:, -1].to_numpy()
        df = df.iloc[:, :clamp_length]
        np_data_list.append(df.to_numpy())
    if dataset == 'AWR':
        for i in range(9):
            df = read_arrf(
                'datasets/ArticularyWordRecognition/ArticularyWordRecognitionDimension{}_TRAIN.arff'.format(i + 1))
            if i == 0:
                label = df.iloc[:, -1].to_numpy()
            df = df.iloc[:, :clamp_length]
            np_data_list.append(df.to_numpy())
    if dataset == 'Epilepsy':
        for i in range(3):
            df = read_arrf('datasets/Epilepsy/EpilepsyDimension{}_TRAIN.arff'.format(i + 1))
            if i == 0:
                label = df.iloc[:, -1].to_numpy()
            df = df.iloc[:, :clamp_length]
            np_data_list.append(df.to_numpy())
    if dataset == 'CT':
        for i in range(3):
            df = read_arrf('datasets/CharacterTrajectories/CharacterTrajectoriesDimension{}_TRAIN.arff'.format(i + 1))
            if i == 0:
                label = df.iloc[:, -1].to_numpy()
            df = df.iloc[:, :clamp_length]
            np_data_list.append(df.to_numpy())
    if dataset == 'Handwriting':
        for i in range(3):
            df = read_arrf('datasets/Handwriting/HandwritingDimension{}_TRAIN.arff'.format(i + 1))
            if i == 0:
                label = df.iloc[:, -1].to_numpy()
            df = df.iloc[:, :clamp_length]
            np_data_list.append(df.to_numpy())

    ## pading data
    if pad_length != 0:
        rows = np_data_list[0].shape[0]
        new_columns = np.zeros((rows, pad_length))
        for i in range(len(np_data_list)):
            origin_data = np_data_list[i]
            np_data_list[i] = np.concatenate((origin_data, new_columns), axis=1)

    X = np.dstack(np_data_list)
    X = np.nan_to_num(X, nan=0.0)
    le = LabelEncoder()
    y = le.fit_transform(label)

    ## scale
    if scale:
        X = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X, y


'''
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
'''


class CustomDataset_Anomaly(Dataset):
    def __init__(self, time_series, labels, anomaly, seq_anomaly):
        """
        This class creates a torch dataset.

        """
        self.time_series = time_series
        self.labels = labels
        self.anomaly = anomaly
        self.seq_anomaly = seq_anomaly

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        time_serie = torch.tensor(self.time_series[idx])
        label = torch.tensor(self.labels[idx])
        anomaly = torch.tensor(self.anomaly[idx])
        seq_anomaly = torch.tensor(self.seq_anomaly[idx])

        return (time_serie, label, anomaly, seq_anomaly)



def get_loader_arrf_train_test(args, dataset_name, sclae=True, clamp_length=-1, pad_length=0, label_modify=0, concat_test=False
                    , anomaly=False, ano_sample_rate=0.3, ano_type_num=6, test_size=0.3):
    X_scaled, y = load_arrf_data(dataset_name, scale=sclae, clamp_length=clamp_length, pad_length=pad_length)
    '''训练集、测试集一起用（数据不够时）'''
    if concat_test == True:
        X_scaled_test, y_test = load_arrf_data(dataset_name + '_Test', scale=sclae, clamp_length=clamp_length,
                                               pad_length=pad_length)
        X_scaled = np.concatenate((X_scaled, X_scaled_test), axis=0)
        y = np.concatenate((y, y_test), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    '''注入异常'''
    if anomaly == True:
        X_train, y_train, anomaly_label_train, seq_anomaly_label_train = inject_anomaly(X_train, y_train, ano_sample_rate=ano_sample_rate, ano_type_num=ano_type_num)
        X_test, y_test, anomaly_label_test, seq_anomaly_label_test = inject_anomaly(X_test, y_test, ano_sample_rate=ano_sample_rate, ano_type_num=ano_type_num)
    else:
        anomaly_label_train = np.zeros(X_train.shape[0], dtype=int)
        anomaly_label_test = np.zeros(X_test.shape[0], dtype=int)
        seq_anomaly_label_train = np.zeros((X_train.shape[0],X_train.shape[1]), dtype=int)
        seq_anomaly_label_test = np.zeros((X_test.shape[0],X_train.shape[1]), dtype=int)

    print('训练集数量：{}'.format(X_train.shape[0]))
    print('测试集数量：{}'.format(X_test.shape[0]))

    if label_modify != 0:
        y_train = y_train + label_modify
        y_test = y_test + label_modify
    # create dataset
    trainset = CustomDataset_Anomaly(X_train, y_train, anomaly_label_train, seq_anomaly_label_train)
    testset = CustomDataset_Anomaly(X_test, y_test, anomaly_label_test, seq_anomaly_label_test)
    ## create dataloader
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    return trainloader, X_train, anomaly_label_train, testloader, X_test, anomaly_label_test




if __name__ == "__main__":
    X, y = load_arrf_data('SAD', scale=True, clamp_length=-1)
    A_X, label,seq_label = inject_anomaly(X, ano_sample_rate=0.3, ano_type_num=6)
    print(A_X.shape)
    print(label.shape)
    print(seq_label.shape)
    print(np.count_nonzero(label))
    count=0
    for i in range(len(seq_label)):
        if np.count_nonzero(seq_label[i])>0:
            count+=1
    print(count)

    '''
    test_data1 = np.arange(80).reshape(16, 5).astype(float)
    test_data = np.stack((test_data1,test_data1,test_data1,test_data1,test_data1,test_data1),axis=0)
    A_X, index = inject_anomaly(test_data, ano_sample_rate=1, ano_type_num=6)
    print(index)
    '''
