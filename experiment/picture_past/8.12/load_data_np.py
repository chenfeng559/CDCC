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
                   ano_time_rate_max: float = 1,ano_time_rate_min:float=0.5, ano_type_num: int = 6, random_state=42):
    dataset_copy = dataset.copy()
    y_copy = y.copy()

    anomalies = ['peaks', 'time_noise', #,'soft_replacing'
                 'flip', 'amplitude_scaling',
                 'uniform_replacing', 'average']
    inj = InjectAnomalies(ano_col_rate=ano_col_rate, ano_time_rate_max=ano_time_rate_max, ano_time_rate_min=ano_time_rate_min, random_state=random_state)
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
                                                  use_global_config=True,rep_data=dataset[index])
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

def read_np(file):
    data = np.load(file)
    return data

def split_sample(array_list, window_size, step=None):
    win_rows = window_size[0]
    win_cols = window_size[1]
    samples = []
    for array in array_list:
        num_rows = array.shape[0]
        num_cols = array.shape[1]
        for row_start in range(0, num_rows - win_rows + 1, step or 1):
            for col_start in range(0, num_cols - win_cols + 1, step or 1):
                sample = array[row_start:row_start + win_rows, col_start:col_start + win_cols]
                samples.append(sample)
    return np.stack(samples)



def load_npz_data(dataset:str, scale:bool, window_size:list,step:int=None):
    """
    Load real dataset.
    """
    array_x_list = []
    array_y_list = []

    if dataset == 'SWaT_train':
        array_x = read_np('datasets/SWaT/train.npy')
        array_y = np.zeros((array_x.shape[0],1))
        array_x_list.append(array_x)
        array_y_list.append(array_y)
    if dataset == 'SWaT_test':
        array = read_np('datasets/SWaT/test.npz')    
        array_x = array['x']
        array_y = array['y'].reshape(-1,1)
        array_x_list.append(array_x)
        array_y_list.append(array_y)

    if dataset == 'WADI_train':
        array_x = read_np('datasets/WADI/train.npy')
        array_y = np.zeros((array_x.shape[0],1))
        array_x_list.append(array_x)
        array_y_list.append(array_y)
    if dataset == 'WADI_test':
        array = read_np('datasets/WADI/test.npz')
        array_x = array['x']
        array_y = array['y'].reshape(-1,1)
        array_x_list.append(array_x)
        array_y_list.append(array_y)

    if dataset == 'PSM_train':
        array_x = read_np('datasets/PSM/train.npy')
        array_y = np.zeros((array_x.shape[0],1))
        array_x_list.append(array_x)
        array_y_list.append(array_y)
    if dataset == 'PSM_test':
        array = read_np('datasets/PSM/test.npz')
        array_x = array['x']
        array_y = array['y'].reshape(-1,1)
        array_x_list.append(array_x)
        array_y_list.append(array_y)
   
    if dataset == 'MSL_train':
        for i in range(1,28):
            array_x = read_np('datasets/MSL/MSL_{}/train.npy'.format(i))
            array_y = np.zeros((array_x.shape[0],1))
            array_x_list.append(array_x)
            array_y_list.append(array_y)
    if dataset == 'MSL_test':
        for i in range(1,28):
            array = read_np('datasets/MSL/MSL_{}/test.npz'.format(i))
            array_x = array['x']
            array_y = array['y'].reshape(-1,1)
            array_x_list.append(array_x)
            array_y_list.append(array_y)

    if dataset == 'SMAP_train':
        for i in range(1,54):
            array_x = read_np('datasets/SMAP/SMAP_{}/train.npy'.format(i))
            array_y = np.zeros((array_x.shape[0],1))
            array_x_list.append(array_x)
            array_y_list.append(array_y)
    if dataset == 'SMAP_test':
        for i in range(1,54):
            array = read_np('datasets/SMAP/SMAP_{}/test.npz'.format(i))
            array_x = array['x']
            array_y = array['y'].reshape(-1,1)
            array_x_list.append(array_x)
            array_y_list.append(array_y)

    if dataset == 'SMD_train':
        for i in range(1,29):
            array_x = read_np('datasets/SMD/SMD_{}/train.npy'.format(i))
            array_y = np.zeros((array_x.shape[0],1))
            array_x_list.append(array_x)
            array_y_list.append(array_y)
    if dataset == 'SMD_test':
        for i in range(1,29):
            array = read_np('datasets/SMD/SMD_{}/test.npz'.format(i))
            array_x = array['x']
            array_y = array['y'].reshape(-1,1)
            array_x_list.append(array_x)
            array_y_list.append(array_y)
    
    
    X = split_sample(array_x_list,window_size,step)
    y = split_sample(array_y_list,[window_size[0],1],step)

    ## scale
    if scale:
        X = TimeSeriesScalerMeanVariance().fit_transform(X)

    return X,y


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

def get_loader_np_train_test(args, dataset_name:str, sclae:bool=True, window_size:list=[55,51], step:int=1, anomaly=False, ano_sample_rate=0.3, ano_type_num=6, ano_time_rate_max=1, ano_time_rate_min=0.5):
    X_train_scaled, y_train = load_npz_data(dataset_name+'_train',sclae,window_size,step)
    X_test_scaled, y_test = load_npz_data(dataset_name+'_test',sclae,window_size,step)
    '''注入异常'''
    if anomaly == True:
        X_train_scaled, y_train, anomaly_label_train, seq_anomaly_label_train = inject_anomaly(X_train_scaled, y_train, ano_sample_rate=ano_sample_rate, ano_type_num=ano_type_num,
                                                                                ano_time_rate_max=ano_time_rate_max ,ano_time_rate_min=ano_time_rate_min)
    else:
        anomaly_label_train = np.zeros(X_train_scaled.shape[0], dtype=int)
        seq_anomaly_label_train = np.zeros((X_train_scaled.shape[0],X_train_scaled.shape[1]), dtype=int)
    
    anomaly_label_test = []
    for  i in range(len(y_test)):
        if 1 in y_test[i]:
            anomaly_label_test.append(1)
        else:
            anomaly_label_test.append(0)
    anomaly_label_test = np.array(anomaly_label_test)
    seq_anomaly_label_test = y_test

    print('Dataset{}-train samples :{}'.format(dataset_name,X_train_scaled.shape))
    print('Dataset{}-test samples :{}'.format(dataset_name,X_test_scaled.shape))
    
    # create dataset
    trainset = CustomDataset_Anomaly(X_train_scaled, y_train, anomaly_label_train, seq_anomaly_label_train)
    testset = CustomDataset_Anomaly(X_test_scaled, y_test, anomaly_label_test, seq_anomaly_label_test)

    ## create dataloader
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2,drop_last=args.drop_last
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=True, num_workers=2,drop_last=args.drop_last
    )
    return trainloader, X_train_scaled, anomaly_label_train, testloader, X_test_scaled, anomaly_label_test



if __name__ == "__main__":
    '''
    from config import get_arguments
    parser = get_arguments()
    args = parser.parse_args()
    import os
    path_weights = args.path_weights.format(args.dataset_name)
    if not os.path.exists(path_weights):
        os.makedirs(path_weights)
    args.path_weights_ae = os.path.join(path_weights, "autoencoder_weight.pth")
    args.path_weights_main = os.path.join(path_weights, "full_model_weigths.pth")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.batch_size = 64
    args.drop_last = False
    trainloader, X_train_scaled, anomaly_label_train, testloader, X_test_scaled, anomaly_label_test = get_loader_np_train_test(args,'WADI',True,[55,123],55,anomaly=True)
    i=1
    for batch_idx, (inputs, labels, anomaly_labels, seq_anomaly_labels) in enumerate(trainloader):
               print(batch_idx)
    '''
    array_x_list=[]
    array_x = read_np('datasets/PSM/train.npy')
    array_x_list.append(array_x)
    print(np.concatenate((array_x_list),axis=0).shape) 




