from numbers import Rational
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.io import arff
from inject import InjectAnomalies

'''异常注入函数'''
def inject_anomaly(dataset: np.ndarray, y: np.ndarray, ano_sample_rate: float, ano_col_rate: float,
                     ano_time_rate_max: float, ano_time_rate_min:float, ano_type_num: int, random_state=42):
    """
    Args:
        dataset (np.ndarray): 
        y (np.ndarray): 
        ano_sample_rate (float): sample inject rate
        ano_col_rate (float): feature col inject rate
        ano_time_rate_max (float): max time step inject rate
        ano_time_rate_min (float): min time step inject rate
        ano_type_num (int): num of inject anomaly type
        random_state (int)
    Returns:
        _type_: dataset_final(异常数据+原数据), y_final(异常数据+原数据class label), 
        anomaly_label_final(每个样本的异常类型), seq_anomaly_label_final(每个样本窗口的时间步异常label)
    """    
    dataset_copy = dataset.copy()
    y_copy = y.copy()

    anomalies = ['peaks', #'soft_replacing', #,'time_noise'
                 'flip', 'amplitude_scaling',
                 'uniform_replacing', 'average']
    inj = InjectAnomalies(ano_col_rate=ano_col_rate, ano_time_rate_max=ano_time_rate_max, ano_time_rate_min=ano_time_rate_min, random_state=random_state)
    # 随机要注入异常的样本索引（不重复）,靠main函数中seed函数来固定随机性
    anomaly_index = np.random.choice(dataset.shape[0], size=int(ano_sample_rate * dataset.shape[0]), replace=False)
    # print(anomaly_index)
    # 存储异常标签，无异常0，1~6：peaks~average
    anomaly_label = np.zeros(dataset.shape[0], dtype=int)
    # 存储每个样本的每个时间步是否存在异常
    seq_anomaly_label = np.zeros((dataset.shape[0], dataset.shape[1]), dtype=int)

    # 按比例注入每种异常
    # ratio = [0.2, 0.1, 0.3, 0.2, 0.1]
    # every_anomaly_type_num = [int(len(anomaly_index) * r) for r in ratio]
    # current_count = 0

    every_anomaly_type_num = len(anomaly_index) // ano_type_num
    anomaly_type_index = 0
    # 存储真正被注入异常的样本的index
    true_anomaly_index = []
    # 每种异常均匀注入

    # for i in range(len(anomaly_index)):
    #     index = anomaly_index[i]
    #     if current_count >= every_anomaly_type_num[anomaly_type_index]:
    #         anomaly_type_index += 1
    #         current_count = 0
    #     if anomaly_type_index > 5:
    #         break
    #     dataset[index], la = inj.inject_anomalies(dataset[index],
    #                                               anomaly_type=anomalies[anomaly_type_index],
    #                                               use_global_config=True,rep_data=dataset[index])
    #     if 1 in la:
    #         anomaly_label[index] = anomaly_type_index + 1
    #         true_anomaly_index.append(index)
    #         seq_anomaly_label[index] = la
    #     current_count += 1


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
    y_final = np.concatenate((y, y_selected), axis=0)
    anomaly_label_final = np.concatenate((anomaly_label, np.zeros(dataset_selected.shape[0])), axis=0)
    seq_anomaly_label_final = np.concatenate((seq_anomaly_label, np.zeros((dataset_selected.shape[0], dataset_selected.shape[1]), dtype=int)))
    return dataset_final, y_final, anomaly_label_final, seq_anomaly_label_final

def read_np(file):
    data = np.load(file)
    return data

def split_sample(array_list, window_size, step=None):
     # 检查输入是否为列表或单个数组
    if isinstance(array_list, np.ndarray):
        array_list = [array_list]
    if not array_list:
        raise ValueError("Input array list is empty")
    
    win_rows = window_size[0]
    win_cols = window_size[1]
    # print("array list [0] : " ,{array_list[0])
    print("win_rows, win_cols: ", {win_rows}, {win_cols})
    samples = []
    for array in array_list:
        num_rows = array.shape[0]
        num_cols = array.shape[1]
        print(f"### num rows{num_rows}, num cols{num_cols}")
        #窗口遍历矩阵
        for row_start in range(0, num_rows - win_rows + 1, step or 1):
            for col_start in range(0, num_cols - win_cols + 1, 1):
                sample = array[row_start:row_start + win_rows, col_start:col_start + win_cols]
                # print("sample shape: ", sample.shape)
                samples.append(sample)
    return np.stack(samples)

def split_csvEttm(array, window_size, step=1):
    win_rows = window_size[0]
    win_cols = window_size[1]
    samples = []
    num_rows = array.shape[0]
    num_cols = array.shape[1]
    for row_start in range(0, num_rows - win_rows + 1, step or 1):
            for col_start in range(0, num_cols - win_cols + 1, 1):
                sample = array[row_start:row_start + win_rows, col_start:col_start + win_cols]
                samples.append(sample)
    return sample

def load_npz_data(dataset:str, scale:bool, window_size:list, step:int=None):
    """
    Load real dataset.
    """
    array_x_list = []
    array_y_list = []

    if dataset == 'SWaT_train':
        array_x = read_np('datasets/SWaT/train.npy')
        array_y = np.zeros((array_x.shape[0], 1))
        array_x_list.append(array_x)
        array_y_list.append(array_y)
    if dataset == 'SWaT_test':
        array = read_np('datasets/SWaT/test.npz')    
        array_x = array['x']
        array_y = array['y'].reshape(-1, 1)
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
    
    
    X = split_sample(array_x_list, window_size, step)
    y = split_sample(array_y_list, [window_size[0], 1], step)
    # X = split_sample(array_x_list, [window_size[0], 1], step)

    ## scale
    if scale:
        X = TimeSeriesScalerMeanVariance().fit_transform(X)

    # return X,y
    return X, y


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

class CDCCDataset(Dataset):
    def __init__(self, time_series):
        """
        This class creates a torch dataset.

        """
        self.time_series = time_series
        # self.labels = labels

    def __len__(self):
        # return len(self.labels)
        return 

    def __getitem__(self, idx):
        time_series = torch.tensor(self.time_series[idx])
        # label = torch.tensor(self.labels[idx])
        # anomaly = torch.tensor(self.anomaly[idx])
        # seq_anomaly = torch.tensor(self.seq_anomaly[idx])

        return time_series

def get_loader_np_train_test(args, dataset_name:str, scale:bool=True, window_size: list=[55, 1], step:int=1, anomaly=False):
    ano_sample_rate = args.ano_sample_rate
    ano_type_num = args.ano_type_num
    ano_col_rate = args.ano_col_rate
    ano_time_rate_max = args.ano_time_rate_max
    ano_time_rate_min = args.ano_time_rate_min
    # ratio = args.ratio
    X_train_scaled, y_train = load_npz_data(dataset_name+'_train', scale, window_size, step)
    X_test_scaled, y_test = load_npz_data(dataset_name+'_test', scale, window_size, step)
    '''注入异常'''
    if anomaly == True:
        X_train_scaled, y_train, anomaly_label_train, seq_anomaly_label_train = inject_anomaly(X_train_scaled, y_train, ano_sample_rate, ano_col_rate,
                                                                                ano_time_rate_max, ano_time_rate_min, ano_type_num)
    else:
        anomaly_label_train = np.zeros(X_train_scaled.shape[0], dtype=int)
        seq_anomaly_label_train = np.zeros((X_train_scaled.shape[0], X_train_scaled.shape[1]), dtype=int)
    
    anomaly_label_test = []
    for i in range(len(y_test)):
        if 1 in y_test[i]:
            anomaly_label_test.append(1)
        else:
            anomaly_label_test.append(0)
    anomaly_label_test = np.array(anomaly_label_test)
    seq_anomaly_label_test = y_test

    print('Dataset{}-train samples :{}'.format(dataset_name, X_train_scaled.shape))
    print('Dataset{}-test samples :{}'.format(dataset_name, X_test_scaled.shape))

    # trainset = CDCCDataset(X_train_scaled)
    # testset = CDCCDataset(X_test_scaled)
    
    # # create dataset
    trainset = CustomDataset_Anomaly(X_train_scaled, y_train, anomaly_label_train, seq_anomaly_label_train)
    testset = CustomDataset_Anomaly(X_test_scaled, y_test, anomaly_label_test, seq_anomaly_label_test)

    ## create dataloader
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=args.drop_last
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=args.drop_last
    )
    return trainloader, X_train_scaled, anomaly_label_train, testloader, X_test_scaled, anomaly_label_test
    # return trainloader, X_train_scaled, testloader, X_test_scaled



if __name__ == "__main__":
    print('test')


