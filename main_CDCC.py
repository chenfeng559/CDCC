import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import importlib
import io
import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
import numpy as np
import data_config
import scipy.optimize as opt
from sklearn.metrics import normalized_mutual_info_score, rand_score
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import yaml
import sys
import inspect
import argparse
import warnings
warnings.filterwarnings("ignore")
from load_data_np import split_sample ,split_csvEttm
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from utils_DTC import evaluation_clu_metric
from tqdm import tqdm
import time

def myloads(jstr):
    return yaml.safe_load(io.StringIO(jstr))


parser = argparse.ArgumentParser(description='model')

parser.add_argument('-f', dest='argFile', type=str, required=False,
                    default='config\CDCC.yaml',
                    help='Specify the test parameter file via the YAML file.')
# start_time= str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
parser.add_argument('-log', dest='log_path', type=str, required=False,
                    # default='',
                    help='Specify the path where the results are stored')
parser.add_argument('-m', dest='metrics', type=str, required=False,
                    # default="",
                    help='Verify the list of metrics, split by commas')
parser.add_argument('-a', dest='alg', type=str, required=False,
                    # default="",
                    help='algorithm')
parser.add_argument('-d', dest='data', type=str, required=False,
                    # default="data",
                    help='Data Directory')
parser.add_argument('-r', dest='params', type=myloads, required=False,
                    # default="{}",
                    help='''Algorithm parameters in JSON format, such as "{d:20,lr:0.1,n_itr:1000}" ''')
import os
import zipfile

import numpy as np
import pandas as pd

class MetricAbstract:
    def __init__(self):
        self.bigger= True
    def __str__(self):
        return self.__class__.__name__


    def __call__(self,groundtruth,pred ) ->float:
        raise Exception("Not callable for an abstract function")

class RI(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        return rand_score(groundtruth, pred)


class NMI(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        return normalized_mutual_info_score(groundtruth, pred)

class Silhouette(MetricAbstract):
    def __call__(self, h, pred) -> float:
        return evaluation_clu_metric('Silhouette Coefficient', h, pred)

def list_file_with_prefix(paths, prefix):
    result=[]
    for data_file in paths:
        s=data_file.split('/')[1]
        if s.startswith(prefix):
            result.append(data_file)
    return result

def parse_data(data_dir):
    if os.path.isfile(data_dir):
        print(f"Contents of data_dir: {data_dir}")
        z = zipfile.ZipFile(data_dir, mode='r')
        dir_list = z.namelist()
        print("Contents of zip file:", dir_list)

        path_train = list_file_with_prefix(dir_list, "TRAIN")
        path_test = list_file_with_prefix(dir_list, "TEST")
        print("Train path:", path_train)
        print("Test path:", path_test)
    else:
        print('data_dir should  be a zip file !')
    train_set = csv_to_X_y(path_train, z)
    test_set = csv_to_X_y(path_test, z)

    # Combine training set data and test set data
    X = np.concatenate((train_set[0], test_set[0]), axis=0)
    y = np.concatenate((train_set[1], test_set[1]), axis=0)

    return (X, y)

def read_np(file):
    data = np.load(file)
    return data

def load_npz_data(dataset:str, scale:bool, window_size:list, step:int=None):
    """
    Load real dataset.
    """
    array_x_list = []
    array_y_list = []

    if dataset == 'SWaT_train':
        array_x = read_np('datasets/SWaT/train.npy')
        print("raw dateser array_x.shape: ",{array_x.shape})
        array_y = np.zeros((array_x.shape[0], 1))
        print("raw dateser array_y.shape: ",{array_y.shape})
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
    
    
    # X = split_sample(array_x_list, window_size, step)
    # y = split_sample(array_y_list, [window_size[0], 1], step)
   
    X = split_sample(array_x_list, window_size, step)
    y = split_sample(array_y_list, [window_size[0], 1], step)
    
    print("after split X.Shape :",{X.shape})
    print("after split y.shape :",{y.shape})
    ## scale
    if scale:
        X = TimeSeriesScalerMeanVariance().fit_transform(X)
    X = np.transpose(X, (0,2,1))

    return X, y
    # return X

def parse_MTS_data(data_dir, scale, window_size, step):
    X, y = load_npz_data(data_dir, scale, window_size, step)
    print("after transpose X.shape :",{X.shape})
    return (X, y)


def csv_to_X_y(filepath, z):
    list_X = []
    y = None
    for path in filepath:
        dataframe = pd.read_csv(z.open(path), header=None)
        if path.endswith('label.csv'):
            y = np.squeeze(dataframe.values)
        else:
            list_X.append(np.expand_dims(dataframe.values, axis=-1))
    X = np.concatenate(list_X, axis=-1)

    assert (y is not None)
    assert (y.shape[0] == X.shape[0])
    X = np.transpose(X, (0, 2, 1))
    return X, y

def set_seed(seed=2333):
    import random, os, torch, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def update_parameters(param:dict, to_update:dict)->dict:
    for k, v in param.items():
        if k in to_update:
            if to_update[k] is not None:
                if isinstance(param[k],(dict,)):
                    param[k].update(to_update[k])
                else:
                    param[k]=to_update[k]
            to_update.pop(k)
    param.update(to_update)
    return param

def my_import(name):
    components = name.split('.')
    model_name = '.'.join(components[:-1])
    class_name = components[-1]
    mod = importlib.import_module(model_name)
    cls = getattr(mod, class_name)
    return cls

def create_obj_from_json(js):
    if isinstance(js,dict):
        assert len(js.items()) == 1
        for key,values in js.items():
            cls = my_import(key)()
            if isinstance(values, dict):
                for k, v in values.items():
                    setattr(cls, k, create_obj_from_json(v))
            elif values is None:
                pass
            else:
                raise Exception("Not valid parameters(Must be dict):",values)
            return cls
    elif isinstance(js,(set,list)):
        return [create_obj_from_json(x) for x in js]
    else:
        return js
    
def create_progress_callback(start_time, pbar):
    def update_progress(info):
        elapsed_time = time.time() - start_time
        pbar.set_postfix(
            loss=f"{info['loss']:.4f}",
            max_result=f"{info['max_result']:.4f}",
            elapsed=f"{elapsed_time:.2f}s"
        )
        pbar.update(1)
    return update_progress


def parse_csvdate(data_dir, test_size=0.2, window_size=[1, 10], step=1):
    if not os.path.isfile(data_dir):
        print('data_dir should be a zip file!')
        return None

    try:
        with zipfile.ZipFile(data_dir, 'r') as z:
            csv_file = z.namelist()[0]  # 假设只有一个 CSV 文件
            print(f"Reading CSV file: {csv_file}")
            with z.open(csv_file) as f:
                df = pd.read_csv(f, parse_dates=['date'])
    except Exception as e:
        print(f"Error reading ZIP file: {e}")
        return None

    print("Data loaded. Shape:", df.shape)

    # 设置 'date' 列为索引
    df.set_index('date', inplace=True)

    # 分离特征和目标变量
    feature_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    target_column = 'OT'

    # 标准化特征
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    # 准备特征和标签
    X = df[feature_columns].values
    y = df[target_column].values.reshape(-1, 1)

    # 使用 split_sample 函数创建滑动窗口
    print(f"X shape before split_sample: {X.shape}")
    # print(f"y shape before split_sample: {y.shape}")
    print("start spliting")
    X = split_csvEttm(X, window_size, step)
    # y = split_csvEttm(y, [window_size[0]], step)

    # 转置 X 以匹配之前的格式 (samples, features, time_steps)
    X = np.transpose(X, (0, 2, 1))
    # X_test = np.transpose(X_test, (0, 2, 1))
    # print(f"Train set shape: Y: {y.shape}")
    print(f"Train set shape: X: {X.shape}")
    # print(f"Test set shape: X: {X_test.shape}")

    return X,y

def main():
    #Read the model parameters from the configuration file
    args = parser.parse_args()
    if args.argFile is not None:
        with open(args.argFile) as infile:
            filedict = yaml.safe_load(infile)
    else:
        filedict = {}

    # Read the data set path
    data_dir = filedict['data_dir']
    # data = os.path.basename(data_dir)
    # data = os.path.splitext(data)[0]
    split_path = data_dir.split('/')
    data = split_path[-2]
    # data_dir_list = os.listdir(data_dir)
    # print(data_dir_list)
    args.data = data_dir
    arg_dict= {
            "algorithm": args.alg,
            "algorithm_parameters": args.params,
            "data_dir": args.data,
            "log_path": args.log_path}
    update_parameters(filedict, arg_dict)
    algorithm = create_obj_from_json({filedict['algorithm']: filedict['algorithm_parameters']})
    # print("Algorithm class source:", inspect.getsource(algorithm.__class__))
    # print("Algorithm class file:", inspect.getfile(algorithm.__class__))
    # print(dir(algorithm))
    # sys.exit()

    model_save = './model_save/' + data
    if not os.path.exists(model_save):
        os.makedirs(model_save)
    algorithm.model_save_path = model_save+'/'+ data +'.pt'
    # The output dimension of the convolutional layer
    algorithm.CNNoutput_channel = data_config.CNNoutput_channel[data]


    
    #loading the data 加载原本是npz格式的数据
    
    #loading the data
    data_dir = 'SWaT_train'
    window_size = filedict['window_size']
    step = window_size[0]
    print("loading data ...")
    ds = parse_MTS_data(data_dir=data_dir, scale=True, window_size=window_size, step=step)
    print("Dataset type:", type(ds))
    # sys.exit()

    """
    loading the data 加载csv格式的数据
    """
    # print("loading data ...")
    # data_dir = 'datasets\ETTm1\documents-export-2024-10-15.zip'
    # ds = parse_csvdate(data_dir)
    # sys.exit()
    print("Dataset type:", type(ds))
    print("Dataset length:", len(ds))
    if len(ds) > 0:
        print("First element type:", type(ds[0]))
        print("First element shape:", ds[0].shape)

    print("finish loading...")
    # Evaluation index
    # metrics = [NMI(), RI()]
    # sys.exit()
    metrics = [Silhouette()]
    print("strat training...")
    total_epochs = algorithm.epochs
    print("total epochs:", total_epochs)
    start_time = time.time()
    
    with tqdm(total=10, desc="Training Progress") as pbar:
            update_progress = create_progress_callback(start_time, pbar) 
            algorithm.train(ds, valid_ds=None, valid_func=metrics, cb_progress=update_progress)
            
    print("finish training...")
    sys.exit()
    pred = algorithm.predict(ds)
    true_label = np.array(ds[1])
    results = [m(true_label, pred) for m in metrics]
    metrics_name = [str(m) for m in metrics]
    print("RESULTS="+str(dict(zip(metrics_name, results))))


if __name__ == '__main__':
    main()
