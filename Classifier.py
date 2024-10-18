import torch.nn as nn
import torch
import random
import os
import numpy as np
import torch.nn.functional as F
from utils_DTC import compute_similarity_3d
from config import get_arguments
from TAE import  TAE
from Cluster_layer import ClusterNet
from load_data_np import get_loader_np_train_test
from utils_DTC import print_cluster_info,print_error_info,compute_class_specific_accuracy,print_f1_f1pa,evaluation_clu_metric,get_logger
import sys
from algorithm import CDCC

class Input(nn.Module):
    """
    input_embeding_units: numbers of input embeding 
    inchannel：numbers of input feature
    """

    def __init__(self, input_embeding_units, inchannels):
        super().__init__()
        self.input_embeding_units = input_embeding_units
        self.in_channels = inchannels
        self.input_embeding = nn.Linear(self.in_channels,self.input_embeding_units)
    def forward(self, x):
        x = self.input_embeding(x)
        return x

class Output(nn.Module):
    """
    input_embeding_units: numbers of input embeding 
    inchannel：numbers of input feature
    """

    def __init__(self, input_embeding_units, inchannels):
        super().__init__()
        self.input_embeding_units = input_embeding_units
        self.in_channels = inchannels
        self.output_embeding = nn.Linear(self.input_embeding_units,self.in_channels)
    def forward(self, x):
        x = self.output_embeding(x)
        return x

class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Input embeding、Output embeding
        self.Input_embeding_list = nn.ModuleList()
        self.Output_embeding_list = nn.ModuleList()
        for i in range(args.dataset_num):
            self.Input_embeding_list.append(Input(args.input_embeding_units,args.inchannels[i]))
            self.Output_embeding_list.append(Output(args.input_embeding_units,args.inchannels[i]))
            
        self.input_dim = args.input_embeding_units*args.timesteps
        self.hidden_dim = args.classfier_hidden_dim
        self.output_dim = 1

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, datasetid):
        x = self.Input_embeding_list[datasetid](x)
        x = x.reshape(x.shape[0],-1)

        # print(x.shape)
        # sys.exit()

        x = F.relu(self.fc1(x))


        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def get_data(args, dataset_name, channels):
    return get_loader_np_train_test(args, dataset_name, True, [args.window_size, channels], 55, anomaly=True)

def train(model, trainloader_list, epochs, optimizer, args):
    loss_ce = nn.BCELoss()
    for epoch in range(epochs):
        class_pred_pro_list = [] #list(class_pred_pro of per dataset)
        class_label_list = [] #list(class_label of per dataset)
        for i in range(len(trainloader_list)):
            trainloader = trainloader_list[i]
            train_loss = 0
            all_class_pred_pro = []
            all_class_label = []
            for batch_idx, (inputs, labels, anomaly_labels, seq_anomaly_labels) in enumerate(trainloader):
                #输入，分类label，异常类型label（0~n），seq级别异常label（0，1）
                inputs = inputs.type(torch.FloatTensor).to(args.device)
                class_label = np.ones(anomaly_labels.shape[0])
                class_label[anomaly_labels == 0] = 0
                class_label = torch.tensor(class_label).float().unsqueeze(1).to(args.device)
                optimizer.zero_grad()
                # Q,P only compute for normal sample
                class_pred_pro = model(inputs, i)
                # classification loss
                loss_cls = loss_ce(class_pred_pro, class_label) 
                total_loss = loss_cls
                total_loss.backward()
                optimizer.step()
                all_class_pred_pro.extend(class_pred_pro.cpu().detach().numpy())
                all_class_label.extend(class_label.cpu().detach().numpy())
                train_loss += total_loss.item()
            print("For epoch：{} dataset：NO.{} Loss is : {} ".format(
                        epoch,
                        i,
                        train_loss / (batch_idx + 1),)
                )
            all_class_pred_pro = np.array(all_class_pred_pro)
            all_class_label = np.array(all_class_label)
            class_pred_pro_list.append(all_class_pred_pro)
            class_label_list.append(all_class_label)    
    #返回最后一个epoch的结果
    return (class_pred_pro_list, class_label_list)


def test(model, testloader_list, args):
    class_pred_pro_list = [] #list(class_pred_pro of per dataset)
    class_label_list = [] #list(class_label of per dataset)
    # for i in range(len(trainloader_list)):
    for i in range(len(testloader_list)):
        trainloader = testloader_list[i]   
        # trainloader = trainloader_list[i]
        train_loss = 0
        all_class_pred_pro = []
        all_class_label = []
        for batch_idx, (inputs, labels, anomaly_labels, seq_anomaly_labels) in enumerate(trainloader):
            #输入，分类label，异常类型label（0~n），seq级别异常label（0，1）
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            class_label = np.ones(anomaly_labels.shape[0])
            class_label[anomaly_labels == 0] = 0
            class_label = torch.tensor(class_label).float().unsqueeze(1).to(args.device)

            class_pred_pro = model(inputs, i)


            all_class_pred_pro.extend(class_pred_pro.cpu().detach().numpy())
            all_class_label.extend(class_label.cpu().detach().numpy())
        all_class_pred_pro = np.array(all_class_pred_pro)
        all_class_label = np.array(all_class_label)
        class_pred_pro_list.append(all_class_pred_pro)
        class_label_list.append(all_class_label)    
    #返回最后一个epoch的结果
    return (class_pred_pro_list, class_label_list)



if __name__ == "__main__":
    '''set args and datasets'''
    set_seed(42)
    from config import get_arguments
    parser = get_arguments()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.batch_size = 64
    args.drop_last = True
    args.window_size = 100

    '''get data'''
    datasets = ['SWaT'] #, 'WADI']#, 'MSL', 'PSM', 'SMAP', 'SMD']
    channels = [51]#, 123]#, 55, 25, 25, 38]

    data_list = [get_data(args, dataset, channels[i]) for i, dataset in enumerate(datasets)]

    trainloader_list = []
    testloader_list = []
    channel_list = []
    X_scaled_list = []
    anomaly_label_train_list = []
    anomaly_label_test_list = []

    for i, (trainloader, X_scaled, anomaly_label_train, testloader, X_test, anomaly_label_test) in enumerate(data_list):
        trainloader_list.append(trainloader)
        testloader_list.append(testloader)
        channel_list.append(X_scaled.shape[2])
        X_scaled_list.append(X_scaled)
        anomaly_label_train_list.append(anomaly_label_train)
        anomaly_label_test_list.append(anomaly_label_test)

    args.n_clusters = 14
    args.inchannels = channel_list
    args.timesteps = X_scaled_list[0].shape[1]
    args.dataset_num = len(trainloader_list)

    model = Classifier(args)
    model.to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr_cluster, momentum=args.momentum
    )

    '''train'''
    class_pred_pro_list, class_label_list = train(model, trainloader_list, 30, optimizer, args)
    compute_class_specific_accuracy(class_pred_pro_list, class_label_list)

    '''test'''
    class_pred_pro_list_test, class_label_list_test = test(model, testloader_list, args)
    compute_class_specific_accuracy(class_pred_pro_list_test, class_label_list_test)