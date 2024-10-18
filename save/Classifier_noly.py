import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import random
import numpy as np
from torch.utils.data import dataloader
from config import get_arguments
from TAE import  TAE
from load_data_new import get_loader_arrf_train_test
from utils import compute_classfication


class Classifier(nn.Module):
    """
    input_embeding_units: numbers of input embeding 
    inchannel：numbers of input feature
    """

    def __init__(self, args):
        super().__init__()
        self.input_dim = 55*24#args.n_hidden*args.n_units[1]#args.n_hidden*args.n_units[0]
        self.hidden_dim = args.classfier_hidden_dim
        self.output_dim = 1

        self.fc1 = nn.Linear(self.input_dim,self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self, x):
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

def train_classifier(model,trainloader,epochs,optimizer):
    loss_ce = nn.BCELoss()
    for epoch in range(epochs):
        all_class_pred_pro = []
        all_class_label = []
        train_cls_loss = 0
        for batch_idx, (inputs, labels, anomaly_labels, seq_anomaly_labels) in enumerate(trainloader):
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            class_label = np.ones(anomaly_labels.shape[0])
            class_label[anomaly_labels == 0] = 0
            class_label = torch.tensor(class_label).float().unsqueeze(1).to(args.device)

            optimizer.zero_grad()
            class_pred_pro = model(inputs.reshape(inputs.shape[0], -1))
            # classification loss
            loss_cls = loss_ce(class_pred_pro,class_label) 

            total_loss = loss_cls
            total_loss.backward()
            optimizer.step()
            all_class_pred_pro.extend(class_pred_pro.cpu().detach().numpy())
            all_class_label.extend(class_label.cpu().detach().numpy())
            train_cls_loss+=loss_cls.item()
        print(
            "For epoch：{} ".format(epoch),
            "cls loss:{}".format(train_cls_loss / (batch_idx + 1))
            )
    return all_class_pred_pro,all_class_label


if __name__ == "__main__":
    '''set args and datasets'''
    set_seed(42)
    from config import get_arguments
    parser = get_arguments()
    args = parser.parse_args()

    path_weights = args.path_weights.format(args.dataset_name)
    if not os.path.exists(path_weights):
        os.makedirs(path_weights)
    args.path_weights_ae = os.path.join(path_weights, "autoencoder_weight.pth")
    args.path_weights_main = os.path.join(path_weights, "full_model_weigths.pth")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.batch_size = 16
    args.drop_last = False


    trainloader, X_scaled, anomaly_label_train, testloader, X_test, anomaly_label_test = get_loader_arrf_train_test(args,'NATOPS',pad_length=4
                                                                                            ,concat_test=True,anomaly=True,test_size=0.3,ano_sample_rate=1)
    trainloader_2, X_scaled_2, anomaly_label_train_2, testloader_2, X_test_2, anomaly_label_test_2 = get_loader_arrf_train_test(args,'SAD',clamp_length=55
                                                                                            ,label_modify=6,anomaly=True,test_size=0.3,ano_sample_rate=1)

    trainloader_list = [trainloader, trainloader_2]
    testloader_list = [testloader, testloader_2]
    args.n_clusters = 13
    args.inchannels = [X_scaled.shape[2], X_scaled_2.shape[2]]# list(features of per dataset)
    args.timesteps = X_scaled.shape[1]
    args.dataset_num = len(trainloader_list)
    model = Classifier(args)
    model = model.to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr_cluster, momentum=args.momentum
    )
    all_class_pred_pro,all_class_label = train_classifier(model,trainloader_list[0],100,optimizer)
    print(compute_classfication([all_class_pred_pro],[all_class_label]))
    

