import math
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from algorithm.CDCC.CDCC import CDCC
from algorithm.CDCC.dataset import Load_Dataset, MyDataset
from torch.utils.data import DataLoader
import sys
from algorithm.CDCC import contrastive_loss

class model():
    def __init__(self):
        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 0.0003
        self.weight_decay = 0.000000001

        # ------freq_encoder parameters------
        self.input_channels = 1  # The number of input channels of the convolutional network with a UTS of 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 64  # The number of convolutional network output channels
        self.num_classes = None
        self.dropout = 0.30

        # --------------------------------
        self.epochs = 300
        self.model_save_path = None
        self.tensorboard_path=None

        # contrastive_loss parameters
        self.instance_temperature = 0.5
        self.cluster_temperature = 1.0
        self.lam = 0.5  # Loss function coefficient

        # device parameters
        self.device = 'cuda'

        # DataLoader parameters
        self.batch_size = 128
        self.drop_last = True
        self.num_workers = 0

        # Time augmentations parameters
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8

        # Frequency augmentations parameters
        self.remove_frequency_ratio = 0.1
        self.add_frequency_ratio = 0.1

        # Parameters for the instance-level and cluster-level mapping networks
        self.CNNoutput_channel = None
        self.feature_dim = 256
        self.hidden_size = 1024
        self.output_size = 512

        self.dropout_rate = 0.10
        self.num_layers = 2  # The number of layers of BiLSTM

    def step_epoch(self, optimizer, dataset, criterion_instance, criterion_cluster, epoch):  # 每一次迭代
        loss_epoch = 0
        total_loss=[]
        for step, (x_data, y_data, aug1, aug2, x_data_f, aug1_f, aug2_f) in enumerate(dataset):
            optimizer.zero_grad()

            x_data = x_data.to(torch.float32).to(self.device)
            x_data_f = x_data_f.to(torch.float32).to(self.device)

            aug1 = aug1.to(torch.float32).to(self.device)
            aug1_f = aug1_f.to(torch.float32).to(self.device)

            aug2 = aug2.to(torch.float32).to(self.device)
            aug2_f = aug2_f.to(torch.float32).to(self.device)

            """Representation"""
            h_t, z_i_t, z_c_t, h_t_aug, z_i_t_aug, z_c_t_aug = self.model(aug1, aug2,'t')
            h_f, z_i_f, z_c_f, h_f_aug, z_i_f_aug, z_c_f_aug = self.model(aug1_f, aug2_f, 'f')

            #Time domain contrastive constraints
            loss_i_t = criterion_instance(z_i_t, z_i_t_aug)
            loss_c_t = criterion_cluster(z_c_t, z_c_t_aug)
            loss_t = loss_i_t + loss_c_t

            #Frequency domain contrastive constraints
            loss_i_f=criterion_instance(z_i_f, z_i_f_aug)
            loss_c_f = criterion_cluster(z_c_f, z_c_f_aug)
            loss_f = loss_i_f + loss_c_f

            #Cross-domain contrastive constraints
            loss_i_t_f = criterion_instance(z_i_t_aug, z_i_f_aug)
            loss_c_t_f = criterion_cluster(z_c_t_aug, z_c_f_aug)
            loss_tf =  loss_i_t_f + loss_c_t_f

            #Loss Function
            loss = self.lam*(loss_t + loss_f )+ (1-self.lam) * loss_tf
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            loss_epoch += loss.item()
        total_loss = torch.tensor(total_loss).mean()
        return total_loss.item()

    def train(self, ds, valid_ds = None, valid_func=None, cb_progress=lambda x:None):
        #Make sure that the dimensions of your data are [num_instance,in_channel,series_length]
        # self.class_num=len(np.unique(ds[1]))

        # 无标签数据自定义聚类个数
        self.class_num = 6
        print("self.class_num:",self.class_num)
        self.input_channels = ds[0].shape[1]
        self.input_size = ds[0].shape[2]

        trainset=Load_Dataset(self, ds)
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.batch_size, shuffle=True,
                                             num_workers=self.num_workers, drop_last=self.drop_last)
        
        test_set = MyDataset(ds)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False,
                                                   num_workers=self.num_workers, drop_last=False)
        # device变为gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CDCC(self).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2),
                                           weight_decay=self.weight_decay)
        criterion_instance = contrastive_loss.InstanceLoss(self.batch_size,
                                                           self.instance_temperature,
                                                           self.device).to(self.device)
        criterion_cluster = contrastive_loss.ClusterLoss(self.class_num,
                                                         self.cluster_temperature,
                                                         self.device).to(self.device)
        max_result = 0
        print("### Start Training")
        for epoch in range(1,self.epochs+1):
            print(f"current epoch: ",epoch)
            self.model.train()
            loss_epoch = self.step_epoch(optimizer, train_loader, criterion_instance, criterion_cluster, epoch)
            predict_labels, true_label = self.predict_epoch(test_loader)

            # 转成list计算聚类metric,新代码，意义不明
            pred_list, z_list = [], []
            z_list.append(predict_labels)
            pred_list.append(true_label)

            #Adjust the learning rate
            adjust_learning_rate(optimizer, self.lr, epoch, self.epochs)
            # result = torch.subtract(torch.trace(WTW), torch.trace(FTWTWF))
            # Silhouette系数值越大，说明聚类效果越好
            result=[e(z_list, pred_list) for e in valid_func]
            valid_f=[str(v) for v in valid_func]
            if max_result<result[0]:
                max_result = result[0]
            # if max_result<result[1]:
            #     max_result = result[1]
                # save model
                torch.save(self.model, self.model_save_path)
            if epoch%1==0:
                print(epoch, "/", self.epochs, "\t loss:", loss_epoch)
                print(valid_f)
                print(result)

        cb_progress({
                'epoch': epoch,
                'total_epochs': self.epochs,
                'loss': loss_epoch,
                'metrics': result,
                'max_result': max_result
        })
        self.pred_labels = predict_labels
        return train_loader

    def predict_epoch(self, test_loader):
        self.model.eval()
        feature_vector = []
        labels_vector = []
        for step, (x_data, y_data) in enumerate(test_loader):
            x = x_data.to(torch.float32).to(self.device)
            with torch.no_grad():
                # c, _, _ = self.model.forward_cluster(x)
                c, h_time, z_time = self.model.forward_cluster(x)
            # c = c.detach()
            c = z_time.detach()
            h = h_time.detach()
            # feature_vector.extend(c.cpu().detach().numpy())
            # labels_vector.extend(y_data)
            feature_vector.extend(h.cpu().detach().numpy())
            labels_vector.extend(c.cpu().detach().numpy())
        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector)

        # 归一化
        sum_rows = np.sum(labels_vector, axis=1)
        sum_rows = sum_rows[:, np.newaxis]
        labels_vector = labels_vector / sum_rows

        return feature_vector, labels_vector

    def predict(self, ds, cb_progress=lambda x:None):
        return self.pred_labels #Take the clustering results of the last epoch

def adjust_learning_rate(optimizer, lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr