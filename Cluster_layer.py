import torch.nn as nn
import torch
from utils_DTC import compute_similarity_3d
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering,KMeans
from TAE import TAE
import os
import numpy as np

class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = args.n_hidden*args.n_units[1]
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

'''DTC Model'''
class ClusterNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        ## init with the pretrained autoencoder model
        self.tae = TAE(args)
        self.tae.load_state_dict(torch.load(args.path_weights_ae, map_location=args.device))
        self.tae.to(args.device)
        ## clustering model
        self.alpha_ = args.alpha
        self.centr_size = int(args.timesteps/args.pool_size)
        self.n_clusters = args.n_clusters
        self.device = args.device
        self.similarity = args.similarity
        self.centroids = nn.Parameter(torch.randn((self.n_clusters,self.centr_size,args.n_units[1])))
        ## classfier 
        self.classfier = Classifier(args)
        self.classfier.to(args.device)

    def init_centroids(self, x_list, anomaly_list):
        """This function initializes centroids with kmeans /agglomerative clustering+ complete linkage 
        Args:
            x_list ( list(tensor) ): list(per dataset all samples)
            anomaly_list ( list(tensor) ): list(per dataset anomaly label)
        """        
        ##get hidden 
        for i in range(len(x_list)):
            z, _ = self.tae(x_list[i].detach(), i)#batch,seq,feature)
            #只用正常样本初始化聚类中心
            nomaly_index = np.where(anomaly_list[i] == 0)
            z = z[nomaly_index]
            if i == 0:
                z_np = z.detach().cpu()
                z_total = z
            else :
                z_np = torch.cat((z_np,z.detach().cpu()),0) # cat every dataset data
                z_total = torch.cat((z_total,z),0)
        '''
        ##clustering 使用agglomerative clustering + complete linkage 
        """
        assignements = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage="complete", affinity="precomputed"
        ).fit_predict(
            compute_similarity(z_np, z_np, similarity=self.similarity)
        )
        ##get centroids
        centroids_ = torch.zeros(
            (self.n_clusters, self.centr_size), device=self.device
        )
        for cluster_ in range(self.n_clusters):
            index_cluster = [
                k for k, index in enumerate(assignements) if index == cluster_
            ]
            centroids_[cluster_] = torch.mean(z_total.detach()[index_cluster], dim=0) 
        '''
        # 使用kmeans初始化聚类中心
        km = KMeans(n_clusters=self.n_clusters, n_init=10).fit(z_np.reshape(z_np.shape[0], -1))
        # compute centroid
        centroids_ = km.cluster_centers_.reshape(self.n_clusters, z_np.shape[1], z_np.shape[2])
        centroids_ = torch.tensor(centroids_)
        print('centroid_shape:{}'.format(centroids_.shape))
        self.centroids = nn.Parameter(centroids_)

    def forward(self, x, datasetid, nomaly_indices):
        """
        Args:
            x (tensor): inputs tensor, shape[B, L, D]
            datasetid (int): id number of dataset
            normaly_indices (tensor): index of normal sample
        Returns:
            _type_: _description_
        """        
        z, x_reconstr = self.tae(x, datasetid)
        z_nomaly = z[nomaly_indices]

        ## only clusterize z_nomaly
        similarity = compute_similarity_3d(
            z_nomaly, self.centroids, similarity=self.similarity
        )
        ## Q (batch_size , n_clusters)
        Q = 1.0 / (1.0 + (similarity ** 2) / self.alpha_)
        Q = Q.pow((self.alpha_ + 1.0) / 2.0)
        sum_rows_Q = Q.sum(dim=1, keepdim=True) 
        Q = Q / sum_rows_Q
        ## P : ground truth distribution
        P = Q ** 2 / Q.sum(0)
        P = (P.T / P.sum(1)).T

        ## classification
        class_pred_pro = self.classfier(z.reshape(z.shape[0], -1))

        return z, x_reconstr, Q, P, class_pred_pro
'''
    def update_centroids_manual(self, x_list):
        ##get hidden 
        for i in range(len(x_list)):
            z, _ = self.tae(x_list[i].detach(), i)#batch,seq,feature)
            if i == 0:
                z_np = z.detach().cpu()
            else :
                z_np = torch.cat((z_np,z.detach().cpu()),0) # cat every dataset data
        ##clustering
        assignements = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage="complete", affinity="precomputed"
        ).fit_predict(
            compute_similarity(z_np, z_np, similarity=self.similarity)
        )
        ##get centroids
        centroids_ = torch.zeros(
            (self.n_clusters, self.centr_size), device=self.device
        )
        for cluster_ in range(self.n_clusters):
            index_cluster = [
                k for k, index in enumerate(assignements) if index == cluster_
            ]
            centroids_[cluster_] = torch.mean(z_np.detach()[index_cluster], dim=0) #原来用的z，这改成z_np了，应该没什么影响
        self.centroids = nn.Parameter(centroids_)
'''

if __name__ == "__main__":
    print('test')