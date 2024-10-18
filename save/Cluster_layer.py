import torch.nn as nn
import torch
from utils import compute_similarity_3d
from sklearn.cluster import AgglomerativeClustering,KMeans
from TAE import TAE
from load_data import get_loader
import os


class ClusterNet(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    n_clusters : num of clusters
    similarity : similarity_metric
    """

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

    def init_centroids(self, x_list):
        """
        This function initializes centroids with agglomerative clustering+ complete linkage/  kmeans
        """
        
        ##get hidden 
        for i in range(len(x_list)):
            z, _ = self.tae(x_list[i].detach(), i)#batch,seq,feature)
            if i == 0:
                z_np = z.detach().cpu()
                z_total = z
            else :
                z_np = torch.cat((z_np,z.detach().cpu()),0) # cat every dataset data
                z_total = torch.cat((z_total,z),0)
        '''
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
            centroids_[cluster_] = torch.mean(z_total.detach()[index_cluster], dim=0) 
        '''
        km = KMeans(n_clusters=self.n_clusters, n_init=10).fit(z_np.reshape(z_np.shape[0], -1))
        # compute centroid
        centroids_ = km.cluster_centers_.reshape(self.n_clusters, z_np.shape[1], z_np.shape[2])
        centroids_ = torch.tensor(centroids_)
        print('centroid_shape:{}'.format(centroids_.shape))
        self.centroids = nn.Parameter(centroids_)





    def forward(self, x, datasetid):
        z, x_reconstr = self.tae(x, datasetid)
        z_np = z.detach().cpu()

        similarity = compute_similarity_3d(
            z, self.centroids, similarity=self.similarity
        )

        ## Q (batch_size , n_clusters)
        '''
        Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)
        sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)
        Q = Q / sum_columns_Q
        '''
        Q = 1.0 / (1.0 + (similarity ** 2) / self.alpha_)
        Q = Q.pow((self.alpha_ + 1.0) / 2.0)
        sum_rows_Q = Q.sum(dim=1, keepdim=True) 
        Q = Q / sum_rows_Q
        
        ## P : ground truth distribution
        '''
        P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1, -1)
        sum_columns_P = torch.sum(P, dim=1).view(-1, 1)
        P = P / sum_columns_P
        '''
        P = Q ** 2 / Q.sum(0)
        P = (P.T / P.sum(1)).T
        
        return z, x_reconstr, Q, P
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
    from config import get_arguments
    parser = get_arguments()
    args = parser.parse_args()

    path_weights = args.path_weights.format(args.dataset_name)
    if not os.path.exists(path_weights):
        os.makedirs(path_weights)
    args.path_weights_ae = os.path.join(path_weights, "autoencoder_weight.pth")
    args.path_weights_main = os.path.join(path_weights, "full_model_weigths.pth")

    args.batch_size = 10
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, X_scaled = get_loader(args)  
    trainloader_list = [trainloader, trainloader]
    args.timesteps = X_scaled.shape[1]
    args.dataset_num = len(trainloader_list)
    # list(features of per dataset)
    args.inchannels = [X_scaled.shape[2], X_scaled.shape[2]]
    model = ClusterNet(args)

    #example_init_centroids
    X_tensor = torch.from_numpy(X_scaled).type(torch.FloatTensor).to(args.device)
    X_list = [X_tensor,X_tensor]
    model.init_centroids(X_list)

    #test ClusterNet
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        print(inputs.shape)
        inputs = inputs.type(torch.FloatTensor).to(args.device)
        z, x_reconstr, Q, P = model(inputs,0)
        print(Q.shape)

