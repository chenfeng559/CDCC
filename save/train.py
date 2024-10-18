import torch
import torch.nn as nn
import os
import random
import numpy as np
from torch.utils.data import dataloader
from config import get_arguments
from TAE import  TAE
from Cluster_layer import ClusterNet
from sklearn.metrics import roc_auc_score
from load_data import get_loader
from load_data_new import get_loader_arrf, get_loader_arrf_train_test
from load_data_pt import get_loader_pt
from utils import test_metric, save_preds,print_cluster_info,test_metric_cluster,print_error_info


def pretrain_autoencoder(args, trainloader_list, verbose=True):
    """
    function for the autoencoder pretraining
    """
    print("Pretraining autoencoder... \n")

    ## define TAE architecture
    tae = TAE(args)
    tae = tae.to(args.device)

    ## MSE loss
    loss_ae = nn.MSELoss()
    ## Optimizer
    optimizer = torch.optim.Adam(tae.parameters(), lr=args.lr_ae)
    tae.train()

    for epoch in range(args.epochs_ae):
        # train for NO.i dataset
        for i in range(len(trainloader_list)):
            print("Pretraining  for No.{} dataset".format(i))
            all_loss = 0
            trainloader = trainloader_list[i]
            for batch_idx, (inputs, _, _) in enumerate(trainloader):
                inputs = inputs.type(torch.FloatTensor).to(args.device)
                optimizer.zero_grad()
                z, x_reconstr = tae(inputs,i)
                # l2 regularization
                l2_regularization = torch.sum(tae.Input_embeding_list[i].input_embeding.weight ** 2) + \
                        torch.sum(tae.Input_embeding_list[i].input_embeding.bias ** 2)
                loss_mse = loss_ae(inputs, x_reconstr)+l2_regularization*1e-1

                loss_mse.backward()
                all_loss += loss_mse.item()
                optimizer.step()
            if verbose:
                print(
                    "Pretraining autoencoder loss for epoch {} is : {}".format(
                        epoch, all_loss / (batch_idx + 1)
                    )
                )

    print("Ending pretraining autoencoder. \n")
    # save weights
    torch.save(tae.state_dict(), args.path_weights_ae)



def kl_loss_function(input, pred):
    out = input * torch.log((input) / (pred))
    return torch.mean(torch.sum(out, dim=1))



def train_ClusterNET(model, trainloader_list, X_tensor_list, optimizer_clu, epoch, args, verbose):
    """
    Function for training one epoch of the DTC
    """
    model.train()
    loss_construct = nn.MSELoss()
    preds_list = [] #list(preds of per dataset)
    gt_list = [] #list(gt of per dataset)
    anomaly_labels_list = [] #list(anomaly_label of per dataset)
    train_loss_list = [] #list(train_loss of per dataset)
    z_list = [] #list(z of per dataset)


    for i in range(len(trainloader_list)):
        trainloader = trainloader_list[i]
        train_loss = 0
        train_kl_loss = 0
        train_tae_loss = 0
        all_preds, all_gt = [], []
        all_z = []
        all_anomaly_label = []
        for batch_idx, (inputs, labels, anomaly_labels) in enumerate(trainloader):
            inputs = inputs.type(torch.FloatTensor).to(args.device)

            all_gt.append(labels.cpu().detach())
            all_anomaly_label.append(anomaly_labels.cpu().detach())

            optimizer_clu.zero_grad()
            z, x_reconstr, Q, P = model(inputs, i)
            # l2 regularization
            l2_regularization = torch.sum(model.tae.Input_embeding_list[0].input_embeding.weight ** 2) + \
            torch.sum(model.tae.Input_embeding_list[0].input_embeding.bias ** 2)
            loss_mse = loss_construct(inputs, x_reconstr)
            loss_KL = kl_loss_function(P, Q)
            total_loss = loss_mse + args.gamma*loss_KL+l2_regularization*1e-1
            total_loss.backward()
            optimizer_clu.step()

            preds = Q#torch.max(Q, dim=1)[1]
            all_preds.extend(preds.cpu().detach().numpy())
            all_z.extend(z.reshape(z.shape[0], -1).cpu().detach().numpy())
            train_loss += total_loss.item()
            train_kl_loss += loss_KL.item()
            train_tae_loss += loss_mse.item()

        if verbose:
            print(
                "For epoch：{} ".format(epoch),
                "dataset：NO.{} ".format(i),
                " Loss is : %.3f" % (train_loss / (batch_idx + 1)),
                "mse loss:{}".format(train_tae_loss / (batch_idx + 1)),
                "kl loss:{}".format(train_kl_loss / (batch_idx + 1))
            )
        all_gt = torch.cat(all_gt, dim=0).numpy()
        all_anomaly_label = torch.cat(all_anomaly_label, dim=0).numpy()
        all_preds = np.array(all_preds)
        all_z = np.array(all_z)


        preds_list.append(all_preds)
        gt_list.append(all_gt)
        anomaly_labels_list.append(all_anomaly_label)
        train_loss_list.append(train_loss / (batch_idx + 1))
        z_list.append(all_z)

    
    return (preds_list,gt_list,train_loss_list,anomaly_labels_list,z_list)



def train_ClusterNET_with_fake(model, trainloader_list, X_tensor_list, optimizer_clu, epoch, args, verbose):
    """
    Function for training one epoch of the DTC
    """
    model.train()
    loss_construct = nn.MSELoss()
    preds_list = [] #list(preds of per dataset)
    gt_list = [] #list(gt of per dataset)
    anomaly_labels_list = [] #list(anomaly_label of per dataset)
    train_loss_list = [] #list(train_loss of per dataset)
    z_list = [] #list(z of per dataset)
    errors_list = []


    for i in range(len(trainloader_list)):
        trainloader = trainloader_list[i]
        train_loss = 0
        train_kl_loss = 0
        train_tae_loss = 0
        all_preds, all_gt = [], []
        all_z = []
        all_anomaly_label = []
        all_errors = []
        for batch_idx, (inputs, labels, anomaly_labels) in enumerate(trainloader):
            inputs = inputs.type(torch.FloatTensor).to(args.device)

            all_gt.append(labels.cpu().detach())
            all_anomaly_label.append(anomaly_labels.cpu().detach())
            nomaly_indices = torch.where(anomaly_labels == 0)

            optimizer_clu.zero_grad()
            z, x_reconstr, Q, P = model(inputs, i)
            # l2 regularization
            l2_regularization = torch.sum(model.tae.Input_embeding_list[0].input_embeding.weight ** 2) + \
            torch.sum(model.tae.Input_embeding_list[0].input_embeding.bias ** 2)
            # only compute normal sample reconstruct loss
            loss_mse = loss_construct(inputs[nomaly_indices], x_reconstr[nomaly_indices])
            loss_KL = kl_loss_function(P, Q)
            errors = inputs-x_reconstr
            
            total_loss = loss_mse + args.gamma*loss_KL+l2_regularization*1e-1
            total_loss.backward()
            optimizer_clu.step()

            preds = Q#torch.max(Q, dim=1)[1]
            all_preds.extend(preds.cpu().detach().numpy())
            all_z.extend(z.reshape(z.shape[0], -1).cpu().detach().numpy())
            all_errors.extend(errors.cpu().detach().numpy())

            train_loss += total_loss.item()
            train_kl_loss += loss_KL.item()
            train_tae_loss += loss_mse.item()

        if verbose:
            print(
                "For epoch：{} ".format(epoch),
                "dataset：NO.{} ".format(i),
                " Loss is : %.3f" % (train_loss / (batch_idx + 1)),
                "mse loss:{}".format(train_tae_loss / (batch_idx + 1)),
                "kl loss:{}".format(train_kl_loss / (batch_idx + 1))
            )
        all_gt = torch.cat(all_gt, dim=0).numpy()
        all_anomaly_label = torch.cat(all_anomaly_label, dim=0).numpy()
        all_preds = np.array(all_preds)
        all_z = np.array(all_z)


        preds_list.append(all_preds)
        gt_list.append(all_gt)
        anomaly_labels_list.append(all_anomaly_label)
        train_loss_list.append(train_loss / (batch_idx + 1))
        z_list.append(all_z)
        errors_list.append(all_errors)

    
    return (preds_list,gt_list,train_loss_list,anomaly_labels_list,z_list,errors_list)
    




def training_function(args, model, trainloader_list, X_scaled_list, verbose=True):

    """
    function for training the DTC network.
    """
    ## initialize clusters centroids
    X_tensor_list = []
    for i in range(len(X_scaled_list)):
        X_tensor = torch.from_numpy(X_scaled_list[i]).type(torch.FloatTensor).to(args.device)
        X_tensor_list.append(X_tensor)
    model.init_centroids(X_tensor_list)
    model = model.to(args.device)
    optimizer_clu = torch.optim.SGD(
        model.parameters(), lr=args.lr_cluster, momentum=args.momentum
    )

    ## train clustering model
    minloss = 1000000
    print("Training full model ...")
    for epoch in range(args.max_epochs):
        #preds_list, gt_list, train_loss_list,anomaly_label_list,z_list = train_ClusterNET(model, trainloader_list, X_tensor_list, optimizer_clu, epoch, args, verbose=verbose)
        preds_list, gt_list, train_loss_list,anomaly_label_list,z_list,errors_list = train_ClusterNET_with_fake(model, trainloader_list, X_tensor_list, optimizer_clu, epoch, args, verbose=verbose)

        mean_loss = 0
        for loss in train_loss_list:
            mean_loss+=loss
        mean_loss = mean_loss/len(train_loss_list)
        if mean_loss < minloss:
            minloss = mean_loss
            patience = 0
        else:
            patience += 1
            if patience == args.max_patience:
                break 

    torch.save(model.state_dict(), args.path_weights_main)
    return preds_list,gt_list,anomaly_label_list,z_list,errors_list


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


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
    args.drop_last = True

    #trainloader, X_scaled = get_loader(args)
    #trainloader, X_scaled = get_loader_arrf(args,'NATOPS',pad_length=4,concat_test=True,anomaly=False)
    #trainloader, X_scaled = get_loader_arrf(args,'SAD',clamp_length=55,anomaly=False)
    #trainloader_2, X_scaled_2 = get_loader_arrf(args,'SAD',clamp_length=55,label_modify=6,anomaly=False)
    #trainloader, X_scaled = get_loader_pt('data/HAR',args,clamp_len=90)
    #trainloader_2, X_scaled_2 = get_loader_pt('data/Epilepsy',args,clamp_len=125,label_modify=6)

    trainloader, X_scaled, testloader, X_test = get_loader_arrf_train_test(args,'NATOPS',pad_length=4,concat_test=True,anomaly=False,test_size=0.3)
    trainloader_2, X_scaled_2, testloader_2, X_test_2 = get_loader_arrf_train_test(args,'SAD',clamp_length=55,label_modify=6,anomaly=False,test_size=0.3)

    

    trainloader_list = [trainloader, trainloader_2]
    args.n_clusters = 13
    args.inchannels = [X_scaled.shape[2], X_scaled_2.shape[2]]# list(features of per dataset)
    args.timesteps = X_scaled.shape[1]
    args.dataset_num = len(trainloader_list)


    '''pretrain tae'''
    #pretrain_autoencoder(args,trainloader_list)

    '''train DTC Net'''
    model = ClusterNet(args)
    X_scaled_list = [X_scaled, X_scaled_2]
    preds_list, gt_list, anomaly_label_list, z_list, errors_list = training_function(args,model,trainloader_list,X_scaled_list)
    print_cluster_info(preds_list, gt_list)

    '''print test_metric'''
    #roc_auc = test_metric('roc_auc',preds_list,gt_list)
    acc = test_metric('acc',preds_list,gt_list)
    f1 = test_metric('f1',preds_list,gt_list)
    confusion_martix = test_metric('confusion_martix',preds_list,gt_list,title = 'Confusion Matrix', picture_name='picture/8.2_cluster_num/SAD(400)+NATOPS(360)——L2*1e-1-cluster13.png')
    #print('roc_auc_score:{:.3f}'.format(roc_auc))
    print('acc:{:.3f}'.format(acc))
    print('f1:{:.3f}'.format(f1))

    Silhouette_Coefficient = test_metric_cluster('Silhouette Coefficient',z_list,preds_list)
    print('Silhouette_Coefficient:{:.3f}'.format(Silhouette_Coefficient))


    '''print erros'''
    erros_auc = print_error_info(errors_list, anomaly_label_list)


    '''save preds'''
    #save_preds(preds_list,gt_list,anomaly_label_list,'SAD(400)-30% anomaly-1.xlsx')


