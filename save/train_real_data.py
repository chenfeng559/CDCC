import torch
import torch.nn as nn
import os
import random
import numpy as np
from config import get_arguments
from TAE import  TAE
from Cluster_layer import ClusterNet
from load_data_np import get_loader_np_train_test
from utils import  test_metric_cluster,print_error_info,compute_class_specific_accuracy,print_f1_f1pa,compute_confusion_martix


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
            for batch_idx, (inputs, _, anomaly_labels,_) in enumerate(trainloader):
                inputs = inputs.type(torch.FloatTensor).to(args.device)
                optimizer.zero_grad()
                z, x_reconstr = tae(inputs,i)
                nomaly_indices = torch.where(anomaly_labels == 0)
                # l2 regularization
                l2_regularization = torch.sum(tae.Input_embeding_list[i].input_embeding.weight ** 2) + \
                        torch.sum(tae.Input_embeding_list[i].input_embeding.bias ** 2)
                # only reconstr nomaly sample
                loss_mse = loss_ae(inputs[nomaly_indices], x_reconstr[nomaly_indices])+l2_regularization*1e-1
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


def train_ClusterNET_with_fake(model, trainloader_list, optimizer_clu, epoch, args, verbose):
    """
    Function for training one epoch of the DTC
    """
    model.train()
    loss_construct = nn.MSELoss()
    loss_ce = nn.BCELoss()
    preds_list = [] #list(preds of per dataset)
    gt_list = [] #list(gt of per dataset)
    anomaly_labels_list = [] #list(anomaly_label of per dataset)
    seq_anomaly_labels_list = []  #list(seq_anomaly_label of per dataset)
    train_loss_list = [] #list(train_loss of per dataset)
    z_list = [] #list(z of per dataset)
    errors_list = []
    class_pred_pro_list = []
    class_label_list = []


    for i in range(len(trainloader_list)):
        trainloader = trainloader_list[i]
        train_loss = 0
        train_kl_loss = 0
        train_tae_loss = 0
        train_cls_loss = 0
        all_preds, all_gt = [], []
        all_z = []
        all_anomaly_label = []
        all_seq_anomaly_label = []
        all_errors = []
        all_class_pred_pro = []
        all_class_label = []
        for batch_idx, (inputs, labels, anomaly_labels, seq_anomaly_labels) in enumerate(trainloader):
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            class_label = np.ones(anomaly_labels.shape[0])
            class_label[anomaly_labels == 0] = 0
            class_label = torch.tensor(class_label).float().unsqueeze(1).to(args.device)


            all_anomaly_label.append(anomaly_labels.cpu().detach())
            all_seq_anomaly_label.append(seq_anomaly_labels.cpu().detach())
            nomaly_indices = torch.where(anomaly_labels == 0)
            all_gt.append(labels[nomaly_indices].cpu().detach())

            optimizer_clu.zero_grad()
            # Q,P only compute for normal sample
            z, x_reconstr, Q, P, class_pred_pro = model(inputs, i, nomaly_indices)
            #z, x_reconstr, Q, P = model(inputs, i, nomaly_indices)
            # l2 regularization
            l2_regularization = torch.sum(model.tae.Input_embeding_list[i].input_embeding.weight ** 2) + \
            torch.sum(model.tae.Input_embeding_list[i].input_embeding.bias ** 2)
            # only compute normal sample reconstruct loss
            loss_mse = loss_construct(inputs[nomaly_indices], x_reconstr[nomaly_indices])
            # cluster loss
            loss_KL = kl_loss_function(P, Q)
            # classification loss
            loss_cls = loss_ce(class_pred_pro,class_label) 

            errors = inputs-x_reconstr
            
            total_loss = loss_mse + args.gamma*loss_KL+ l2_regularization*(1e-1)+ loss_cls
            total_loss.backward()
            optimizer_clu.step()

            preds = Q#torch.max(Q, dim=1)[1]
            all_preds.extend(preds.cpu().detach().numpy())
            all_z.extend(z.reshape(z.shape[0], -1)[nomaly_indices].cpu().detach().numpy())
            all_errors.extend(errors.cpu().detach().numpy())
            all_class_pred_pro.extend(class_pred_pro.cpu().detach().numpy())
            all_class_label.extend(class_label.cpu().detach().numpy())

            train_loss += total_loss.item()
            train_kl_loss += loss_KL.item()
            train_tae_loss += loss_mse.item()
            train_cls_loss += loss_cls.item()

        if verbose:
            print(
                "For epoch：{} ".format(epoch),
                "dataset：NO.{} ".format(i),
                " Loss is : %.3f" % (train_loss / (batch_idx + 1)),
                "mse loss:{}".format(train_tae_loss / (batch_idx + 1)),
                "kl loss:{}".format(train_kl_loss / (batch_idx + 1)),
                "cls loss:{}".format(train_cls_loss / (batch_idx + 1))
            )
        all_gt = torch.cat(all_gt, dim=0).numpy()
        all_anomaly_label = torch.cat(all_anomaly_label, dim=0).numpy()
        all_seq_anomaly_label = torch.cat(all_seq_anomaly_label, dim=0).numpy()
        all_preds = np.array(all_preds)
        all_z = np.array(all_z)
        all_errors = np.array(all_errors)
        all_class_pred_pro = np.array(all_class_pred_pro)
        all_class_label = np.array(all_class_label)


        preds_list.append(all_preds)
        gt_list.append(all_gt)
        anomaly_labels_list.append(all_anomaly_label)
        seq_anomaly_labels_list.append(all_seq_anomaly_label)
        train_loss_list.append(train_loss / (batch_idx + 1))
        z_list.append(all_z)
        errors_list.append(all_errors)
        class_pred_pro_list.append(all_class_pred_pro)
        class_label_list.append(all_class_label)
    

    
    return (preds_list,gt_list,train_loss_list,anomaly_labels_list,seq_anomaly_labels_list,z_list,errors_list,class_pred_pro_list,class_label_list)
    


def training_function(args, model, trainloader_list, X_scaled_list, anomaly_list, verbose=True):

    """
    function for training the DTC network.
    """
    ## initialize clusters centroids
    X_tensor_list = []
    for i in range(len(X_scaled_list)):
        X_tensor = torch.from_numpy(X_scaled_list[i]).type(torch.FloatTensor).to(args.device)
        X_tensor_list.append(X_tensor)
    model.init_centroids(X_tensor_list,anomaly_list)
    model = model.to(args.device)
    optimizer_clu = torch.optim.SGD(
        model.parameters(), lr=args.lr_cluster, momentum=args.momentum
    )
    print(model)

    ## train clustering model
    minloss = 1000000
    print("Training full model ...")
    for epoch in range(args.max_epochs):
        #preds_list, gt_list, train_loss_list,anomaly_label_list,z_list = train_ClusterNET(model, trainloader_list, X_tensor_list, optimizer_clu, epoch, args, verbose=verbose)
        preds_list, gt_list, train_loss_list,anomaly_label_list,seq_anomaly_label_list,z_list,errors_list,class_pred_pro_list,class_label_list = train_ClusterNET_with_fake(model, trainloader_list, optimizer_clu, epoch, args, verbose=verbose)
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
    return preds_list,gt_list,anomaly_label_list,seq_anomaly_label_list,z_list,errors_list,class_pred_pro_list,class_label_list


def test_ClusterNET_with_fake(model, testloader_list, args):
    """
    Function for training one epoch of the DTC
    """
    model.eval()
    preds_list = [] #list(preds of per dataset)
    gt_list = [] #list(gt of per dataset)
    anomaly_labels_list = [] #list(anomaly_label of per dataset)
    seq_anomaly_labels_list = []  #list(seq_anomaly_label of per dataset)
    z_list = [] #list(z of per dataset)
    errors_list = []
    class_pred_pro_list = []
    class_label_list = []

    for i in range(len(testloader_list)):
        testloader = testloader_list[i]
        all_preds, all_gt = [], []
        all_z = []
        all_anomaly_label = []
        all_seq_anomaly_label = []
        all_errors = []
        all_class_pred_pro = []
        all_class_label = []

        for batch_idx, (inputs, labels, anomaly_labels, seq_anomaly_labels) in enumerate(testloader):
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            class_label = np.ones(anomaly_labels.shape[0])
            class_label[anomaly_labels == 0] = 0
            class_label = torch.tensor(class_label).float()
            class_label = torch.unsqueeze(class_label,1)
            class_label = class_label.to(args.device)
            

            all_anomaly_label.append(anomaly_labels.cpu().detach())
            all_seq_anomaly_label.append(seq_anomaly_labels.cpu().detach())
            nomaly_indices = torch.where(anomaly_labels == 0)
            all_gt.append(labels[nomaly_indices].cpu().detach())

            # Q,P only compute for normal sample
            z, x_reconstr, Q, P, class_pred_pro = model(inputs, i, nomaly_indices)
            #z, x_reconstr, Q, P = model(inputs, i, nomaly_indices)
            errors = inputs-x_reconstr
            
            preds = Q#torch.max(Q, dim=1)[1]
            all_preds.extend(preds.cpu().detach().numpy())
            all_z.extend(z.reshape(z.shape[0], -1)[nomaly_indices].cpu().detach().numpy())
            all_errors.extend(errors.cpu().detach().numpy())
            all_class_pred_pro.extend(class_pred_pro.cpu().detach().numpy())
            all_class_label.extend(class_label.cpu().detach().numpy())

        all_gt = torch.cat(all_gt, dim=0).numpy()
        all_anomaly_label = torch.cat(all_anomaly_label, dim=0).numpy()
        all_seq_anomaly_label = torch.cat(all_seq_anomaly_label, dim=0).numpy()
        all_preds = np.array(all_preds)
        all_z = np.array(all_z)
        all_errors = np.array(all_errors)
        all_class_pred_pro = np.array(all_class_pred_pro)
        all_class_label = np.array(all_class_label)


        preds_list.append(all_preds)
        gt_list.append(all_gt)
        anomaly_labels_list.append(all_anomaly_label)
        seq_anomaly_labels_list.append(all_seq_anomaly_label)
        z_list.append(all_z)
        errors_list.append(all_errors)
        class_pred_pro_list.append(all_class_pred_pro)
        class_label_list.append(all_class_label)
    
    return (preds_list,gt_list,anomaly_labels_list,seq_anomaly_labels_list,z_list,errors_list,class_pred_pro_list,class_label_list)
    


def test_function(args, model, testloader_list):

    """
    function for training the DTC network.
    """
    ## load model
    model_path = args.path_weights_main
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)

    ## test clustering model
    print("Test full model ...")
    preds_list, gt_list,anomaly_label_list,seq_anomaly_label_list,z_list,errors_list,class_pred_pro_list,class_label_list = test_ClusterNET_with_fake(model, testloader_list, args)
    return preds_list,gt_list,anomaly_label_list,seq_anomaly_label_list,z_list,errors_list,class_pred_pro_list,class_label_list


def print_test_info(datasets, preds_list,anomaly_label_list,seq_anomaly_label_list,z_list,errors_list):
    '''print test_metric'''
    Silhouette_Coefficient = test_metric_cluster('Silhouette Coefficient',z_list,preds_list)
    print('Silhouette_Coefficient:{:.3f}'.format(Silhouette_Coefficient))

    '''print erros'''
    print_error_info(datasets,errors_list, anomaly_label_list, seq_anomaly_label_list)


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
    #    trainloader_1, X_scaled_1, anomaly_label_train_1, testloader_1, X_test_1, anomaly_label_test_1 = get_loader_np_train_test(args,'SWaT',True,[55,51],55,anomaly=True)
    return get_loader_np_train_test(args, dataset_name, True, [55, channels], 55, anomaly=True,ano_sample_rate=0.5)

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

    args.batch_size = 128#64
    args.drop_last = False



    '''get data'''
    datasets = ['SWaT', 'WADI']#, 'MSL', 'PSM', 'SMAP', 'SMD']
    channels = [51, 123]#, 55, 25, 25, 38]

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
    
    
    args.n_clusters = 13
    args.inchannels = channel_list
    args.timesteps = X_scaled_list[0].shape[1]
    args.dataset_num = len(trainloader_list)
    
    '''pretrain tae'''
    pretrain_autoencoder(args,trainloader_list)
    '''train DTC Net'''
    model = ClusterNet(args)
    preds_list, gt_list, anomaly_label_list, seq_anomaly_label_list, z_list, errors_list,class_pred_pro_list,class_label_list = training_function(args,model,trainloader_list,X_scaled_list,anomaly_label_train_list)
    #preds_list, gt_list, anomaly_label_list, seq_anomaly_label_list, z_list, errors_list,class_pred_pro_list,class_label_list = test_function(args,model,trainloader_list)
    print_test_info(datasets, preds_list, anomaly_label_list, seq_anomaly_label_list, z_list, errors_list)
    #print_f1_f1pa(datasets, errors_list, anomaly_label_list, seq_anomaly_label_list)
    acc_0,acc_1 = compute_class_specific_accuracy(class_pred_pro_list,class_label_list)
    print('classification acc_0:{}  acc_1:{}'.format(acc_0,acc_1))
 



