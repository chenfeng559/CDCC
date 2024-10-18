import torch
import torch.nn as nn
import os
import random
import numpy as np
from config import get_arguments
from TAE import TAE
from Cluster_layer import ClusterNet
from load_data_np import get_loader_np_train_test
from utils_DTC import print_cluster_info,print_error_info,\
        compute_class_specific_accuracy,print_f1_f1pa,evaluation_clu_metric,get_logger
import sys

def pretrain_autoencoder(args, trainloader_list, verbose=True):
    """
    function for the autoencoder pretraining
    """
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
            all_loss = 0
            trainloader = trainloader_list[i]
            for batch_idx, (inputs, _, anomaly_labels,_) in enumerate(trainloader):
                inputs = inputs.type(torch.FloatTensor).to(args.device)
                optimizer.zero_grad()
                z, x_reconstr = tae(inputs,i)
                normal_indices = torch.where(anomaly_labels == 0)
                # l2 regularization
                l2_regularization = torch.sum(tae.Input_embeding_list[i].input_embeding.weight ** 2) + \
                        torch.sum(tae.Input_embeding_list[i].input_embeding.bias ** 2)
                # only reconstruct normaly sample
                loss_mse = loss_ae(inputs[normal_indices], x_reconstr[normal_indices]) \
                    + l2_regularization*args.gamma2
                loss_mse.backward()
                all_loss += loss_mse.item()
                optimizer.step()
            if verbose:
                logger.info(                    
                    "Pretraining  for No.{} dataset, loss for epoch {} is : {}".format(
                        i, epoch, all_loss / (batch_idx + 1)
                    ))
    logger.info("Ending pretraining autoencoder.")
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
    errors_list = [] #list(erros of per dataset)
    class_pred_pro_list = [] #list(class_pred_pro of per dataset)
    class_label_list = [] #list(class_label of per dataset)

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
            #输入，分类label，异常类型label（0~n），seq级别异常label（0，1）
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
            l2_regularization_input = torch.sum(model.tae.Input_embeding_list[i].input_embeding.weight ** 2) + \
            torch.sum(model.tae.Input_embeding_list[i].input_embeding.bias ** 2)
            l2_regularization_output = torch.sum(model.tae.Output_embeding_list[i].output_embeding.weight ** 2) + \
            torch.sum(model.tae.Output_embeding_list[i].output_embeding.bias ** 2)
            l2_regularization = l2_regularization_input+l2_regularization_output
            # only compute normal sample reconstruct loss
            loss_mse = loss_construct(inputs[nomaly_indices], x_reconstr[nomaly_indices])
            # cluster loss
            loss_KL = kl_loss_function(P, Q)
            # classification loss
            loss_cls = loss_ce(class_pred_pro, class_label) 

            errors = (inputs-x_reconstr)**2
            
            total_loss = loss_mse + args.gamma1*loss_KL + l2_regularization*args.gamma2 + loss_cls*args.gamma3
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
            logger.info(
                "For epoch：{} dataset：NO.{} Loss is : {} mse loss:{} kl loss:{} cls loss:{}".format(
                    epoch,
                    i,
                    train_loss / (batch_idx + 1),
                    train_tae_loss / (batch_idx + 1),
                    train_kl_loss / (batch_idx + 1),
                    train_cls_loss / (batch_idx + 1)
                )
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
    
    return (preds_list,gt_list,train_loss_list,anomaly_labels_list,seq_anomaly_labels_list,\
        z_list,errors_list,class_pred_pro_list,class_label_list)
    

def training_function(args, model, trainloader_list, X_scaled_list, anomaly_list, verbose=True):
    """
    function for training the DTC network.
    """
    ## initialize clusters centroids
    X_tensor_list = []
    for i in range(len(X_scaled_list)):
        X_tensor = torch.from_numpy(X_scaled_list[i]).type(torch.FloatTensor).to(args.device)
        X_tensor_list.append(X_tensor)
    model.init_centroids(X_tensor_list, anomaly_list)
    model = model.to(args.device)
    optimizer_clu = torch.optim.SGD(
        model.parameters(), lr=args.lr_cluster, momentum=args.momentum
    )
    print(model)

    ## train clustering model
    minloss = 1000000
    for epoch in range(args.max_epochs):
        preds_list, gt_list, train_loss_list, anomaly_label_list, seq_anomaly_label_list, \
            z_list, errors_list, class_pred_pro_list, class_label_list \
                = train_ClusterNET_with_fake(model, trainloader_list, optimizer_clu, epoch, args, verbose=verbose)

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
    return preds_list,gt_list,anomaly_label_list,seq_anomaly_label_list,z_list, \
        errors_list,class_pred_pro_list,class_label_list


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
            errors = (inputs - x_reconstr) ** 2
            
            preds = Q #torch.max(Q, dim=1)[1]
            # print(z)
            # print(z.shape) [64, 11, 1]
            # print(preds.shape)[53, 14]
            # sys.exit()
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
        # print(all_preds.shape)
        # sys.exit()


        preds_list.append(all_preds)
        gt_list.append(all_gt)
        anomaly_labels_list.append(all_anomaly_label)
        seq_anomaly_labels_list.append(all_seq_anomaly_label)
        z_list.append(all_z)
        errors_list.append(all_errors)
        class_pred_pro_list.append(all_class_pred_pro)
        class_label_list.append(all_class_label)
    
    return (preds_list, gt_list, anomaly_labels_list, seq_anomaly_labels_list,\
         z_list, errors_list, class_pred_pro_list, class_label_list)
    

def test_function(args, model, testloader_list):

    """
    function for training the DTC network.
    """
    ## test clustering model
    preds_list, gt_list, anomaly_label_list, seq_anomaly_label_list, z_list, errors_list, \
        class_pred_pro_list,class_label_list = test_ClusterNET_with_fake(model, testloader_list, args)
    # z_list = z_list[0]
    # print(z_list.shape[0])
    # preds_list=preds_list[0]
    # print(preds_list.shape[0])
    # sys.exit()
    #聚类metric以及异常auc
    print_test_info(datasets, preds_list, anomaly_label_list, seq_anomaly_label_list, z_list, errors_list, logger)
    #异常f1
    print_f1_f1pa(datasets, errors_list, anomaly_label_list, seq_anomaly_label_list, logger)
    #用于验证分类器性能
    #compute_class_specific_accuracy(class_pred_pro_list, class_label_list)
    return 


def print_test_info(datasets, preds_list,anomaly_label_list,seq_anomaly_label_list,z_list,errors_list, logger):
    '''print test_metric'''
    Silhouette_Coefficient = evaluation_clu_metric('Silhouette Coefficient', z_list, preds_list)
    calinski_harabasz_score = evaluation_clu_metric('calinski_harabasz_score',z_list, preds_list)
    davies_bouldin_score = evaluation_clu_metric('davies_bouldin_score', z_list, preds_list)

    logger.info('Silhouette_Coefficient:{:.3f}'.format(Silhouette_Coefficient))
    logger.info('calinski_harabasz_score:{:.3f}'.format(calinski_harabasz_score))
    logger.info('davies_bouldin_score:{:.3f}'.format(davies_bouldin_score))
    '''print erros'''
    print_error_info(datasets,errors_list, anomaly_label_list, seq_anomaly_label_list, logger)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False 
    # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


def get_data(args, dataset_name, channels):
    return get_loader_np_train_test(args, dataset_name, True, [55, channels], 55, anomaly=False)

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
    args.batch_size = 64
    args.drop_last = True

    logger = get_logger('experiment_log/DTC-Cluster-exp/exp_real_data.log')

    '''get data'''
    datasets = ['SWaT']#, 'WADI']#, 'MSL', 'PSM', 'SMAP', 'SMD']
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

    # '''Pretrain TAE'''
    # logger.info('Pretrain TAE')
    # pretrain_autoencoder(args, trainloader_list)
    # '''train DTC Net'''
    # logger.info('Training DTC Model')
    # model = ClusterNet(args)
    # preds_list, gt_list, anomaly_label_list, seq_anomaly_label_list, z_list, errors_list, \
    #     class_pred_pro_list, class_label_list = training_function(args, model,\
    #         trainloader_list, X_scaled_list, anomaly_label_train_list)


    '''test DTC Net'''
    logger.info('Testing DTC Model')
    model_test = ClusterNet(args)
    model_path = args.path_weights_main
    model_test.load_state_dict(torch.load(model_path))
    model_test.to(args.device)
    test_function(args, model_test, testloader_list)

    '''logging args'''
    datasets_arguments_info = (
        f"--dataset_name: {args.dataset_name}, "f"--path_weights: {args.path_weights}, "
        f"--path_data: {args.path_data}, "f"--dataset_num: {args.dataset_num}, "f"--inchannels: {args.inchannels}, "
        f"--n_clusters: {args.n_clusters}, "f"--timesteps: {args.timesteps}, "
    )
    anomaly_arguments_info=(     
        f"--ano_sample_rate: {args.ano_sample_rate}, "f"--ano_type_num: {args.ano_type_num}, "
        f"--ano_col_rate: {args.ano_col_rate}, "f"--ano_time_rate_max: {args.ano_time_rate_max}, "f"--ano_time_rate_min: {args.ano_time_rate_min}, "
        )   
    model_arguments_info=(
        f"--pool_size: {args.pool_size}, "f"--n_filters: {args.n_filters}, "f"--kernel_size: {args.kernel_size}, "
        f"--strides: {args.strides}, "f"--n_units: {args.n_units}, "f"--similarity: {args.similarity}, "
        f"--alpha: {args.alpha}, "f"--input_embeding_units: {args.input_embeding_units}, "
        f"--gamma1: {args.gamma1}, "f"--gamma2: {args.gamma2}, "f"--gamma3: {args.gamma3}, "
        f"--classfier_hidden_dim: {args.classfier_hidden_dim}, "
        )
    train_arguments_info=(    
        f"--batch_size: {args.batch_size}, "
        f"--epochs_ae: {args.epochs_ae}, "
        f"--max_epochs: {args.max_epochs}, "
        f"--max_patience: {args.max_patience}, "
        f"--lr_ae: {args.lr_ae}, "
        f"--lr_cluster: {args.lr_cluster}, "
        f"--momentum: {args.momentum}"
        )  
    logger.info(datasets_arguments_info)
    logger.info(anomaly_arguments_info)
    logger.info(model_arguments_info)
    logger.info(train_arguments_info)



