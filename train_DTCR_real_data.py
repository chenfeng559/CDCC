import torch
import torch.nn as nn
import numpy as np
import random
import os
from sklearn.cluster import KMeans
from utils_DTCR import evaluation_cls_metric, evaluation_clu_metric, plot_confusion_matrix,print_cluster_info
from utils_DTC import print_error_info, print_f1_f1pa, compute_class_specific_accuracy, get_logger
from load_data_np import get_loader_np_train_test
from DTCR.Model_DTCR import DTCR

def train_ClusterNET_with_fake(model, trainloader_list, optimizer_clu, epoch, args, verbose):
    """
    Function for training one epoch of the DTC
    """
    model.train()
    loss_construct = nn.MSELoss()
    loss_ce = nn.BCELoss()
    encoder_state_list = [] #list(z of per dataset)
    gt_list = [] #list(gt of per dataset)
    anomaly_labels_list = [] #list(anomaly_label of per dataset)
    seq_anomaly_labels_list = []  #list(seq_anomaly_label of per dataset)
    train_loss_list = [] #list(train_loss of per dataset)
    errors_list = [] #list(erros of per dataset)
    class_pred_pro_list = [] #list(class_pred_pro of per dataset)
    class_label_list = [] #list(class_label of per dataset)

    for i in range(len(trainloader_list)):
        trainloader = trainloader_list[i]
        train_loss = 0
        train_kmeans_loss = 0
        train_constr_loss = 0
        train_cls_loss = 0
        all_encoder_state, all_gt = [], []
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
            # reconstr、kmeans for all samples
            inputs_reconstr, encoder_state, WTW, FTWTWF, class_pred_pro = model(inputs,i)

            # l2 regularization
            l2_regularization_input = torch.sum(model.Input_embeding_list[i].input_embeding.weight ** 2) + \
            torch.sum(model.Input_embeding_list[i].input_embeding.bias ** 2)
            l2_regularization_output = torch.sum(model.Output_embeding_list[i].output_embeding.weight ** 2) + \
            torch.sum(model.Output_embeding_list[i].output_embeding.bias ** 2)
            l2_regularization = l2_regularization_input+l2_regularization_output

            # only compute normal sample reconstruct loss
            loss_constr = loss_construct(inputs, inputs_reconstr)
            # cluster loss
            loss_Kmeans = torch.subtract(torch.trace(WTW),torch.trace(FTWTWF))
            # classification loss
            loss_cls = loss_ce(class_pred_pro,class_label) 

            errors = (inputs-inputs_reconstr)**2
            
            total_loss = loss_constr + loss_Kmeans*args.gamma1+l2_regularization*args.gamma2+loss_cls*args.gamma3
            total_loss.backward()
            optimizer_clu.step()

            all_encoder_state.extend(encoder_state.cpu().detach().numpy())
            all_errors.extend(errors.cpu().detach().numpy())
            all_class_pred_pro.extend(class_pred_pro.cpu().detach().numpy())
            all_class_label.extend(class_label.cpu().detach().numpy())

            train_loss += total_loss.item()
            train_kmeans_loss += loss_Kmeans.item()
            train_constr_loss += loss_constr.item()
            train_cls_loss += loss_cls.item()

            # 每10个epoch进行一次F矩阵更新
            if epoch%10==0 and epoch!=0:
                part_hidden_val = np.array(encoder_state.cpu().detach()).reshape(-1, np.sum(args.hidden_structs) * 2)
                W = part_hidden_val.T
                U, sigma, VT = np.linalg.svd(W)
                sorted_indices = np.argsort(sigma)
                topk_evecs = VT[sorted_indices[:-args.cluster_num - 1:-1], :]
                F_new = topk_evecs.T
                model.F = nn.Parameter(torch.tensor(F_new).to(args.device), requires_grad=False)

        if verbose:
            logger.info(
                "For epoch：{} dataset：NO.{} Loss is : {} mse loss:{} kmeans loss:{} cls loss:{}".format(
                    epoch,
                    i,
                    train_loss / (batch_idx + 1),
                    train_constr_loss / (batch_idx + 1),
                    train_kmeans_loss / (batch_idx + 1),
                    train_cls_loss / (batch_idx + 1)
                )
            )
        all_gt = torch.cat(all_gt, dim=0).numpy()
        all_anomaly_label = torch.cat(all_anomaly_label, dim=0).numpy()
        all_seq_anomaly_label = torch.cat(all_seq_anomaly_label, dim=0).numpy()
        all_encoder_state = np.array(all_encoder_state)
        all_errors = np.array(all_errors)
        all_class_pred_pro = np.array(all_class_pred_pro)
        all_class_label = np.array(all_class_label)


        encoder_state_list.append(all_encoder_state)
        gt_list.append(all_gt)
        anomaly_labels_list.append(all_anomaly_label)
        seq_anomaly_labels_list.append(all_seq_anomaly_label)
        train_loss_list.append(train_loss / (batch_idx + 1))
        errors_list.append(all_errors)
        class_pred_pro_list.append(all_class_pred_pro)
        class_label_list.append(all_class_label)

    return (encoder_state_list,gt_list,train_loss_list,anomaly_labels_list,seq_anomaly_labels_list,errors_list,class_pred_pro_list,class_label_list)
   
def training_function(args, model, trainloader_list, verbose=True):
    """
    function for training the DTC network.
    """
    model = model.to(args.device)
    optimizer_clu = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum
    )
    print(model)

    ## train clustering model
    minloss = 1000000
    for epoch in range(args.max_epochs):
        encoder_state_list, gt_list, train_loss_list,anomaly_label_list,seq_anomaly_label_list,errors_list,class_pred_pro_list,class_label_list = train_ClusterNET_with_fake(model, trainloader_list, optimizer_clu, epoch, args, verbose=verbose)

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
    torch.save(model.state_dict(), args.model_save_path)
    return encoder_state_list,gt_list,anomaly_label_list,seq_anomaly_label_list,errors_list,class_pred_pro_list,class_label_list

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def test_ClusterNET_with_fake(model, testloader_list, args):
    """
    Function for training one epoch of the DTC
    """
    model.eval()

    encoder_state_list = [] #list(z of per dataset)
    gt_list = [] #list(gt of per dataset)
    anomaly_labels_list = [] #list(anomaly_label of per dataset)
    seq_anomaly_labels_list = []  #list(seq_anomaly_label of per dataset)
    errors_list = [] #list(erros of per dataset)
    class_pred_pro_list = [] #list(class_pred_pro of per dataset)
    class_label_list = [] #list(class_label of per dataset)

    for i in range(len(testloader_list)):
        testloader = testloader_list[i]
        all_encoder_state, all_gt = [], []
        all_anomaly_label = []
        all_seq_anomaly_label = []
        all_errors = []
        all_class_pred_pro = []
        all_class_label = []
        for batch_idx, (inputs, labels, anomaly_labels, seq_anomaly_labels) in enumerate(testloader):
            #输入，分类label，异常类型label（0~n），seq级别异常label（0，1）
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            class_label = np.ones(anomaly_labels.shape[0])
            class_label[anomaly_labels == 0] = 0
            class_label = torch.tensor(class_label).float().unsqueeze(1).to(args.device)

            all_anomaly_label.append(anomaly_labels.cpu().detach())
            all_seq_anomaly_label.append(seq_anomaly_labels.cpu().detach())
            nomaly_indices = torch.where(anomaly_labels == 0)
            all_gt.append(labels[nomaly_indices].cpu().detach())

            # reconstr、kmeans for all samples
            inputs_reconstr, encoder_state, WTW, FTWTWF, class_pred_pro = model(inputs,i)
            errors = (inputs-inputs_reconstr)**2
            
            all_encoder_state.extend(encoder_state[nomaly_indices].cpu().detach().numpy())
            all_errors.extend(errors.cpu().detach().numpy())
            all_class_pred_pro.extend(class_pred_pro.cpu().detach().numpy())
            all_class_label.extend(class_label.cpu().detach().numpy())


        all_gt = torch.cat(all_gt, dim=0).numpy()
        all_anomaly_label = torch.cat(all_anomaly_label, dim=0).numpy()
        all_seq_anomaly_label = torch.cat(all_seq_anomaly_label, dim=0).numpy()
        all_encoder_state = np.array(all_encoder_state)
        all_errors = np.array(all_errors)
        all_class_pred_pro = np.array(all_class_pred_pro)
        all_class_label = np.array(all_class_label)


        encoder_state_list.append(all_encoder_state)
        gt_list.append(all_gt)
        anomaly_labels_list.append(all_anomaly_label)
        seq_anomaly_labels_list.append(all_seq_anomaly_label)
        errors_list.append(all_errors)
        class_pred_pro_list.append(all_class_pred_pro)
        class_label_list.append(all_class_label)

    return (encoder_state_list,gt_list,anomaly_labels_list,seq_anomaly_labels_list,errors_list,class_pred_pro_list,class_label_list)

def test_function(args, model, testloader_list, datasets):
    """
    function for testing the DTC network.
    """
    encoder_state_list, gt_list, anomaly_labels_list, seq_anomaly_labels_list, errors_list, class_pred_pro_list, class_label_list = test_ClusterNET_with_fake(model, testloader_list, args)
    encoder_state_all = np.concatenate(encoder_state_list,axis=0)
    gt_all = np.concatenate(gt_list,axis=0)
    # 聚类
    km = KMeans(n_clusters=args.cluster_num)
    km_idx = km.fit_predict(encoder_state_all)
    # 打印聚类信息
    #print_cluster_info(km_idx,gt_all)
    # 评估聚类结果：acc、roc_auc、f1
    #evaluation_cls_metric(prediction=km_idx, label=gt_all)
    # 评估聚类结果：轮廓系数
    Silhouette_Coefficient = evaluation_clu_metric('Silhouette Coefficient',encoder_state_all,km_idx)
    calinski_harabasz_score = evaluation_clu_metric('calinski_harabasz_score',encoder_state_all,km_idx)
    davies_bouldin_score = evaluation_clu_metric('davies_bouldin_score',encoder_state_all,km_idx)
    logger.info('Silhouette_Coefficient:{:.3f}'.format(Silhouette_Coefficient))
    logger.info('calinski_harabasz_score:{:.3f}'.format(calinski_harabasz_score))
    logger.info('davies_bouldin_score:{:.3f}'.format(davies_bouldin_score))

    # 绘制混淆矩阵
    #plot_confusion_matrix(prediction=km_idx,label=gt_all,picture_name='experiment/DTC-Cluster-exp/NATOPS.png')
    '''print erros'''
    print_error_info(datasets,errors_list, anomaly_labels_list, seq_anomaly_labels_list,logger)
    print_f1_f1pa(datasets, errors_list, anomaly_labels_list, seq_anomaly_labels_list,logger)
    #compute_class_specific_accuracy(class_pred_pro_list,class_label_list)
    


def get_data(args, dataset_name, channels):
    return get_loader_np_train_test(args, dataset_name, False, [55, channels], 55, anomaly=False)


if __name__ == "__main__":
    '''set args and datasets'''
    set_seed(42)
    from config_DTCR import get_arguments
    parser = get_arguments()
    args = parser.parse_args()
    args.model_save_path = 'model_weights_DTCR/model_multi_dataset_sample1.pth'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.batch_size = 64
    args.drop_last = True

    logger = get_logger('experiment_log/DTCR-Cluster-exp/exp_real_data.log')

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

    args.cluster_num = 14
    args.inchannels = channel_list
    args.timesteps = X_scaled_list[0].shape[1]
    args.dataset_num = len(trainloader_list)


    '''train DTCR Model'''
    # logger.info('Training DTCR Model')
    # model = DTCR(args)
    # encoder_state_list, gt_list, anomaly_label_list, seq_anomaly_label_list, errors_list, class_pred_pro_list, class_label_list = training_function(args,model,trainloader_list)
    
    '''test DTCR Model'''
    logger.info('Testing DTCR Model')
    args.train_decoder = False
    model_test = DTCR(args)
    model_path = args.model_save_path
    model_test.load_state_dict(torch.load(model_path))
    model_test.to(args.device)
    test_function(args,model_test,testloader_list,datasets)

    '''logging args'''
    datasets_arguments_info = (
        f"--dataset_name: {args.dataset_name}, "f"--path_weights: {args.path_weights}, "
        f"--dataset_num: {args.dataset_num}, "f"--cluster_num: {args.cluster_num}, "
        f"--batch_first: {args.batch_first}, "f"--drop_last: {args.drop_last}, "
    )
    anomaly_arguments_info=(     
        f"--ano_sample_rate: {args.ano_sample_rate}, "f"--ano_type_num: {args.ano_type_num}, "
        f"--ano_col_rate: {args.ano_col_rate}, "f"--ano_time_rate_max: {args.ano_time_rate_max}, "f"--ano_time_rate_min: {args.ano_time_rate_min}, "
        )   
    model_arguments_info=(
        f"--hidden_structs: {args.hidden_structs}, "f"--dilations: {args.dilations}, "f"--inchannels: {args.inchannels}, "
        f"--cell_type: {args.cell_type}, "f"--input_embeding_units: {args.input_embeding_units}, "
        f"--gamma1: {args.gamma1}, "f"--gamma2: {args.gamma2}, "f"--gamma3: {args.gamma3}, "
        )
    train_arguments_info=(    
        f"--batch_size: {args.batch_size}, "
        f"--max_epochs: {args.max_epochs}, "
        f"--max_patience: {args.max_patience}, "
        f"--lr: {args.lr}, "
        f"--momentum: {args.momentum}, "
        )  
    logger.info(datasets_arguments_info)
    logger.info(anomaly_arguments_info)
    logger.info(model_arguments_info)
    logger.info(train_arguments_info)


