from operator import gt
import torch
import numpy as np
import logging
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score,davies_bouldin_score #聚类内部评估指标
from typing import Iterable
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import f1_score
from scipy.stats import iqr
from munkres import Munkres,print_matrix
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys



'''用于计算real-fake的分类性能'''
def compute_class_specific_accuracy(pred_list, label_list):
    """
    Args:
        pred_list (list): list(per dataset pred prob)
        label_list (list): list(per dataset real_fake label)
    """    
    for i in range(len(pred_list)):
        preds_all = pred_list[i]
        label_all = label_list[i]
        # 将预测结果转换为二分类标签
        preds_label = (preds_all > 0.5).astype(int)
        
        # 初始化每个类别的准确率变量
        accuracy_class_0 = 0
        accuracy_class_1 = 0
        
        # 计算每个类别的准确率
        for pred, label in zip(preds_label, label_all):
            if label == 0:
                if pred == 0:
                    accuracy_class_0 += 1  # 正确预测为0
            else:  # label == 1
                if pred == 1:
                    accuracy_class_1 += 1  # 正确预测为1
        
        # 计算每个类别的总数
        total_class_0 = np.sum(label_all == 0)
        total_class_1 = np.sum(label_all == 1)
        
        # 计算准确率
        accuracy_class_0 /= total_class_0 if total_class_0 > 0 else 1
        accuracy_class_1 /= total_class_1 if total_class_1 > 0 else 1
    
        print('acc_0:{},  acc_1:{}'.format(accuracy_class_0,accuracy_class_1))
        print('precision:{}'.format(precision_score(label_all,preds_label)))
        print('recall:{}'.format(recall_score(label_all,preds_label)))
        print('f1:{}'.format(f1_score(label_all,preds_label)))

'''打印聚类结果：每个cluster中各个class样本的分布情况'''
def print_cluster_info(preds_list,gt_list,logger):
    """
    Args:
        preds_all (np array): cluster label
        gt_all (np array): class label
    """    
    preds_all = np.concatenate(preds_list, axis=0)
    gt_all = np.concatenate(gt_list, axis=0)
    preds_all = np.argmax(preds_all, axis=1)
    # 步骤1: 创建一个字典来存储每个聚类的真实标签列表
    cluster_to_labels = defaultdict(list)
    # 填充字典
    for pred, gt in zip(preds_all, gt_all):
        cluster_to_labels[pred].append(gt)
    # 打印信息
    for cluster, labels in cluster_to_labels.items():
        # 统计每个聚类中每个真实标签的数量
        label_counts = np.bincount(labels)
        logger.info(f"Cluster {cluster} bincount {label_counts}")
    return 

'''plot confusion martix'''
def plot_confusion_martix(preds_list, gt_list, title='Confusion Matrix', picture_name='picture.png'):
    """
    Args:
        preds_list ( list(np array) ): list(cluster label of per dataset)
        gt_list ( list(np array) ): list(class label of per dataset)
        title (str): con_matrix title
        picture_name (str): picture save path
    """    
    preds_all = np.concatenate(preds_list,axis=0)
    gt_all = np.concatenate(gt_list,axis=0)
    predicted_labels = best_map(preds_all,gt_all)
    conf_matrix = confusion_matrix(gt_all, predicted_labels)
    # 使用 seaborn 创建热图
    plt.figure(figsize=(10, 7))  # 可以调整图形大小
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(len(np.unique(gt_all))), yticklabels=range(len(np.unique(gt_all))),
                linewidths=.5, square=True, cbar_kws={"shrink": .5})
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(picture_name, format='png', dpi=300)  # 保存为PNG格式，
    return conf_matrix

'''分类metric评估聚类结果'''
def evaluation_cls_metric(metric_name, preds_list, gt_list):
    """
    Args:
        metric_name (str): metric_name
        preds_list ( list(np array) ): list(per dataset clustering pred)
        gt_list ( list(np array) ): list(per dataset class label)
    Returns:
        _type_: metric result
    """    
    preds_all = np.concatenate(preds_list,axis=0)
    gt_all = np.concatenate(gt_list,axis=0)
    predicted_labels = best_map(preds_all,gt_all)
    if metric_name=='roc_auc':
        roc_score = roc_auc_score(gt_all, preds_all, multi_class='ovr')
        return roc_score
    elif metric_name=='acc':
        accuracy = accuracy_score(gt_all, predicted_labels)
        return accuracy
    elif metric_name=='f1':
        f1 = f1_score(gt_all, predicted_labels, average='macro')
        return f1
    else:
        return None

'''聚类metric评估聚类结果'''
def evaluation_clu_metric(metric_name, X_list, preds_list):
    print("here is evaluation_clu_metric")
    """
    Args:
        metric_name (str)
        X_list ( list(np array) ): list(per dataset hidden)
        preds_list ( list(np array) ):  list(per dataset clustering pred)
    Returns:
        metric result
    """    
    preds_all = np.concatenate(preds_list, axis=0)
    preds_all = np.argmax(preds_all, axis=1)

    X = np.concatenate(X_list, axis=0)
    if metric_name=='Silhouette Coefficient':#内部评估指标
        silhouette_avg = silhouette_score(X, preds_all)
        print("Silhouette Coefficient: %0.3f" % silhouette_avg)
        return silhouette_avg
    if metric_name=='calinski_harabasz_score':
        ch_score = calinski_harabasz_score(X,preds_all)
        print("Calinski-Harabasz Index: %0.3f" % ch_score)
        return ch_score
    if metric_name=='davies_bouldin_score':
        dbi = davies_bouldin_score(X,preds_all)
        print("Davies-Bouldin Index: %0.3f" % dbi)
        return dbi
    return 

def compute_CE(x):
    """
    x shape : (n , n_hidden)
    return : output : (n , 1)
    """
    return torch.sqrt(torch.sum(torch.square(x[:, 1:] - x[:, :-1]), dim=1))

'''compute distance between z and centroids'''
def compute_similarity_3d(z, centroids, similarity="EUC"):
    """
    Args:
        z (tensor): hidden，shape[Batch,Length,Dim]
        centroids (tensor): cluster centorids, shape[cluster_num, cluster_center_size ]
        similarity (str): similarity computed metric.
    Returns:
        _type_: distance
    """    
    if similarity == 'EUC':
        distance = torch.sum(torch.sqrt(torch.sum(torch.square(torch.unsqueeze(z, dim=1) - centroids), dim=2)), dim=-1)
    elif similarity == 'CID':
        # 计算CID距离
        ce_x = torch.sqrt(torch.sum(torch.square(z[:, 1:, :] - z[:, :-1, :]), dim=1))
        ce_w = torch.sqrt(torch.sum(torch.square(centroids[:, 1:, :] - centroids[:, :-1, :]), dim=1))
        ce = torch.max(torch.unsqueeze(ce_x, dim=1), ce_w) / torch.min(torch.unsqueeze(ce_x, dim=1), ce_w)
        ed = torch.sqrt(torch.sum(torch.square(torch.unsqueeze(z, dim=1) - centroids), dim=2))
        distance = torch.sum(ed * ce, dim=-1)
    elif similarity == 'COR':
        # 计算相关性距离
        inputs_norm = (z - torch.unsqueeze(torch.mean(z, dim=1), dim=1)) / torch.unsqueeze(torch.std(z, dim=1), dim=1)
        clusters_norm = (centroids - torch.unsqueeze(torch.mean(centroids, dim=1), dim=1)) / torch.unsqueeze(torch.std(centroids, dim=1), dim=1)
        pcc = torch.mean(torch.unsqueeze(inputs_norm, dim=1) * clusters_norm, dim=2)  # Pearson correlation coefficients
        distance = torch.sum(torch.sqrt(2.0 * (1.0 - pcc)), dim=-1)
    return distance

'''errors to anomaly_score'''
def get_final_error(errors: np.ndarray, ignore_dims=None, topk=1 ):
    """
    Args:
        errors (np.ndarray): errors of one sample, shape:[1,Length,Dim]
        ignore_dims (int): ignore_dim index
        topk (int): topk value
    Returns:
        final_errors: anomaly score of per time point, shape[Length]
    """    
    # Normalization
    median, iqr_ = np.mean(errors, axis=0), iqr(errors, axis=0)
    errors = (errors - median) / (iqr_ + 1e-9)

    # 最大值法
    if ignore_dims:
        errors[:, ignore_dims] = 0
    error_poses = np.argmax(errors, axis=1)
    topk_indices = np.argpartition(errors, -topk, axis=1)
    final_errors = np.take_along_axis(errors, topk_indices[:, -topk:], axis=1).sum(axis=1)
    return final_errors

'''打印异常检测性能评估结果——auc'''
def print_error_info(datasets, errors_list, anomaly_label_list, seq_anomaly_label_list, logger):
    """
    Args:
        datasets ( list(str) ): list(per dataset name)
        errors_list ( list(np array) ): list(per dataset errors(all sample) )
        anomaly_label_list ( list(np array) ): list(per dataset anomaly_label)
        seq_anomaly_label_list ( list(np array) ): list(per dataset seq_anomaly_label)
    """    
    anomaly_score_all = []
    anomaly_label_all = []
    #分别计算每个数据集上的异常检测性能
    for i in range(len(errors_list)):
        anomaly_score = []
        anomaly_label = []
        errors = errors_list[i]
        labels = seq_anomaly_label_list[i]
        for j in range(len(errors)):
            # score = get_final_error(errors[j])
            score = errors[j]
            anomaly_score.append(score)
            anomaly_label.append(labels[j])
            anomaly_score_all.append(score)
            anomaly_label_all.append(labels[j])
        anomaly_score = np.concatenate(anomaly_score, axis=0)
        anomaly_score = get_final_error(anomaly_score)
        anomaly_label = np.concatenate(anomaly_label, axis=0)
        logger.info(datasets[i])
        logger.info('anomaly_sample_num:{}'.format(np.sum(anomaly_label)))
        logger.info('total_sample_num:{}'.format(anomaly_label.shape[0]))
        roc_score = roc_auc_score(anomaly_label, anomaly_score)
        logger.info('roc_auc_score:{}'.format(roc_score))
    #计算全部数据集上的异常检测性能
    # anomaly_score_all =  np.concatenate(anomaly_score_all, axis=0)
    # anomaly_label_all =  np.concatenate(anomaly_label_all, axis=0)
    # logger.info('dataset_total')
    # logger.info('anomaly_sample_num:{}'.format(np.sum(anomaly_label_all)))
    # logger.info('total_sample_num:{}'.format(anomaly_label_all.shape[0]))
    # roc_score = roc_auc_score(anomaly_label_all, anomaly_score_all)
    # logger.info('roc_auc_score:{}'.format(roc_score))

'''打印异常检测性能评估结果——f1'''
def print_f1_f1pa(datasets, errors_list, anomaly_label_list, seq_anomaly_label_list, logger):
    for i in range(len(errors_list)):
        anomaly_score = []
        anomaly_label = []
        errors = errors_list[i]
        labels = seq_anomaly_label_list[i]
        # labels = anomaly_label_list[i]
        # print(labels.shape)
        # sys.exit()

        # 取最后一个时间点的作为异常分数, 一般用于stride取1时
        # pos = errors.shape[1] - 1
        # for j in range(len(errors)):
        #     score = errors[j]
        #     score = score[pos, :]
        #     anomaly_score.append(score)
        #     label = labels[j]
        #     anomaly_label.append(label[pos, :])

        # anomaly_score = np.array(anomaly_score)
        # anomaly_score = get_final_error(anomaly_score)
        # # anomaly_score =  np.concatenate(anomaly_score, axis=0)
        # anomaly_label =  np.concatenate(anomaly_label, axis=0)

        for j in range(len(errors)):
            score = errors[j]
            # score = get_final_error(errors[j])
            anomaly_score.append(score)
            anomaly_label.append(labels[j])

        anomaly_score =  np.concatenate(anomaly_score, axis=0)
        anomaly_score = get_final_error(anomaly_score)
        anomaly_label =  np.concatenate(anomaly_label, axis=0)
        # print(anomaly_score.shape)
        # sys.exit()
        logger.info(datasets[i])
        logger.info('anomaly_sample_num:{}'.format(np.sum(anomaly_label)))
        logger.info('total_sample_num:{}'.format(anomaly_label.shape[0]))
        best_thr, best_f1 = get_best_threshold(anomaly_score, anomaly_label, adjust=False)
        logger.info('best_thr:{}'.format(best_thr))
        logger.info('best_f1:{}'.format(best_f1))

'''记录日志'''
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s",datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


## 匈牙利算法,用于对齐cluster与class
def best_map(preds_all,gt_all):
    L1= gt_all
    L2= np.argmax(preds_all, axis=1)
    #L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)       # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)        # 标签的大小
    Label2 = np.unique(L2)       
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def adjust_predicts(pred, true, min_ratio=0):
    """
    Adjust predicted results by groud truth.

    Returns:
        - pred: adjusted pred.
    """ 
    def _get_anomaly_interval():
        true_diff = np.diff(true, prepend=[0], append=[0])
        low = np.argwhere(true_diff > 0).reshape(-1)
        high = np.argwhere(true_diff < 0).reshape(-1)
        return low, high

    for low, high in zip(*_get_anomaly_interval()):
        if pred[low: high].sum() >= max(1, min_ratio * (high - low)):
            pred[low: high] = 1

    return pred

def get_best_threshold(scores, true, iter_steps=1000, adjust=False, min_ratio=0):
    """
    Get threshold of anomaly scores corresponding to the best f1.

    Returns:
        - threshold: Best threshold.
        - f1: best f1 score.
    """
    sorted_index = np.argsort(scores)
    th_vals = np.linspace(0, len(scores) - 1, num=iter_steps)

    best_f1, best_thr = 0, 0
    for th_val in th_vals:
        cur_thr = scores[sorted_index[int(th_val)]]
        cur_pred = (scores >= cur_thr).astype(int)
        if adjust:
            cur_pred = adjust_predicts(cur_pred, true, min_ratio)
        cur_f1 = f1_score(true, cur_pred)

        if cur_f1 > best_f1:
            best_f1, best_thr = cur_f1, cur_thr

    return best_thr, best_f1






'''保存预测结果
def save_preds(preds_list, gt_list, anomaly_label_list, save_name='test1.xlsx'):
    preds_all = np.concatenate(preds_list,axis=0)
    gt_all = np.concatenate(gt_list,axis=0)
    anomaly_label_all = np.concatenate(anomaly_label_list,axis=0)
    predicted_labels = np.array(compute_dominant(preds_all, gt_all), dtype=int)
    total_array = np.stack((gt_all, predicted_labels, anomaly_label_all),axis=1)
    # 将NumPy数组转换为Pandas DataFrame
    df = pd.DataFrame(total_array, columns=['gt', 'predicted_label', 'anomaly_label'])
    # 保存DataFrame到Excel文件
    df.to_excel(save_name, index=False, engine='openpyxl')
'''

'''compute_dominant
def compute_dominant(preds_all,gt_all):

    preds_all = np.argmax(preds_all, axis=1)
    # 步骤1: 创建一个字典来存储每个聚类的真实标签列表
    cluster_to_labels = defaultdict(list)

    # 填充字典
    for pred, gt in zip(preds_all, gt_all):
        cluster_to_labels[pred].append(gt)

    # 步骤2和3: 计算每个聚类的主导标签
    dominant_labels = {}
    for cluster, labels in cluster_to_labels.items():
        # 统计每个聚类中每个真实标签的数量
        label_counts = np.bincount(labels)
        # 选择数量最多的真实标签作为主导标签
        dominant_label = np.argmax(label_counts)
        dominant_labels[cluster] = dominant_label

    # 打印每个聚类的主导标签
    for cluster, dominant_label in dominant_labels.items():
        print(f"Cluster {cluster} has dominant true label {dominant_label}")
    # 修改标签
    updated_preds_all = [dominant_labels[cluster] for cluster in preds_all]
    return updated_preds_all
'''

if __name__ == "__main__":
    a = [1, 2, 3] ; b = [3, 2, 1]; c = [7, 8, 9]
    w2 = np.array([a,b,c])
    print(get_final_error(w2,topk=2))
