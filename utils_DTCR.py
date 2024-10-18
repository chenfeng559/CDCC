import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from scipy.special import comb
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from munkres import Munkres
from collections import defaultdict


''' 匈牙利算法,用于将cluster与class进行最优匹配。当cluster_num大于class时，无法进行。'''
def best_map(gt_all,preds_all):
    """
    Args:
        gt_all (np array): class label
        preds_all (np array): cluster label
    Returns:
        cluster label which is mapping class label
    """    
    #L1 should be the labels and L2 should be the clustering number we got
    L1= gt_all
    L2= preds_all
    Label1 = np.unique(L1)       
    nClass1 = len(Label1)        
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
    # 修改cluster 标签为其对应的class 标签
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

'''绘制混淆矩阵，只有分类数据集才能够绘制混淆矩阵。'''
def plot_confusion_matrix(prediction, label, title='con_matrix',picture_name='model_weights_DTCR/picture/DTCR_1.png'):
    """
    Args:
        prediction (np array): cluster label
        label (np array): class label
        title (str): con_matrix title
        picture_name (str): picture save path
    """    
    prediction = best_map(label, prediction)
    conf_matrix = confusion_matrix(label, prediction)
    # 使用 seaborn 创建热图
    plt.figure(figsize=(10, 7))  # 可以调整图形大小
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(len(np.unique(label))), yticklabels=range(len(np.unique(label))),
                linewidths=.5, square=True, cbar_kws={"shrink": .5})
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(picture_name, format='png', dpi=300)  # 保存为PNG格式，

'''打印聚类结果：每个cluster中各个class样本的分布情况'''
def print_cluster_info(preds_all,gt_all,logger):
    """
    Args:
        preds_all (np array): cluster label
        gt_all (np array): class label
    """    
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

'''分类metric评估聚类性能'''
def evaluation_cls_metric(prediction, label, logger):
    prediction = best_map(label, prediction)
    acc = accuracy_score(label, prediction)
    f1 = f1_score(label, prediction, average='macro')
    #ri = rand_index_score(label, prediction)
    #nmi = metrics.normalized_mutual_info_score(label, prediction)
    #ari = metrics.adjusted_rand_score(label, prediction)
    logger.info('acc:{}  f1:{}'.format(acc, f1))

'''聚类metric评估聚类性能'''
def evaluation_clu_metric(metric_name,hidden_state_all, cluster_label):
    """
    Args:
        metric_name (str)
        hidden_state_list (np array): all sample hidden_state
        cluster_label (np array): all cluster label
    Returns:
        metric result
    """
    if metric_name=='Silhouette Coefficient':#内部评估指标
        silhouette_avg = silhouette_score(hidden_state_all, cluster_label)
        return silhouette_avg
    if metric_name=='calinski_harabasz_score':#内部评估指标
        ch_score = calinski_harabasz_score(hidden_state_all,cluster_label)
        return ch_score
    if metric_name=='davies_bouldin_score':#内部评估指标
        dbi = davies_bouldin_score(hidden_state_all,cluster_label)
        return dbi
    return


def ri_score(label, prediction):
    conf_matrix = confusion_matrix(label, prediction)
    TN, FP = conf_matrix[0]  
    FN, TP = conf_matrix[1]  
    print(TP,TN)
    n = len(label)
    return (TP+TN)/(n*(n-1)/2)

def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    print(tp,fp,fn,tn)
    return (tp + tn) / (tp + fp + fn + tn)
