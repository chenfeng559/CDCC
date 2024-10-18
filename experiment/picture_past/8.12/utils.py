from operator import gt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.metrics import silhouette_score, precision_score, recall_score
from typing import Iterable
import numpy as np
from scipy.stats import rankdata
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import f1_score
from scipy.stats import iqr
from munkres import Munkres,print_matrix
from collections import defaultdict
import pandas as pd

def compute_confusion_martix(y_pred,y_true,threshold,type):
    y_pred = (y_pred > threshold).astype(int)

    if type=='sklearn':
        conf_matrix = confusion_matrix(y_true, y_pred)
        TN, FP = conf_matrix[0]  
        FN, TP = conf_matrix[1]  
    else:
        # 初始化计数器
        TP = 0  # 真正例
        FN = 0  # 假负例
        TN = 0  # 真负例
        FP = 0  # 假正例

        # 遍历真实标签和预测标签
        for true, pred in zip(y_true, y_pred):
            if true == 1 and pred == 1:
                # 真实为正且预测为正，增加TP计数
                TP += 1
            elif true == 1 and pred == 0:
                # 真实为正但预测为负，增加FN计数
                FN += 1
            elif true == 0 and pred == 0:
                # 真实为负且预测为负，增加TN计数
                TN += 1
            elif true == 0 and pred == 1:
                # 真实为负但预测为正，增加FP计数
                FP += 1

    print(f"True Positives (TP): {TP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")


def compute_class_specific_accuracy(pred_list, label_list):
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







'''print cluster info'''
def print_cluster_info(preds_list,gt_list):
    preds_all = np.concatenate(preds_list,axis=0)
    gt_all = np.concatenate(gt_list,axis=0)
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
        print(f"Cluster {cluster} bincount {label_counts}")



'''分类metric以及混淆矩阵'''
def test_metric(metric_name, preds_list, gt_list, title='Confusion Matrix', picture_name='SAD(400)-0% anomaly.png'):
    preds_all = np.concatenate(preds_list,axis=0)
    gt_all = np.concatenate(gt_list,axis=0)
    #predicted_labels = compute_dominant(preds_all, gt_all)
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
    elif metric_name=='confusion_martix':
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
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
    else:
        return None

def test_metric_cluster(metric_name, X_list, preds_list):
    preds_all = np.concatenate(preds_list,axis=0)
    preds_all = np.argmax(preds_all, axis=1)
    X = np.concatenate(X_list,axis=0)
    if metric_name=='Silhouette Coefficient':
        silhouette_avg = silhouette_score(X, preds_all)
        return silhouette_avg
    return 




def compute_CE(x):
    """
    x shape : (n , n_hidden)
    return : output : (n , 1)
    """
    return torch.sqrt(torch.sum(torch.square(x[:, 1:] - x[:, :-1]), dim=1))



def compute_similarity_3d(z, centroids, similarity="EUC"):
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

def get_final_error(errors: np.ndarray, ignore_dims=None,
                        topk=1 ):

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



def print_error_info(datasets,errors_list, anomaly_label_list, seq_anomaly_label_list):
    anomaly_score_all = []
    anomaly_label_all = []
    for i in range(len(errors_list)):
        anomaly_score = []
        anomaly_label = []
        errors = errors_list[i]
        labels = seq_anomaly_label_list[i]
        for j in range(len(errors)):
            score = get_final_error(errors[j])
            anomaly_score.append(score)
            anomaly_label.append(labels[j])
            anomaly_score_all.append(score)
            anomaly_label_all.append(labels[j])
        anomaly_score = np.concatenate(anomaly_score, axis=0)
        #anomaly_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min())
        anomaly_label = np.concatenate(anomaly_label, axis=0)
        print(datasets[i])
        print('anomaly_sample_num:{}'.format(np.sum(anomaly_label)))
        print('total_sample_num:{}'.format(anomaly_label.shape[0]))
        roc_score = roc_auc_score(anomaly_label, anomaly_score)
        print('roc_auc_score:{}'.format(roc_score))
        #compute_confusion_martix(anomaly_score,anomaly_label,threshold=0.37203344654024095,type='sklearn')

    anomaly_score_all =  np.concatenate(anomaly_score_all, axis=0)
    anomaly_label_all =  np.concatenate(anomaly_label_all, axis=0)
    print('dataset_total')
    print('anomaly_sample_num:{}'.format(np.sum(anomaly_label_all)))
    print('total_sample_num:{}'.format(anomaly_label_all.shape[0]))
    roc_score = roc_auc_score(anomaly_label_all, anomaly_score_all)
    print('roc_auc_score:{}'.format(roc_score))

    

## 匈牙利算法
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

def print_f1_f1pa(datasets,errors_list, anomaly_label_list, seq_anomaly_label_list):
    for i in range(len(errors_list)):
        anomaly_score = []
        anomaly_label = []
        errors = errors_list[i]
        labels = seq_anomaly_label_list[i]
        for j in range(len(errors)):
            score = get_final_error(errors[j])
            anomaly_score.append(score)
            anomaly_label.append(labels[j])

        anomaly_score =  np.concatenate(anomaly_score, axis=0)
        #anomaly_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max()- anomaly_score.min())
        #print(anomaly_score.min())
        #print(anomaly_score.max())
        anomaly_label =  np.concatenate(anomaly_label, axis=0)
        print(datasets[i])
        print('anomaly_sample_num:{}'.format(np.sum(anomaly_label)))
        print('total_sample_num:{}'.format(anomaly_label.shape[0]))
        best_thr, best_f1 = get_best_threshold(anomaly_score, anomaly_label, adjust=False)
        print('best_thr:{}'.format(best_thr))
        print('best_f1:{}'.format(best_f1))







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

'''
def compute_similarity(z, centroids, similarity="EUC"):
    """
    Function that compute distance between a latent vector z and the clusters centroids.

    similarity : can be in [CID,EUC,COR] :  euc for euclidean,  cor for correlation and CID
                 for Complexity Invariant Similarity.
    z shape : (batch_size, n_hidden)
    centroids shape : (n_clusters, n_hidden)
    output : (batch_size , n_clusters)
    """
    n_clusters, n_hidden = centroids.shape[0], centroids.shape[1]
    bs = z.shape[0]

    if similarity == "CID":
        CE_z = compute_CE(z).unsqueeze(1)  # shape (batch_size , 1)
        CE_cen = compute_CE(centroids).unsqueeze(0)  ## shape (1 , n_clusters )
        z = z.unsqueeze(0).expand((n_clusters, bs, n_hidden))
        mse = torch.sqrt(torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2))
        CE_z = CE_z.expand((bs, n_clusters))  # (bs , n_clusters)
        CE_cen = CE_cen.expand((bs, n_clusters))  # (bs , n_clusters)
        CF = torch.max(CE_z, CE_cen) / torch.min(CE_z, CE_cen)
        return torch.transpose(mse, 0, 1) * CF

    elif similarity == "EUC":
        z = z.expand((n_clusters, bs, n_hidden))
        mse = torch.sqrt(torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2))
        return torch.transpose(mse, 0, 1)

    elif similarity == "COR":
        std_z = (
            torch.std(z, dim=1).unsqueeze(1).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        mean_z = (
            torch.mean(z, dim=1).unsqueeze(1).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        std_cen = (
            torch.std(centroids, dim=1).unsqueeze(0).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        mean_cen = (
            torch.mean(centroids, dim=1).unsqueeze(0).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        ## covariance
        z_expand = z.unsqueeze(1).expand((bs, n_clusters, n_hidden))
        cen_expand = centroids.unsqueeze(0).expand((bs, n_clusters, n_hidden))
        prod_expec = torch.mean(
            z_expand * cen_expand, dim=2
        )  ## (bs , n_clusters)
        pearson_corr = (prod_expec - mean_z * mean_cen) / (std_z * std_cen)
        return torch.sqrt(2 * (1 - pearson_corr))
'''

'''cluster_num < dataset1+dataset2
def compute_dominant_small(preds_list,gt_list):
    import numpy as np
    from collections import defaultdict
    updata_preds_list = []
    for i in range(len(preds_list)):
        print('=====================================================')
        cluster_to_labels = defaultdict(list)
        preds = preds_list[i]
        gts = gt_list[i]
        preds = np.argmax(preds, axis=1)
        for pred, gt in zip(preds, gts):
            cluster_to_labels[pred].append(gt)
        
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
        updated_preds = [dominant_labels[cluster] for cluster in preds]
        updata_preds_list.append(updated_preds.copy())
    update_preds_all = np.concatenate(updata_preds_list,axis=0)
    return update_preds_all
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
