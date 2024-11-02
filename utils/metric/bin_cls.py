import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

key_list = ['ACC', 'AUC', 'MCC', 'Q-value', 'F1', 'F0.5', 'F2', 'SE', 'SP', 'PPV', 'NPV', 'TP', 'FP', 'TN', 'FN']


def search_best_threshold(logits, labels, metric, mode='max', start=1.0, end=-1.0, step=-0.002, **kwargs):
    best_metric, best_metrics, best_t = 0, None, None
    for t in np.arange(start, end, step):
        t = round(t, 3)
        metrics = compute_metrics(logits, labels, threshold=t, only_dict=True, **kwargs)
        if mode == 'max' and metrics[metric] >= best_metric:
            best_metrics = metrics
            best_metric = metrics[metric]
            best_t = t
        if mode == 'min' and metrics[metric] < best_metric:
            best_metrics = metrics
            best_metric = metrics[metric]
            best_t = t

    best_metrics['threshold'] = best_t
    best_metrics = pd.DataFrame(best_metrics, index=[0])
    return best_metrics


def compute_metrics(pred, target, threshold=0.5, softmax=False, only_dict=False, only_df=True):
    '''
    基于torchmetrics.functional实现的二分类指标
    :param pred: 预测结果 (logtis/pred_probs) [B, 2]
    :param target: 实际标签 (label) [B]
    :param threshold: 分类阈值
    :param softmax: 是否需要对pred参数进行softmax归一化
    :param only_dict: 返回的dict是否只需要包含基本指标的键值对就可以了 (即不返回list和dataframe)
    :param only_df: 是否只返回dataframe
    :return: 指标字典 (dict)
    '''
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    if len(pred.shape) == 1:
        pred = pred.reshape([-1, 1])
    pred = pred.softmax(dim=-1)[:, -1] if softmax else pred[:, -1]
    pred_label = (pred >= threshold).long()
    pred = pred.cpu().numpy()
    pred_label.cpu().numpy()
    target = target.cpu().numpy()

    ConfusionMatrix = confusion_matrix(target, pred_label)
    TN = ConfusionMatrix[0][0]
    FP = ConfusionMatrix[0][1]
    FN = ConfusionMatrix[1][0]
    TP = ConfusionMatrix[1][1]
    total = TP + TN + FP + FN
    TP_FN = TP + FN
    TN_FP = TN + FP
    TP_FP = TP + FP
    TN_FN = TN + FN

    ACC = (TP + TN) / total if total != 0 else np.nan
    SE = TP / TP_FN if TP_FN != 0 else np.nan
    SP = TN / TN_FP if TN_FP != 0 else np.nan
    PPV = TP / TP_FP if TP_FP != 0 else np.nan
    NPV = TN / TN_FN if TN_FN != 0 else np.nan
    AUC = roc_auc_score(target, pred)
    MCC = (TP * TN - FP * FN) / np.sqrt(TP_FP * TP_FN * TN_FP * TN_FN) if TP_FP * TP_FN * TN_FP * TN_FN != 0 else np.nan
    F1 = 2 * PPV * SE / (PPV + SE) if PPV + SE != 0 else np.nan
    F05 = 1.25 * PPV * SE / (0.25 * PPV + SE) if 0.25 * PPV + SE != 0 else np.nan
    F2 = 5 * PPV * SE / (4 * PPV + SE) if 4 * PPV + SE != 0 else np.nan
    Q = (SE + SP) / 2

    metric_list = [ACC, AUC, MCC, Q, F1, F05, F2, SE, SP, PPV, NPV, TP, FP, TN, FN]
    metric_list = [round(x, 3) for x in metric_list]
    metric_dict = dict(zip(key_list, metric_list))
    if only_dict:
        return metric_dict
    if only_df:
        metric_df = pd.DataFrame(metric_dict, index=[0])
        return metric_df

    metric_df = pd.DataFrame(metric_dict, index=[0])
    metric_dict['keys'] = key_list
    metric_dict['values'] = metric_list
    metric_dict['df'] = metric_df
    return metric_dict
