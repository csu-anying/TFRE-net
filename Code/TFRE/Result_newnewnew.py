import sys
import numpy as np, os
import torch
from sklearn.metrics import *
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc, f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, classification_report, roc_auc_score, precision_score, recall_score
import torch

def ptbxl_Result(class_num, y_pred, y_test, baseline=0.5, prin=False, k=3):
    # 确保 y_test 和 y_pred 是 numpy 数组
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # 初始化指标列表
    precision_list = []
    recall_list = []
    pr_auc_list = []
    f1_list = []
    f2_list = []  # 存储 F2 分数
    balanced_accuracy_list = []  # 存储平衡准确率
    class_accuracy_list = []  # 存储每个类别的准确率
    auc_list = []  # 存储每个类别的 AUC

    for i in range(class_num):
        # 计算 Precision-Recall 曲线
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
        # 计算 PR AUC
        pr_auc = sklearn_auc(recall, precision)

        # 计算 F1 分数
        binary_predictions = torch.round(torch.tensor(y_pred[:, i]))
        f1 = f1_score(y_test[:, i], binary_predictions.numpy(), zero_division=0)

        # 计算 F2 分数
        f2 = f2_score(y_test[:, i], binary_predictions.numpy(), zero_division=0)

        # 计算平衡准确率（Balanced Accuracy）
        recall_per_class = recall_score(y_test[:, i], binary_predictions.numpy(), zero_division=0)
        balanced_accuracy_list.append(recall_per_class)

        # 找到正例的索引
        positive_indices = np.where(y_test[:, i] == 1)[0]

        # 计算正例的准确率：正例中预测正确的比例
        if len(positive_indices) > 0:
            class_acc = accuracy_score(y_test[positive_indices, i], binary_predictions.numpy()[positive_indices])
        else:
            class_acc = 0  # 如果没有正例，则准确率设为0

        # 计算 AUC
        auc = roc_auc_score(y_test[:, i], y_pred[:, i])

        # 将结果添加到列表中
        precision_list.append(precision_score(y_test[:, i], binary_predictions.numpy(), zero_division=0))
        recall_list.append(recall_score(y_test[:, i], binary_predictions.numpy(), zero_division=0))
        pr_auc_list.append(pr_auc)
        f1_list.append(f1)
        f2_list.append(f2)
        class_accuracy_list.append(class_acc)
        auc_list.append(auc)

    # 计算平均值
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_pr_auc = np.mean(pr_auc_list)
    avg_f1 = np.mean(f1_list)
    avg_f2 = np.mean(f2_list)  # 计算平均 F2 分数
    avg_balanced_accuracy = np.mean(balanced_accuracy_list)  # 计算平衡准确率
    avg_auc = np.mean(auc_list)  # 计算 Macro AUC

    # 计算总体准确率
    binary_predictions = torch.round(torch.tensor(y_pred))

    # Subset accuracy (每个样本标签全对才算对)
    acc = accuracy_score(y_test, binary_predictions.numpy())


    # 打印分类报告
    if prin:
        print(classification_report(y_test, binary_predictions.numpy(), zero_division=0))

    return acc, avg_pr_auc, avg_f1, avg_precision, avg_recall, avg_auc, avg_balanced_accuracy, avg_f2, class_accuracy_list

def f2_score(y_true, y_pred, zero_division=0):
    """
    计算 F2 分数：它对召回率赋予更高的权重。
    """
    precision = precision_score(y_true, y_pred, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, zero_division=zero_division)
    if precision + recall == 0:
        return 0
    return 5 * (precision * recall) / (4 * precision + recall)


def Sen(con_mat, n=4):
    sen = []
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)

    return sen


def Spe(con_mat, n=4):
    spe = []
    temp = 0
    for i in range(n):
        temp += con_mat[i][i]
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = (temp - tp) / (tn + fp)
        # print(spe1)
        spe.append(spe1)

    return spe


def ACC(con_mat, n=4):
    acc = []
    number = np.sum(con_mat[:, :])
    temp = 0
    for i in range(n):
        temp += con_mat[i][i]
    acc = temp / number
    return acc


# def ptb_Result(class_num, y_pred, y_test):
#     y_pred = np.argmax(y_pred, axis=1)
#     # if (int(class_num) == 7):
#     #     target_names = ['AMI', 'ASMI', 'ALMI', 'IMI', 'ILMI', 'other', 'NORM']
#     # else:
#     #     print("num_class error!")
#     #     sys.exit()
#
#     con_mat = confusion_matrix(y_test, y_pred)  # 把输入转变为混淆矩阵
#     n = con_mat.shape[0]
#     sen = Sen(con_mat, n)
#     spe = Spe(con_mat, n)
#     acc = ACC(con_mat, n)
#     return np.sum(sen) / n, np.sum(spe) / n, acc

from sklearn.metrics import (
    precision_recall_curve, auc as sklearn_auc, f1_score, fbeta_score,
    precision_score, recall_score, accuracy_score, roc_auc_score, classification_report
)

def ptb_Result(class_num, y_pred, y_true, prin=False):
    """
    改进后的 ptb_Result：
    - 对齐 ptbxl_Result 的指标计算逻辑
    - 保留原返回顺序和输出结构
    - 修正每类准确率、AUC、平衡准确率的计算方式
    - 适用于多标签或多分类任务（基于 one-hot）
    """

    # ----------- 数据准备 -----------
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 若输入为单列标签（整数标签），则转 one-hot
    if y_true.ndim == 1 or y_true.shape[1] != class_num:
        y_true_oh = np.zeros((len(y_true), class_num))
        y_true_oh[np.arange(len(y_true)), y_true.astype(int)] = 1
        y_true = y_true_oh

    # ----------- 初始化各类指标列表 -----------
    precision_list, recall_list, pr_auc_list = [], [], []
    f1_list, f2_list, auc_list = [], [], []
    balanced_accuracy_list, class_accuracy_list = [], []

    # ----------- 每个类别逐一计算 -----------
    for i in range(class_num):
        y_true_i = y_true[:, i]
        y_prob_i = y_pred[:, i]

        # ---- Precision-Recall AUC ----
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_i, y_prob_i)
        pr_auc = sklearn_auc(recall_curve, precision_curve)

        # ---- 二值预测 (≥0.5) ----
        binary_pred = torch.round(torch.tensor(y_prob_i)).numpy()

        # ---- 分类指标 ----
        prec = precision_score(y_true_i, binary_pred, zero_division=0)
        rec = recall_score(y_true_i, binary_pred, zero_division=0)
        f1 = f1_score(y_true_i, binary_pred, zero_division=0)
        f2 = fbeta_score(y_true_i, binary_pred, beta=2, zero_division=0)

        # ---- Balanced Accuracy ----
        bal_acc = recall_score(y_true_i, binary_pred, zero_division=0)  # 同 ptbxl_Result 的 recall 近似平衡准确率

        # ---- 每类总体准确率 ----
        class_acc = accuracy_score(y_true_i, binary_pred)

        # ---- ROC AUC (基于概率，而非二值) ----
        try:
            auc_val = roc_auc_score(y_true_i, y_prob_i)
        except ValueError:
            auc_val = 0.0

        # ---- 存储 ----
        precision_list.append(prec)
        recall_list.append(rec)
        pr_auc_list.append(pr_auc)
        f1_list.append(f1)
        f2_list.append(f2)
        auc_list.append(auc_val)
        balanced_accuracy_list.append(bal_acc)
        class_accuracy_list.append(class_acc)

    # ----------- 平均指标 -----------
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_pr_auc = np.mean(pr_auc_list)
    avg_f1 = np.mean(f1_list)
    avg_f2 = np.mean(f2_list)
    avg_balanced_accuracy = np.mean(balanced_accuracy_list)
    avg_auc = np.mean(auc_list)

    # ----------- 总体准确率（subset accuracy） -----------
    binary_predictions = torch.round(torch.tensor(y_pred))
    acc = accuracy_score(y_true, binary_predictions.numpy())

    # ----------- 打印报告 -----------
    if prin:
        print("\nClassification Report:")
        print(classification_report(y_true, binary_predictions.numpy(), zero_division=0))

    # ----------- 返回结果（与原函数顺序一致） -----------
    return (
        acc,                   # Overall accuracy
        avg_pr_auc,            # Mean PR-AUC
        avg_f1,                # Mean F1-score
        avg_precision,         # Mean Precision
        avg_recall,            # Mean Recall
        avg_auc,               # Mean AUC
        avg_balanced_accuracy, # Mean Balanced Accuracy
        avg_f2,                # Mean F2-score
        class_accuracy_list    # Class-wise accuracy
    )