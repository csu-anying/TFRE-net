import sys
import numpy as np, os
import torch
from sklearn.metrics import *
from sklearn.metrics import precision_recall_curve, auc as sklearn_auc, f1_score, accuracy_score, precision_score, recall_score, classification_report
# def ptbxl_Result(class_num, y_pred, y_test, baseline=0.5):
#     # # 计算roc_auc
#     # auc = roc_auc_score(y_test, y_pred)
#     # 计算 Precision-Recall 曲线
#     precision, recall, _ = precision_recall_curve(y_test, y_pred)
#     # 计算 PR AUC
#     pr_auc = auc(recall, precision)
#
#     binary_predictions = torch.round(torch.tensor(y_pred))
#     acc = accuracy_score(y_test, binary_predictions.numpy())
#
#     return acc, pr_auc


def ptbxl_Result(class_num, y_pred, y_test, baseline=0.5,prin = False):
    # 确保 y_test 和 y_pred 是 numpy 数组
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # 初始化指标列表
    precision_list = []
    recall_list = []
    pr_auc_list = []
    f1_list = []
    class_accuracy_list = []  # 存储每个类别的准确率

    for i in range(class_num):
        # 计算 Precision-Recall 曲线
        print(f"y_pred shape: {y_pred.shape}")  # 打印预测值的形状
        print(f"y_test shape: {y_test.shape}")  # 打印真实标签的形状
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
        # 计算 PR AUC
        pr_auc = sklearn_auc(recall, precision)

        # 计算 F1 分数
        binary_predictions = torch.round(torch.tensor(y_pred[:, i]))
        f1 = f1_score(y_test[:, i], binary_predictions.numpy(), zero_division=0)

        # 找到正例的索引
        positive_indices = np.where(y_test[:, i] == 1)[0]

        # 计算正例的准确率：正例中预测正确的比例
        if len(positive_indices) > 0:
            class_acc = accuracy_score(y_test[positive_indices, i], binary_predictions.numpy()[positive_indices])
        else:
            class_acc = 0  # 如果没有正例，则准确率设为0

        # 将结果添加到列表中
        precision_list.append(precision_score(y_test[:, i], binary_predictions.numpy(), zero_division=0))
        recall_list.append(recall_score(y_test[:, i], binary_predictions.numpy(), zero_division=0))
        pr_auc_list.append(pr_auc)
        f1_list.append(f1)
        class_accuracy_list.append(class_acc)

    # 计算平均值
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_pr_auc = np.mean(pr_auc_list)
    avg_f1 = np.mean(f1_list)

    # 计算总体准确率
    binary_predictions = torch.round(torch.tensor(y_pred))
    acc = accuracy_score(y_test, binary_predictions.numpy())

    # 打印分类报告
    if prin:
        print(classification_report(y_test, binary_predictions.numpy(), zero_division=0))

    return acc, avg_pr_auc, avg_f1, avg_precision, avg_recall, class_accuracy_list

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

def one_hot_encode(y_test, class_num):
    # 将 y_test 转换为独热编码
    y_test_one_hot = np.eye(class_num)[y_test]
    return y_test_one_hot

def ptb_Result(class_num, y_pred, y_test, baseline=0.5,prin = False):
    y = y_pred
    y_pred = np.argmax(y_pred, axis=1)
    con_mat = confusion_matrix(y_test, y_pred)
    n = con_mat.shape[0]
    acc = ACC(con_mat, n)
    # 确保 y_test 和 y_pred 是 numpy 数组
    y_test = np.array(y_test)
    y_test = one_hot_encode(y_test, class_num)
    y_pred = np.array(y)

    # 初始化指标列表
    precision_list = []
    recall_list = []
    pr_auc_list = []
    f1_list = []
    class_accuracy_list = []  # 存储每个类别的准确率

    for i in range(class_num):
        # 计算 Precision-Recall 曲线
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
        # 计算 PR AUC
        pr_auc = sklearn_auc(recall, precision)

        # 计算 F1 分数
        binary_predictions = torch.round(torch.tensor(y_pred[:, i]))
        f1 = f1_score(y_test[:, i], binary_predictions.numpy(), zero_division=0)

        # 找到正例的索引
        positive_indices = np.where(y_test[:, i] == 1)[0]

        # 计算正例的准确率：正例中预测正确的比例
        if len(positive_indices) > 0:
            class_acc = accuracy_score(y_test[positive_indices, i], binary_predictions.numpy()[positive_indices])
        else:
            class_acc = 0  # 如果没有正例，则准确率设为0

        # 将结果添加到列表中
        precision_list.append(precision_score(y_test[:, i], binary_predictions.numpy(), zero_division=0))
        recall_list.append(recall_score(y_test[:, i], binary_predictions.numpy(), zero_division=0))
        pr_auc_list.append(pr_auc)
        f1_list.append(f1)
        class_accuracy_list.append(class_acc)

    # 计算平均值
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_pr_auc = np.mean(pr_auc_list)
    avg_f1 = np.mean(f1_list)

    # 计算总体准确率
    binary_predictions = torch.round(torch.tensor(y_pred))

    # 打印分类报告
    if prin:
        print(classification_report(y_test, binary_predictions.numpy(), zero_division=0))

    return acc, avg_pr_auc, avg_f1, avg_precision, avg_recall, class_accuracy_list