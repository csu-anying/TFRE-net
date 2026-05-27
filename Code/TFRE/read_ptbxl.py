import pandas as pd
import numpy as np
import wfdb
import random as rn
import ast
import os
import glob
import sys


# 读取ptbxl数据集的文件，这里的df是取了需要折数的原database的文件
def load_ptbxl_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]  # wfdb.rdsamp是专门读取ECG信号的函数
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


# 为norm和MI的二分类问题设置独热编码
# def one_hot_2(y_test):
#     test = []
#     for i in range(len(y_test.values)):
#         if 'NORM' in y_test.values[i]:
#             test.append(0)
#         elif 'MI' in y_test.values[i]:
#             test.append(1)
#     return np.array(test)
def one_hot_2(y_test):
    labels = np.zeros((len(y_test), 2))
    for i in range(len(y_test.values)):
        if len(y_test.values[i]) == 0:
            continue
        if 'NORM' in y_test.values[i]:
            labels[i, 0] = 1
        elif 'MI' in y_test.values[i]:
            labels[i, 1] = 1
    return labels


# 为五分类问题设置独热编码
def one_hot_5(y_test):
    labels = np.zeros((len(y_test), 5))
    for i in range(len(y_test.values)):
        if len(y_test.values[i]) == 0:
            continue
        if 'NORM' in y_test.values[i]:
            labels[i, 4] = 1
        if 'AMI' in y_test.values[i]:
            labels[i, 0] = 1
        if 'ASMI' in y_test.values[i]:
            labels[i, 1] = 1
        if 'ALMI' in y_test.values[i]:
            labels[i, 2] = 1
        if 'other' in y_test.values[i]:
            labels[i, 3] = 1
    return labels


##为七分类问题设置独热编码
def one_hot_7(y_test):
    labels = np.zeros((len(y_test), 7))
    for i in range(len(y_test.values)):
        if len(y_test.values[i]) == 0:
            continue
        if 'NORM' in y_test.values[i]:
            labels[i, 6] = 1
        if 'AMI' in y_test.values[i]:
            labels[i, 0] = 1
        if 'ASMI' in y_test.values[i]:
            labels[i, 1] = 1
        if 'ALMI' in y_test.values[i]:
            labels[i, 2] = 1
        if 'IMI' in y_test.values[i]:
            labels[i, 3] = 1
        if 'ILMI' in y_test.values[i]:
            labels[i, 4] = 1
        if 'other' in y_test.values[i]:
            labels[i, 5] = 1
    return labels


def read_ptbxl(num_class, fold_num):  # 功能就是读取数据和标签，进行标签独热编码
    path = '../'
    sampling_rate = 500
    # SCP-ECG文件是一种用于存储心电图数据的文件格式 #new_ptbxl_database.csv存储了标签
    Y = pd.read_csv(path + 'new_ptbxl_database.csv', index_col='ecg_id', encoding='gb2312')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))  # 取出scp_codes字典中的变量并赋值
    agg_df = pd.read_csv(path + 'scp_statements_257x.csv', index_col=0)  # scp_statements_257x.csv'包含了分类任务里面包含哪几类的信息
    agg_df = agg_df[agg_df.diagnostic == 1]  # 选出用于分类中的标签

    def aggregate_diagnostic(y_dic):  # 选出当前用于分类的的标签
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                if int(num_class) == 2:
                    tmp.append(agg_df.loc[key].diagnostic_class)
                elif int(num_class) == 5:
                    tmp.append(agg_df.loc[key].diagnostic_5class)
                elif int(num_class) == 7:
                    tmp.append(agg_df.loc[key].diagnostic_7class)
        return list(set(tmp))

    Y['diagnostic_class'] = Y.scp_codes.apply(aggregate_diagnostic)  # 筛选标签
    # Y是读取的文件数据，Y.strat_fold是第几折的数据
    strat_fold_train = Y[Y.strat_fold != fold_num]
    X_train = load_ptbxl_data(strat_fold_train, sampling_rate, path)  # 心电图信号数据
    y_train = strat_fold_train.diagnostic_class

    strat_fold_test = Y[Y.strat_fold == fold_num]
    X_test = load_ptbxl_data(strat_fold_test, sampling_rate, path)
    y_test = strat_fold_test.diagnostic_class
    # 独热编码设置，出来是一个（class*输入向量长度）大小的矩阵，里面是独热编码
    if (int(num_class) == 2):
        y_train = one_hot_2(y_train)
        y_test = one_hot_2(y_test)
    elif (int(num_class) == 5):
        y_train = one_hot_5(y_train)
        y_test = one_hot_5(y_test)
    elif (int(num_class) == 7):
        y_train = one_hot_7(y_train)
        y_test = one_hot_7(y_test)

    X_train = np.array(X_train)  # 输入数据，即输入的波形图
    X_test = np.array(X_test)

    def re_format(X):  # 将输入的数据的每个数据转换为数组格式，存放在列表里
        X = X.transpose(2, 0, 1).tolist()
        X_te = []
        for i, a in enumerate(X):  # enumerate用于迭代，对矩阵的每一行操作
            c = []
            for j, b in enumerate(a):  # 这里是对每一行的每个元素操作
                c.append(np.array(b))
            X_te.append(np.array(c))
        return X_te

    return re_format(X_train), y_train, re_format(X_test), y_test
