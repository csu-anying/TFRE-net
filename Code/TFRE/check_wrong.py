import os, sys
import numpy as np
from pytorchtools import EarlyStopping
import torch
import torch.nn.functional as F
# from basemodel_im import PDF3

# from basemodel_new_judge_tree import PDF3  # 正在跑的(1
from basemodel_judge_tree import PDF3  # 正在跑的(1
# from basemodel_im_second import PDF3  # 正在跑的(1
# from basemodel_cab import PDF3 # double_SE # 正在跑的(2
# from iTransformer import *# 马上跑的

import wandb
import random

from basemodel_uni import UNI  # 多头双
# from basemodel_solomutle import PDF3
from read_ptb import *
from read_ptbxl import *
import shap
# from datasets import *
from dataset import *

# from Result_new import *
from Result_newnewnew import *
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
# import learn2learn as l2l


cuda = True if torch.cuda.is_available() else False  # 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # 数据增强函数，增加噪声,前两个能跑的文件
# def add_noise(data, noise_factor=0.1):
#     noise = noise_factor * np.random.randn(*data.shape)
#     return data + noise
#
# # 增加指定样本的数量
# def augment_data(data_list, labels, target_classes, augmentation_fn, augmentation_factor=1):
#     augmented_data_list = []
#     augmented_labels = []
#
#     for data, label in zip(data_list, labels):
#         augmented_data_list.append(data)
#         augmented_labels.append(label)
#         if any(l in target_classes for l in label):  # 检查标签数组中的任何一个标签是否在目标类别中
#             for _ in range(augmentation_factor):
#                 augmented_data_list.append(augmentation_fn(data))
#                 augmented_labels.append(label)
#
#     return np.array(augmented_data_list), np.array(augmented_labels)
#
# # num_view是导联数，data_folder是数据集名称，num_class是分类数，num_fold是第几个文件夹
# def prepare_trte_data(data_folder, num_class, num_fold, num_view=12):
#     if 'PTBXL' in data_folder:
#         data_tr_list, labels_tr, data_te_list, labels_te = read_ptbxl(num_class, num_fold)
#     elif 'PTB' in data_folder:
#         data_tr_list, labels_tr, data_te_list, labels_te = read_ptb(num_class, num_fold)
#
#     # 对训练数据进行数据增强
#     target_classes = [0, 2, 4, 5]
#     augmentation_factor = 5  # 增强倍数，可以根据需要调整
#
#     augmented_tr_lists = []
#     for i in range(len(data_tr_list)):
#         augmented_data, augmented_labels = augment_data(data_tr_list[i], labels_tr, target_classes, add_noise, augmentation_factor)
#         augmented_tr_lists.append(augmented_data)
#         if i == 0:  # 只需要一次更新 labels_tr
#             labels_tr = augmented_labels
#
#     num_tr = augmented_tr_lists[0].shape[0]
#     num_te = data_te_list[0].shape[0]
#     data_mat_list = []
#     for i in range(num_view):
#         data_mat_list.append(np.concatenate((augmented_tr_lists[i], data_te_list[i]), axis=0))
#
#     data_tensor_list = []
#     for i in range(len(data_mat_list)):
#         data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
#
#     idx_dict = {}
#     idx_dict["tr"] = list(range(num_tr))
#     idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
#     data_train_list = []
#     data_test_list = []
#
#     for i in range(len(data_tensor_list)):
#         data_train_list.append(data_tensor_list[i][idx_dict["tr"]])
#         data_test_list.append(data_tensor_list[i][idx_dict["te"]])
#
#     labels = np.concatenate((labels_tr, labels_te))
#     return data_train_list, data_test_list, idx_dict, labels
#
# def freeze_layers(model, layers_to_freeze):
#     for name, param in model.named_parameters():
#         if any(layer in name for layer in layers_to_freeze):
#             param.requires_grad = False

# 数据增强函数，增加噪声，加了下采样的文件
import random


# 定义拼接和裁剪函数
def concat_and_crop(data, crop_size):
    # 将样本的前后拼接起来
    concatenated_data = np.concatenate((data, data), axis=0)
    # 从拼接后的数据中随机裁剪出一个片段
    start = random.randint(0, concatenated_data.shape[0] - crop_size)
    return concatenated_data[start:start + crop_size]


from scipy.interpolate import interp1d


# 定义部分裁剪上采样函数
def random_crop_and_resize(data, min_beat_length, max_beat_length, target_length):
    # 随机裁剪一个心拍长度的片段
    crop_size = random.randint(min_beat_length, max_beat_length)
    start = random.randint(0, data.shape[0] - crop_size)
    cropped_data = data[start:start + crop_size]

    # 拉长到固定长度
    x_original = np.linspace(0, 1, num=crop_size, endpoint=True)
    x_resized = np.linspace(0, 1, num=target_length, endpoint=True)
    f = interp1d(x_original, cropped_data, kind='linear')
    resized_data = f(x_resized)

    return resized_data


def partial_cropping_resizing(data_list, labels, target_classes, min_beat_length, max_beat_length, target_length,
                              augmentation_factor=1):
    augmented_data_list = []
    augmented_labels = []

    for data, label in zip(data_list, labels):
        augmented_data_list.append(data)
        augmented_labels.append(label)
        if any(tc in label for tc in target_classes):
            for _ in range(augmentation_factor):
                resized_data = random_crop_and_resize(data, min_beat_length, max_beat_length, target_length)
                augmented_data_list.append(resized_data)
                augmented_labels.append(label)

    return np.array(augmented_data_list), np.array(augmented_labels)


# # 数据增强函数，增加噪声
# def add_noise(data, noise_factor=0.1):
#     noise = noise_factor * np.random.randn(*data.shape)
#     return data + noise


def downsample_data(data_list, labels, target_class, factor=0.25):
    downsampled_data_list = []
    downsampled_labels = []

    target_indices = [i for i, label in enumerate(labels) if target_class in label]
    non_target_indices = [i for i, label in enumerate(labels) if target_class not in label]

    downsample_size = int(len(target_indices) * factor)
    downsample_indices = random.sample(target_indices, downsample_size)

    selected_indices = non_target_indices + downsample_indices
    selected_indices.sort()

    for idx in selected_indices:
        downsampled_data_list.append(data_list[idx])
        downsampled_labels.append(labels[idx])

    return np.array(downsampled_data_list), np.array(downsampled_labels)


def augment_data(data_list, labels, target_classes, augmentation_fn, augmentation_factor=1):
    augmented_data_list = []
    augmented_labels = []

    for data, label in zip(data_list, labels):
        augmented_data_list.append(data)
        augmented_labels.append(label)
        if any(l in target_classes for l in label):
            for _ in range(augmentation_factor):
                augmented_data_list.append(augmentation_fn(data))
                augmented_labels.append(label)

    return np.array(augmented_data_list), np.array(augmented_labels)


import pickle
import os


# 定义保存数据的函数
def save_fold_data(data_train_list, data_test_list, labels, idx_dict, num_fold, save_folder="data"):
    # 如果保存文件夹不存在，创建文件夹
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 定义每一折数据的保存路径
    save_path = os.path.join(save_folder, f"fold_{num_fold}_data.pkl")

    # 将训练数据、测试数据、标签、索引字典保存为.pkl文件
    with open(save_path, "wb") as f:
        pickle.dump({
            "data_train_list": data_train_list,
            "data_test_list": data_test_list,
            "labels": labels,
            "idx_dict": idx_dict
        }, f)

    print(f"Data for fold {num_fold} saved to {save_path}")


# 定义加载数据的函数
def load_fold_data(num_fold, load_folder="data"):
    # 定义每一折数据的路径
    load_path = os.path.join(load_folder, f"fold_{num_fold}_data.pkl")

    # 检查文件是否存在
    if os.path.exists(load_path):
        with open(load_path, "rb") as f:
            data = pickle.load(f)

        print(f"Data for fold {num_fold} loaded from {load_path}")
        return data["data_train_list"], data["data_test_list"], data["labels"], data["idx_dict"]
    else:
        print(f"No data found for fold {num_fold} at {load_path}.")
        return None
# def prepare_trte_data(data_folder, num_class, num_fold, num_view=12,pretrain_classes=None):
#     # 先尝试从文件中加载已经保存的数据
#     if 'PTBXL' in data_folder:
#         saved_data = load_fold_data(num_fold)
#
#         if saved_data is not None and pretrain_classes==None:
#             # 如果成功加载到数据，直接返回
#             data_train_list, data_test_list, labels, idx_dict = saved_data
#             return data_train_list, data_test_list, idx_dict, labels
#         elif saved_data is not None:
#             data_train_list, data_test_list, labels, idx_dict = saved_data
#             # 分割训练和测试集的标签
#             num_train = len(data_train_list[0])
#             labels_tr = labels[:num_train]  # 训练集标签
#             labels_te = labels[num_train:]  # 测试集标签
#             # 筛选出符合 pretrain_classes 的训练集数据和标签
#             pretrain_classes_set = set(pretrain_classes)
#             filter_indices = [
#                 i for i, label in enumerate(labels_tr) if pretrain_classes_set.intersection(np.nonzero(label)[0])
#             ]
#
#             # 更新 data_train_list 和 labels_tr
#             data_train_list = [data[filter_indices] for data in data_train_list]
#             labels_tr = labels_tr[filter_indices]
#             # 重新组合训练和测试集标签
#             labels = np.concatenate((labels_tr, labels_te))
#             # 更新 idx_dict：调整训练集索引
#             idx_dict["tr"] = list(range(len(labels_tr)))  # 更新训练集索引
#             idx_dict["te"] = list(range(len(labels_tr), len(labels_tr) + len(labels_te)))  # 保持测试集索引不变
#             # 直接返回筛选后的数据
#             return data_train_list, data_test_list, idx_dict, labels
#
#     # 如果数据不存在，执行原有流程生成数据
#     if 'PTBXL' in data_folder:
#         data_tr_list, labels_tr, data_te_list, labels_te = read_ptbxl(num_class, num_fold)
#     elif 'PTB' in data_folder:
#         data_tr_list, labels_tr, data_te_list, labels_te = read_ptb(num_class, num_fold)
#
#     # 如果是预训练阶段，过滤出指定的类别
#     if pretrain_classes is not None:
#         filter_indices = [i for i, label in enumerate(labels_tr) if label in pretrain_classes]
#         data_tr_list = [data[filter_indices] for data in data_tr_list]
#         labels_tr = labels_tr[filter_indices]
#
#     # Step 1: 下采样类别6
#     majority_class = 6 # PTB[6] PTBXL6
#     downsample_factor = 1
#
#     downsampled_tr_lists = []
#     for i in range(len(data_tr_list)):
#         downsampled_data, downsampled_labels = downsample_data(data_tr_list[i], labels_tr, majority_class,
#                                                                factor=downsample_factor)
#         downsampled_tr_lists.append(downsampled_data)
#         if i == 0:
#             labels_tr = downsampled_labels
#
#     # Step 2: 对所有类别进行部分裁剪上采样
#     target_classes = [0, 2, 4, 5]
#
#     augmented_tr_lists = []
#     # 心率范围为40-150 bpm，采样频率为500Hz
#     min_beat_length = int(60 / 150 * 500)  # 200 样本点
#     max_beat_length = int(60 / 40 * 500)  # 750 样本点
#     target_length = 600  # 目标长度PTB600 PTBXL5000
#     augmentation_factor = 0
#     for i in range(len(downsampled_tr_lists)):
#         augmented_data, augmented_labels = partial_cropping_resizing(downsampled_tr_lists[i], labels_tr,
#                                                                      target_classes, min_beat_length, max_beat_length,
#                                                                      target_length, augmentation_factor)
#         augmented_tr_lists.append(augmented_data)
#         if i == 0:
#             labels_tr = augmented_labels
#
#     num_tr = augmented_tr_lists[0].shape[0]
#     num_te = data_te_list[0].shape[0]
#
#     data_mat_list = []
#     for i in range(num_view):
#         data_mat_list.append(np.concatenate((augmented_tr_lists[i], data_te_list[i]), axis=0))
#
#     data_tensor_list = []
#     for i in range(len(data_mat_list)):
#         data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
#
#     idx_dict = {}
#     idx_dict["tr"] = list(range(num_tr))
#     idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
#
#     data_train_list = []
#     data_test_list = []
#
#     for i in range(len(data_tensor_list)):
#         data_train_list.append(data_tensor_list[i][idx_dict["tr"]])
#         data_test_list.append(data_tensor_list[i][idx_dict["te"]])
#
#     labels = np.concatenate((labels_tr, labels_te))
#
#     # 数据生成完成后，将其保存到文件中
#     if pretrain_classes == None:
#         save_fold_data(data_train_list, data_test_list, labels, idx_dict, num_fold)
#     return data_train_list, data_test_list, idx_dict, labels

def prepare_trte_data(data_folder, num_class, num_fold, num_view=12,pretrain_classes=None):
    if 'PTBXL' in data_folder:
        data_tr_list, labels_tr, data_te_list, labels_te = read_ptbxl(num_class, num_fold)
    # read_ptbxl可以读取数据的训练集和测试集的数据列表和标签列表
    elif 'PTB' in data_folder:
        data_tr_list, labels_tr, data_te_list, labels_te = read_ptb(num_class, num_fold)

    num_tr = data_tr_list[0].shape[0]  # 训练数据集的第一维个数
    num_te = data_te_list[0].shape[0]
    # 下面一段是为了转tensor
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))  # 在每个导联下，数组按照第一维合并训练数据集和测试数据集

    data_tensor_list = []
    for i in range(len(data_mat_list)):  # 把十二导联的总数据集合并集变成tensor
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))

    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))  # 训练数据集的第一维长度列表
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))  # 终止是总数据集的第一维长度列表
    data_train_list = []  # 存放训练数据集
    data_test_list = []  # 存放测试数据集

    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]])
        data_test_list.append(data_tensor_list[i][idx_dict["te"]])

    labels = np.concatenate((labels_tr, labels_te))  # 合并总标签集
    return data_train_list, data_test_list, idx_dict, labels  # tensor格式的训练和测试数据集，训练集和测试集个数索引（用于分开标签，总标签）

def freeze_layers(model, layers_to_freeze):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False


def visualize_attributions(model, test_loader, num_classes, fold_num):
    ig = IntegratedGradients(model.get_only_output)
    model.eval()

    contributions = np.zeros((num_classes, 12))  # 12为导联数
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            try:
                data_list, labels = batch[:-1], batch[-1]
                # for i, data in enumerate(data_list[0]):
                # if not isinstance(data, torch.Tensor):
                #     print(f"data_list[0][{i}] is not a tensor, it is: {type(data)}")
                # else:
                #     print(f"Shape of data_list[0][{i}]: {data.shape}")

                data = torch.stack([torch.stack(sample, dim=0) for sample in data_list], dim=0).to(device)
                # print(f"Shape of data[0]: {data[0].shape}")
                # print(f"data.size(0): {data.size(0)}")
                labels = labels.to(device)
            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f"Batch content: {batch}")
                continue

            outputs = model(data, num_classes, return_only_output=True)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            for i in range(data.size(0)):
                input_data = data[i]  # [12,128,5000]
                input_data = input_data.permute(1, 0, 2)  # [128,12,5000]
                target_class = preds[i].item()
                additional_args = (num_classes, 'PTB', 'CE', True)
                # print("input_data.shape:", input_data.shape)  # [128,12,5000]
                for target_class_i in range(7):
                    attributions = ig.attribute(input_data, target=target_class_i,
                                                additional_forward_args=additional_args,
                                                n_steps=5)
                    attributions = attributions.squeeze().cpu().numpy()
                    # print(f"Attributions shape: {attributions.shape}")
                    mean_attributions = np.mean(attributions, axis=(0))
                    # print(f"mean_attributions shape: {mean_attributions.shape}")
                    channel_importances = np.mean(mean_attributions, axis=1)
                    contributions[target_class_i] += channel_importances

    contributions /= len(test_loader.dataset)

    # for i in range(num_classes):
    #     plt.figure(figsize=(10, 6))
    #     plt.bar(range(12), contributions[i])
    #     plt.xlabel('Lead')
    #     plt.ylabel('Mean Attribution Value')
    #     plt.title(f'Contribution of Each Lead to Class {i}')
    #     plt.savefig(f'class_{i}_contributions.png')
    # Assuming contributions is a NumPy array of shape (12, 7)
    contributions_df = pd.DataFrame(contributions)

    # Normalize each row of the contributions_df DataFrame
    normalized_contributions_df = contributions_df / contributions_df.values.sum()

    # Round each number in the normalized contributions DataFrame to two decimal places
    rounded_normalized_contributions_df = normalized_contributions_df.applymap(lambda x: round(x, 2))
    print("rounded_normalized_contributions_df:")
    print(rounded_normalized_contributions_df)
    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Generate a table-like plot using heatmap
    plt.pcolor(rounded_normalized_contributions_df, cmap='coolwarm')

    # Annotate each cell with the corresponding rounded normalized value
    # for (i, j), z in np.ndenumerate(rounded_normalized_contributions_df):
    #     plt.annotate(str(z), xy=(j, i), ha='center', va='center', fontsize=8)
    for (i, j), z in np.ndenumerate(rounded_normalized_contributions_df):
        # 调整 x 坐标为单元格宽度的一半
        x_center = j + 0.5
        # 调整 y 坐标为单元格高度的一半
        y_center = i + 0.5
        plt.annotate(str(z), xy=(x_center, y_center), ha='center', va='center', fontsize=8)
    # Set axis labels and title
    plt.xlabel('Lead')
    plt.ylabel('Class')
    plt.title('Rounded Normalized Contribution of Each Lead to Each Class')

    # Show colorbar
    plt.colorbar()

    # Save the plot as an image
    plt.savefig('PTB_CONTRI_fold_{}.png'.format(fold_num))


def run(model, dataset='PTBXL', fold_num=10):

    # #MAML
    # meta_lr = 1e-3  # meta-learning rate for MAML
    # optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr, weight_decay=1e-4)

    # 配置阶段设定：第一阶段预训练，第二阶段微调
    # phase_settings = [
    #     # (7, ['FeatureEncoder1','MMClasifier7'], 'CE', [0, 2, 4, 5]),  # 第一阶段：使用少数类 (0, 2)
    #     (7, [], 'CE',None,1e-4),  # 第一阶段：使用少数类 (0, 2)
    #     (7, ['FeatureEncoder1','MMClasifier7'], 'CE', [0, 2, 4, 5],1e-6),  # 第一阶段：使用少数类 (0, 2)
    #     # (7, ['FeatureEncoder2','Part_Clasifie','gcn1', 'gcn1_0', 'gcn2', 'gcn2_0', 'gcn3', 'gcn3_0', 'SE_gcn','gat1',
    #     #      'gat1_0', 'gat2', 'gat2_0', 'BN_gat'], 'CE', [0, 2, 4, 5],1e-6)  # 第二阶段：使用所有类，并冻结部分层
    # ]
    phase_settings = [
        # (7, ['FeatureEncoder1','MMClasifier7'], 'CE', [0, 2, 4, 5]),  # 第一阶段：使用少数类 (0, 2)
        (7, [], 'CE',None,1e-3),  # 第一阶段：使用少数类 (0, 2)
        # (7, [], 'CE', [0, 2, 4, 5],1e-7),  # 第一阶段：使用少数类 (0, 2)
    ]


    auc_max = 0
    ig = IntegratedGradients(model)

    for num_classes, freeze_before, crition, pretrain_classes ,learnr in phase_settings:
        num_epoch = 200
        lr = learnr
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4) # PTBXL
        if dataset == "PTB":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        early_stopping = EarlyStopping(patience=15, verbose=True, path='model.pth', auc_score_max=auc_max)

        # 准备数据
        data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(
            dataset, num_classes, fold_num, num_view=12, pretrain_classes=pretrain_classes

        )
        labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
        labels_te_tensor = torch.LongTensor(labels_trte[trte_idx["te"]])

        # 创建数据加载器
        dataset_tr = matDataset(data_tr_list, labels_tr_tensor)
        dataset_te = matDataset(data_test_list, labels_te_tensor)
        train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=64)
        test_loader = torch.utils.data.DataLoader(dataset_te, batch_size=64)

        # 冻结指定层
        if freeze_before:
            freeze_layers(model, freeze_before)

        print("\n开始训练...")

        # #MAML
        # maml = l2l.algorithms.MAML(model, lr=meta_lr)

        train_losses, train_acces, train_aurocs = [], [], []
        eval_losses, eval_acces, eval_aurocs = [], [], []
        all_labels = []
        all_preds = []

        for epoch in range(num_epoch + 1):
            train_loss = 0
            tr_prob = []

            model.train()
            cnt = 0
            for batch_idx, (batch, labels) in enumerate(train_loader):
                cnt += 1
                optimizer.zero_grad()
                loss, logit = model(batch, num_classes, labels, dataset, crition)
                loss.backward()
                optimizer.step()

                # # MAML,需要注释loss.backward() optimizer.step()
                # loss, logit = model(batch, num_classes, labels, dataset, crition)
                # maml.adapt(loss)  # 使用元学习进行快速适应
                # maml.step()  # 更新模型参数

                train_loss += loss.item()
                prob = torch.sigmoid(logit).data.cpu().numpy()
                if len(tr_prob) == 0:
                    tr_prob = prob
                else:
                    tr_prob = np.concatenate((tr_prob, prob), axis=0)

            if 'PTBXL' in dataset:
                train_acc, train_auroc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2, class_accuracies = ptbxl_Result(
                    7,
                    tr_prob,
                    labels_trte[
                        trte_idx[
                            "tr"]])
            elif 'PTB' in dataset:
                # train_sen, train_spe, train_acc = ptb_Result(7, tr_prob, labels_trte7[trte_idx7["tr"]])
                train_acc, train_auroc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2, class_accuracies = ptb_Result(
                    7,
                    tr_prob,
                    labels_trte[
                        trte_idx[
                            "tr"]],
                    )
            eval_loss = 0
            te_prob = []

            model.eval()
            ecnt = 0
            for batch_idx, (batch, labels) in enumerate(test_loader):
                ecnt += 1
                with torch.no_grad():
                    eloss, elogit = model(batch, num_classes, labels, dataset, crition)

                    eval_loss += eloss.item()
                    prob = torch.sigmoid(elogit).data.cpu().numpy()

                    if len(te_prob) == 0:
                        te_prob = prob
                    else:
                        te_prob = np.concatenate((te_prob, prob), axis=0)

                preds = np.argmax(prob, axis=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
            if 'PTBXL' in dataset:
                eval_acc, eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2, class_accuracies = ptbxl_Result(
                    7, te_prob, labels_trte[trte_idx["te"]])
                print(
                    'epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Auroc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}, Eval PR AUC: {:.6f}, Eval F1: {:.6f}, Eval Precision: {:.6f}, Eval Recall: {:.6f}, Eval MACRO AUC: {:.6f}, Eval MACRO ACC: {:.6f},Eval F2: {:.6f}'
                    .format(epoch, train_loss / cnt, train_acc, train_auroc, eval_loss / ecnt, eval_acc,
                            eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2))
                # 打印每个类别的准确率
                for i, class_acc in enumerate(class_accuracies):
                    print('Class {} Acc: {:.6f}'.format(i, class_acc))
                with open('contrast_ex/result/PTBXL/PTBXL_noneed.txt', 'a') as file:
                    # 写入训练和评估的指标
                    file.write(
                        'epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Auroc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}, Eval PR AUC: {:.6f}, Eval F1: {:.6f}, Eval Precision: {:.6f}, Eval Recall: {:.6f}, Eval MACRO AUC: {:.6f}, Eval MACRO ACC: {:.6f},Eval F2: {:.6f}\n'
                        .format(epoch, train_loss / cnt, train_acc, train_auroc, eval_loss / ecnt, eval_acc,
                                eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2))
                    # 写入每个类别的准确率
                    for i, class_acc in enumerate(class_accuracies):
                        file.write('Class {} Acc: {:.6f}\n'.format(i, class_acc))
                auc_max = early_stopping(eval_loss / ecnt, model, eval_pr_auc, eval_f1)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

            elif 'PTB' in dataset:
                eval_acc, eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2, class_accuracies = ptb_Result(
                    7, te_prob,
                    labels_trte[
                        trte_idx["te"]])
                print(
                    'epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Auroc: {:.6f}, Eval Loss: {:.6f}, Eval PR AUC: {:.6f}, Eval F1: {:.6f}, Eval Precision: {:.6f}, Eval Recall: {:.6f}, Eval MACRO AUC: {:.6f}, Eval MACRO ACC: {:.6f},Eval F2: {:.6f}'
                    .format(epoch, train_loss / cnt, train_acc, train_auroc, eval_loss / ecnt,
                            eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2))
                # 打印每个类别的准确率
                for i, class_acc in enumerate(class_accuracies):
                    print('Class {} Acc: {:.6f}'.format(i, class_acc))
                with open('contrast_ex/result/PTB/PTB_SPG_SGD_onlyshuffle.txt', 'a') as file:
                    # 写入训练和评估的指标
                    file.write(
                        'epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Auroc: {:.6f}, Eval Loss: {:.6f}, Eval PR AUC: {:.6f}, Eval F1: {:.6f}, Eval Precision: {:.6f}, Eval Recall: {:.6f}, Eval MACRO AUC: {:.6f}, Eval MACRO ACC: {:.6f},Eval F2: {:.6f}\n'
                        .format(epoch, train_loss / cnt, train_acc, train_auroc, eval_loss / ecnt,
                                eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2))
                    # 写入每个类别的准确率
                    for i, class_acc in enumerate(class_accuracies):
                        file.write('Class {} Acc: {:.6f}\n'.format(i, class_acc))
                auc_max = early_stopping(eval_loss / ecnt, model, eval_pr_auc, eval_f1)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break
    #     # 加载最佳模型权重
    #     early_stopping.load_best_model(model)
    #     model.eval()
    #     with torch.no_grad():
    #         for batch_idx, (batch, labels) in enumerate(test_loader):
    #             _, logit = model(batch, num_classes, labels, dataset, crition)
    #             prob = torch.sigmoid(logit).data.cpu().numpy()
    #             preds = np.argmax(prob, axis=1)
    #
    #             all_labels.extend(labels.cpu().numpy())
    #             all_preds.extend(preds)
    #
    #     # 将多标签指示器矩阵转换为多类标签
    #     all_labels = np.argmax(all_labels, axis=1)  # PTBXL需要这行
    #     cm = confusion_matrix(all_labels, all_preds)
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #     disp.plot()
    #     plt.savefig('PTBXL2_FUSION_fold_{}.png'.format(fold_num))
    #
    # visualize_attributions(model, test_loader, num_classes, fold_num) # argument 'input' (position 1) must be Tensor, not tuple


def train(testonly, num_class=7, dataset='PTBXL'):
    seed = 3407
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    if 'PTBXL' in dataset:
        folds = 10
        dataset_list = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
    elif 'PTB' in dataset:
        folds = 5
        dataset_list = [600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600]
    else:
        print("dataset error!")
        sys.exit()
    for i in range(folds):
        # if i in [ 0,1, 2,3,5,6,7]:
        # if i in [0,1, 2,3,4, 5, 6,7,8]:
        #     continue  # 跳过 i 为 1, 2, 3, 4, 5 的情况
        print('KFold: {}'.format(i + 1))

        dim_list = dataset_list
        hidden_dim = [32]
        model = PDF3(dim_list, hidden_dim, num_class, dropout=0.5)
        model.cuda()
        run(model, dataset, int(i + 1))
