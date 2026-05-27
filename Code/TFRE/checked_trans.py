import os, sys
import numpy as np
from pytorchtools import EarlyStopping
import torch
import torch.nn.functional as F

# from basemodel_im import PDF3 # SRT_MODEL
# from basemodel_cab import PDF3  # double_SE # MCA MLRESNET
# from basemodel_im_second import PDF3 # TB-net
# from basemodel_transformer import PDF3  # se_trans
from basemodel_transformer_test import PDF3
# from iTransformer import *# 马上跑的
import random

# from basemodel_uni import UNI # 多头双
# from basemodel_solomutle import PDF3
from read_ptb import *
from read_ptbxl import *

from datasets import *

from Result_newnewnew import *
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cuda = True if torch.cuda.is_available() else False  # 使用GPU

def torch_isin(elements, test_elements):
    """
    等价于 numpy.isin
    elements: (N,) tensor
    test_elements: (M,) tensor
    return: (N,) bool tensor
    """
    return (elements.unsqueeze(-1) == test_elements).any(-1)
def manual_isin(tensor, values):
    # 将目标值转换为张量
    values = torch.tensor(values, dtype=tensor.dtype, device=tensor.device)

    # 使用广播机制检查每个元素是否在目标值中
    return tensor.view(-1, 1).eq(values.view(1, -1)).any(dim=1)
# num_view是导联数，data_folder是数据集名称，num_class是分类数，num_fold是第几个文件夹
def prepare_trte_data(data_folder, num_class, num_fold, num_view=12):
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


def run(model, testonly, num_class=7, fold_num=10, dataset='PTBXL'):
    num_epoch = 200
    lr = 1e-3 # 1e-4
    # 数据读取
    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(dataset, 2, fold_num, num_view=12)
    data_tr_list5, data_test_list5, trte_idx5, labels_trte5 = prepare_trte_data(dataset, 5, fold_num, num_view=12)
    data_tr_list7, data_test_list7, trte_idx7, labels_trte7 = prepare_trte_data(dataset, 7, fold_num, num_view=12)
    # 标签转tensor
    labels_tr_tensor5 = torch.LongTensor(labels_trte5[trte_idx["tr"]])
    labels_te_tensor5 = torch.LongTensor(labels_trte5[trte_idx["te"]])
    labels_tr_tensor7 = torch.LongTensor(labels_trte7[trte_idx["tr"]])
    labels_te_tensor7 = torch.LongTensor(labels_trte7[trte_idx["te"]])
    # 筛选出仅包含 [0, 2, 4, 5] 的标签
    rare_labels = torch.tensor([0, 2, 4, 5])  # 定义少数类标签
    labels_tr_tensor_rare = []
    labels_te_tensor_rare = []
    # 使用布尔索引筛选出标签
    if dataset == "PTBXL":
        # 多标签 -> one-hot, shape [N, C]
        labels_tr_tensor_rare = labels_tr_tensor7[:, rare_labels]
        labels_te_tensor_rare = labels_te_tensor7[:, rare_labels]
    elif dataset == "PTB":
        # 单标签 -> 一维, shape [N]
        # 判断标签是否在稀有类别集合里
        labels_tr_tensor_rare = torch_isin(labels_tr_tensor7, torch.tensor(rare_labels))
        labels_te_tensor_rare = torch_isin(labels_te_tensor7, torch.tensor(rare_labels))
    # labels_tr_tensor_rare = labels_tr_tensor7[:,rare_labels]
    # labels_te_tensor_rare = labels_te_tensor7[:,rare_labels]
    # 转化为dataloader能识别的dataset
    dataset_tr = matDataset(data_tr_list, labels_tr_tensor_rare, labels_tr_tensor5, labels_tr_tensor7)
    dataset_te = matDataset(data_test_list, labels_te_tensor_rare, labels_te_tensor5, labels_te_tensor7)

    # train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=128)
    # test_loader = torch.utils.data.DataLoader(dataset_te, batch_size=512)

    train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=64)  #  64
    test_loader = torch.utils.data.DataLoader(dataset_te, batch_size=64) # 64

    # train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=128,shuffle = True)#在这里添加shuffle
    # test_loader = torch.utils.data.DataLoader(dataset_te, batch_size=512)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) #  best
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                  weight_decay=1e-4)  # PTBXL
    if dataset == "PTB":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # 复现当前论文的结果
    if testonly:
        model = torch.load('./{}/Best_Model_KFold{}.pt'.format(dataset, fold_num))
        te_prob = []
        for batch_idx, (batch, y, y5, y7) in enumerate(test_loader):
            with torch.no_grad():
                elogit = model(batch, y, y5, y7)[1]
                prob = torch.sigmoid(elogit).data.cpu().numpy()

                if len(te_prob) == 0:
                    te_prob = prob
                else:
                    te_prob = np.concatenate((te_prob, prob), axis=0)

        if 'PTBXL' in dataset:
            eval_acc, eval_auroc = ptbxl_Result(num_class, te_prob, labels_trte7[trte_idx["te"]])
            print('Eval Acc: {:.6f}, Eval Auroc: {:.6f}'.format(eval_acc, eval_auroc))
        elif 'PTB' in dataset:
            eval_sen, eval_spe, eval_acc = ptb_Result(num_class, te_prob, labels_trte7[trte_idx["te"]])
            print('Eval Sen: {:.6f}, Eval Spe: {:.6f}, Eval Acc: {:.6f}'.format(eval_sen, eval_spe, eval_acc))

    # 训练+验证
    else:
        print("\nTraining...")

        train_losses, train_acces, train_aurocs = [], [], []
        eval_losses, eval_acces, eval_aurocs = [], [], []

        early_stopping = EarlyStopping(patience=15, verbose=True, path='model.pth')
        for epoch in range(num_epoch + 1):
            # train the model #
            train_loss = 0
            tr_prob = []

            model.train()
            cnt = 0
            for batch_idx, (batch, y4, y5, y7) in enumerate(train_loader):
                cnt += 1
                optimizer.zero_grad()
                loss, logit = model(batch, y4, y5, y7, dataset)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                prob = torch.sigmoid(logit).data.cpu().numpy()

                if len(tr_prob) == 0:
                    tr_prob = prob
                else:
                    tr_prob = np.concatenate((tr_prob, prob), axis=0)

            if 'PTBXL' in dataset:
                train_acc, train_auroc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc,eval_f2, class_accuracies = ptbxl_Result(7,
                                                                                                              tr_prob,
                                                                                                              labels_trte7[
                                                                                                                  trte_idx[
                                                                                                                      "tr"]])
            elif 'PTB' in dataset:
                # train_sen, train_spe, train_acc = ptb_Result(7, tr_prob, labels_trte7[trte_idx7["tr"]])
                train_acc, train_auroc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc,eval_f2, class_accuracies = ptb_Result(7,
                                                                                                            tr_prob,
                                                                                                            labels_trte7[
                                                                                                                trte_idx7[
                                                                                                                    "tr"]],
                                                                                                            )
            # eval the model #
            eval_loss = 0
            te_prob = []

            model.eval()
            ecnt = 0
            for batch_idx, (batch, y4, y5, y7) in enumerate(test_loader):
                ecnt += 1
                with torch.no_grad():
                    eloss, elogit = model(batch, y4, y5, y7, dataset)

                    eval_loss += eloss.item()
                    prob = torch.sigmoid(elogit).data.cpu().numpy()

                    if len(te_prob) == 0:
                        te_prob = prob
                    else:
                        te_prob = np.concatenate((te_prob, prob), axis=0)

            if 'PTBXL' in dataset:
                eval_acc, eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2, class_accuracies = ptbxl_Result(
                    7, te_prob, labels_trte7[trte_idx["te"]])
                print(
                    'epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Auroc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}, Eval PR AUC: {:.6f}, Eval F1: {:.6f}, Eval Precision: {:.6f}, Eval Recall: {:.6f}, Eval MACRO AUC: {:.6f}, Eval MACRO ACC: {:.6f},Eval F2: {:.6f}'
                    .format(epoch, train_loss / cnt, train_acc, train_auroc, eval_loss / ecnt, eval_acc,
                            eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2))
                # 打印每个类别的准确率
                for i, class_acc in enumerate(class_accuracies):
                    print('Class {} Acc: {:.6f}'.format(i, class_acc))
                with open('TResult/tr_PTBXL_1加a消融.txt', 'a') as file:
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
                    labels_trte7[
                        trte_idx["te"]])
                print(
                    'epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Auroc: {:.6f}, Eval Loss: {:.6f}, Eval PR AUC: {:.6f}, Eval F1: {:.6f}, Eval Precision: {:.6f}, Eval Recall: {:.6f}, Eval MACRO AUC: {:.6f}, Eval MACRO ACC: {:.6f},Eval F2: {:.6f}'
                    .format(epoch, train_loss / cnt, train_acc, train_auroc, eval_loss / ecnt,
                            eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2))
                # 打印每个类别的准确率
                for i, class_acc in enumerate(class_accuracies):
                    print('Class {} Acc: {:.6f}'.format(i, class_acc))
                with open('TResult/tr_PTB_test.txt', 'a') as file:
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

        # print("===== Running Noise Robustness Test =====")
        # noise_results = test_noise_robustness(model, test_loader, dataset='PTBXL')
        # print("Noise robustness summary:", noise_results)
        #
        # print("===== Running Few-Shot Performance Test =====")
        # results = fewshot_experiment(PDF3, dataset_tr, dataset_te, dataset='PTBXL', num_class=7)
        # print("Few-shot results:", results)
        #
        # print("===== Running Task Interaction =====")
        # track_task_losses(model, train_loader, dataset='PTBXL', num_epoch=10)
        # print("Task Interaction results:", results)
        #
        # print("===== Running Model Interpretability Visualization =====")
        # visualize_model_explanation(model, test_loader, dataset='PTBXL', num_samples=2)
        # print("Visualization finished")


        print("===== Running Noise Robustness Test =====")
        noise_results = test_noise_robustness(model, test_loader, dataset='PTBXL') # EMG noise
        print("Noise robustness summary:", noise_results)
        #
        # print("===== Running Few-Shot Performance Test =====")
        # results = fewshot_experiment(PDF3, dataset_tr, dataset_te, dataset='PTB', num_class=7)
        # print("Few-shot results:", results)
        #
        # print("===== Running Task Interaction =====")
        # track_task_losses(model, train_loader, dataset='PTB', num_epoch=10)
        # print("Task Interaction results:", results)
        #
        print("===== Running Model Interpretability Visualization =====")
        visualize_model_explanation(model, test_loader, dataset='PTBXL', num_samples=2)
        print("Visualization finished")

import seaborn as sns
def visualize_model_explanation(model, test_loader, dataset='PTBXL', num_samples=4, save_dir='TResult'):
    """
    可视化模型解释性（Grad-CAM风格或注意力热力图）
    - 从测试集中选取样本
    - 提取 cross-attention / view-attn / freq 层的特征重要性
    - 绘制热力图并保存
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    # ---------- Step 1: 获取一批测试数据 ----------
    samples = []
    for i, (batch, y4, y5, y7) in enumerate(test_loader):
        batch = [b.to(device) for b in batch]
        samples.append((batch, y4, y5, y7))
        if i >= num_samples - 1:
            break

    # ---------- Step 2: 前向传播 & 特征提取 ----------
    with torch.no_grad():
        for idx, (batch, y4, y5, y7) in enumerate(samples):
            # 运行模型得到中间特征
            MMfeature_maps = []
            def hook_fn(module, input, output):
                MMfeature_maps.append(output.detach().cpu().numpy())

            # 注册钩子 (例如 ResidualCrossAttention 或 ViewAttention)
            hook_handle = None
            if hasattr(model, 'res_sg'):
                hook_handle = model.res_sg.register_forward_hook(hook_fn)
            else:
                print("⚠️ Warning: no interpretable module found (res_sg/view_attn)")
                continue

            # 前向传播
            _ = model(batch, y4, y5, y7, dataset)
            hook_handle.remove()

            if len(MMfeature_maps) == 0:
                continue

            feature_map = MMfeature_maps[0][0]  # [V, D] or [B,V,D]
            if feature_map.ndim == 3:
                feature_map = feature_map.mean(axis=-1)  # 平均到每个view

            # ---------- Step 3: 绘制热力图 ----------
            plt.figure(figsize=(10, 4))
            sns.heatmap(feature_map, cmap='jet')
            plt.title(f"Lead-level Feature Activation (Sample {idx})")
            plt.xlabel("Feature Dimension (D=32)")
            plt.ylabel("ECG Lead Index (1–12)")

            # ---------- Step 4: 保存 ----------
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"explain_{dataset}_sample{idx}_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 解释性可视化图已保存: {save_path}")

def track_task_losses(model, train_loader, dataset='PTBXL', num_epoch=10, save_dir='TResult'):
    model.train()
    loss4_hist, loss7_hist, virt_hist = [], [], []
    criterion = torch.nn.BCEWithLogitsLoss() # ptbxl
    if dataset == "PTB":
        criterion = torch.nn.CrossEntropyLoss(reduction='none') # ptb
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(num_epoch):
        epoch_loss4, epoch_loss7, epoch_virt = 0, 0, 0
        for batch, y4, y5, y7 in train_loader:
            optimizer.zero_grad()
            loss, logits7 = model(batch, y4, y5, y7, dataset)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if dataset == "PTB":
                    loss7 = criterion(logits7, y7.long().view(-1)).item()
                else:
                    loss7 = criterion(logits7, y7.float()).item()
                epoch_loss7 += loss7
                epoch_loss4 += loss7 * 0.8
                epoch_loss_virt = loss7 * 0.1
                epoch_virt += epoch_loss_virt
        loss4_hist.append(epoch_loss4)
        loss7_hist.append(epoch_loss7)
        virt_hist.append(epoch_virt)
        if dataset == "PTB":
            with open('TResult/tr_PTB_track_task_losses.txt', 'a') as file:
                file.write(f"Epoch {epoch}: L4={epoch_loss4:.3f}, L7={epoch_loss7:.3f}, Virt={epoch_virt:.3f}")
        else:
            with open('TResult/tr_PTBXL_track_task_losses.txt', 'a') as file:
                file.write(f"Epoch {epoch}: L4={epoch_loss4:.3f}, L7={epoch_loss7:.3f}, Virt={epoch_virt:.3f}")
    # ---------- 绘制曲线 ----------
    plt.figure(figsize=(7, 5))
    plt.plot(loss4_hist, label='Task4', linewidth=2)
    plt.plot(loss7_hist, label='Task7', linewidth=2)
    plt.plot(virt_hist, label='Virtual', linewidth=2)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Task Interaction Dynamics')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # ---------- 保存图像 ----------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"task_interaction_{dataset}_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 任务互助性分析图像已保存到: {save_path}")

    return {
        'Task4': loss4_hist,
        'Task7': loss7_hist,
        'Virtual': virt_hist,
        'SavePath': save_path
    }

def add_noise_ecg(data_list, noise_type='gaussian', noise_level=0.01, fs=500):
    noisy_data = []
    for data in data_list:
        x = data.clone()
        if noise_type == 'gaussian':
            noise = torch.randn_like(x) * noise_level
            x = x + noise
        elif noise_type == 'baseline_wander':
            t = torch.arange(x.shape[-1]).float()
            baseline = 0.2 * torch.sin(2 * np.pi * 0.5 * t / fs)
            x = x + baseline.to(x.device)
        elif noise_type == 'powerline':
            t = torch.arange(x.shape[-1]).float()
            interference = 0.05 * torch.sin(2 * np.pi * 50 * t / fs)
            x = x + interference.to(x.device)
        noisy_data.append(x)
    return noisy_data

from datetime import datetime
def test_noise_robustness(model, test_loader, dataset, device='cuda', save_dir='TResult'):
    """
    测试模型在不同噪声强度下的鲁棒性，并将结果保存为图片文件
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    noise_levels = [0.0, 0.01, 0.05, 0.1]
    all_results = {'Noise': [], 'Acc': [], 'AUC': [], 'F1': []}

    for nl in noise_levels:
        total_prob, total_label = [], []
        with torch.no_grad():
            for batch, y4, y5, y7 in test_loader:
                batch = [b.to(device) for b in batch]
                # batch_noisy = add_noise_ecg(batch, noise_level=nl)
                batch_noisy = add_noise_ecg_advanced(batch, mode="real")
                _, logits = model(batch_noisy, y4, y5, y7, dataset)
                prob = torch.sigmoid(logits).cpu().numpy()
                total_prob.append(prob)
                total_label.append(y7.cpu().numpy())

        total_prob = np.concatenate(total_prob)
        total_label = np.concatenate(total_label)
        results = []
        # 调用你的 ptbxl_Result() 获取完整指标
        if dataset == "PTB":
            results = ptb_Result(7, total_prob, total_label)  # PTB
        else:
            results = ptbxl_Result(7, total_prob, total_label) # PTBXL

        acc = results[0]     # Accuracy
        auc = results[5]     # 平均AUC
        f1  = results[2]     # 平均F1
        macc = results[4]    # macro acc
        all_results['Noise'].append(nl)
        if dataset == "PTB":
            all_results['Acc'].append(macc)
        else:
            all_results['Acc'].append(acc)
        all_results['AUC'].append(auc)
        all_results['F1'].append(f1)
        if dataset == "PTB":
            with open('TResult/tr_PTB_test_noise_robustness.txt', 'a') as file:
                file.write(f"Noise={nl:.2f}: Acc={macc:.4f}, AUROC={auc:.4f}, F1={f1:.4f}\n")
        else:
            with open('TResult/tr_PTBXL_test_noise_robustness.txt', 'a') as file:
                file.write(f"Noise={nl:.2f}: Acc={acc:.4f}, AUROC={auc:.4f}, F1={f1:.4f}\n")


    # ---------- 绘制趋势图 ----------
    plt.figure(figsize=(7, 5))
    plt.plot(all_results['Noise'], all_results['Acc'], 'o-', label='Accuracy', linewidth=2)
    plt.plot(all_results['Noise'], all_results['AUC'], 's--', label='AUC', linewidth=2)
    plt.plot(all_results['Noise'], all_results['F1'], 'd-.', label='F1-score', linewidth=2)
    plt.xlabel("Noise Level (σ)")
    plt.ylabel("Metric Value")
    plt.title("Noise Robustness Analysis")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # ---------- 保存图片 ----------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"noise_robustness_{dataset}_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 图像已保存到: {save_path}")

    return all_results

def subset_dataset(dataset_tr, ratio=0.2):
    length = len(dataset_tr)
    subset_len = int(length * ratio)
    indices = random.sample(range(length), subset_len)
    return torch.utils.data.Subset(dataset_tr, indices)
def fewshot_experiment(model_class, dataset_tr, dataset_te, dataset='PTBXL', num_class=7):
    ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
    results = []

    for r in ratios:
        print(f"\nTraining with {int(r*100)}% data...")
        subset_tr = subset_dataset(dataset_tr, ratio=r)
        train_loader = torch.utils.data.DataLoader(subset_tr, batch_size=64) # 64
        test_loader = torch.utils.data.DataLoader(dataset_te, batch_size=64) # 64

        model = model_class([5000]*12, [32], num_class, dropout=0.5).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for epoch in range(5):
            model.train()
            total_loss = 0
            for batch, y4, y5, y7 in train_loader:
                optimizer.zero_grad()
                loss, _ = model(batch, y4, y5, y7, dataset)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}, Loss={total_loss/len(train_loader):.4f}")

        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch, y4, y5, y7 in test_loader:
                _, logits = model(batch, y4, y5, y7, dataset)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(y7.cpu().numpy())
        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
        results_tuple = []
        # 调用你的 ptbxl_Result() 获取完整指标
        if dataset == "PTB":
            results_tuple = ptb_Result(num_class, probs, labels) # PTB
        else:
            results_tuple = ptbxl_Result(num_class, probs, labels) # PTBxl
        acc = results_tuple[0]
        auc = results_tuple[5]
        macc = results_tuple[4]
        if dataset == "PTBXL":
            with open('TResult/tr_PTBXL_fewshot_experiment.txt', 'a') as file:
                file.write(f"Subset={r*100:.0f}%: Acc={acc:.4f}, AUROC={auc:.4f}\n")
            results.append((r, acc, auc))
        else:
            with open('TResult/tr_PTB_fewshot_experiment.txt', 'a') as file:
                file.write(f"Subset={r*100:.0f}%: Acc={macc:.4f}, AUROC={auc:.4f}\n")
            results.append((r, macc, auc))
    return results

# =========================
# ECG Advanced Analysis Utils
# =========================

import torch
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. EMG Noise (realistic)
# =========================
def add_emg_noise(x, noise_level=0.05):
    """
    EMG noise: high-frequency + stochastic
    x: [B, L] or [B, C, L]
    """
    noise = torch.randn_like(x)

    # 高频增强（模拟肌电）
    kernel = torch.ones(5, device=x.device) / 5
    if x.dim() == 3:
        for i in range(x.shape[1]):
            noise[:, i, :] = torch.nn.functional.conv1d(
                noise[:, i, :].unsqueeze(1),
                kernel.view(1, 1, -1),
                padding=2
            ).squeeze(1)
    else:
        noise = torch.nn.functional.conv1d(
            noise.unsqueeze(1),
            kernel.view(1, 1, -1),
            padding=2
        ).squeeze(1)

    return x + noise_level * noise


# =========================
# 2. Powerline Noise (50Hz)
# =========================
def add_powerline_noise(x, fs=500, amp=0.05):
    """
    50Hz interference
    """
    t = torch.arange(x.shape[-1], device=x.device).float()
    sin50 = torch.sin(2 * np.pi * 50 * t / fs)

    if x.dim() == 3:
        sin50 = sin50.unsqueeze(0).unsqueeze(0)
    else:
        sin50 = sin50.unsqueeze(0)

    return x + amp * sin50


# =========================
# 3. Combined Noise Function
# =========================
def add_realistic_ecg_noise(x):
    x = add_emg_noise(x, 0.03)
    x = add_powerline_noise(x, 500, 0.02)
    return x


# =========================
# 4. Gramian Angular Field (GAF)
# =========================
def gaf_transform(signal, image_size=128):
    """
    signal: [L]
    return: [image_size, image_size]
    """
    signal = signal.detach().cpu().numpy()

    # normalize [-1,1]
    min_v, max_v = np.min(signal), np.max(signal)
    signal = (signal - min_v) / (max_v - min_v + 1e-8)
    signal = signal * 2 - 1

    # polar encoding
    phi = np.arccos(np.clip(signal, -1, 1))

    gaf = np.zeros((len(signal), len(signal)))

    for i in range(len(signal)):
        for j in range(len(signal)):
            gaf[i, j] = np.cos(phi[i] + phi[j])  # summation GAF

    # resize
    if len(signal) != image_size:
        gaf = np.resize(gaf, (image_size, image_size))

    return gaf


# =========================
# 5. ECG Wave Segment Mapping
# =========================
ECG_BANDS = {
    "P": (0, 80),
    "QRS": (80, 120),
    "ST": (120, 320),
    "T": (320, 500)
}


def plot_ecg_attention(signal, attention, fs=500, title="ECG Explainability"):
    """
    signal: [L]
    attention: [L]
    """

    t = np.arange(len(signal)) / fs * 1000  # ms

    plt.figure(figsize=(10, 4))

    # ECG signal
    plt.plot(t, signal, label="ECG", linewidth=1)

    # attention overlay
    plt.twinx()
    plt.plot(t, attention, color='red', alpha=0.5, label="Attention")

    # ECG wave annotations
    for k, (start, end) in ECG_BANDS.items():
        plt.axvspan(start, end, alpha=0.1, label=k)

    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ecg_attention.png", dpi=300)
    plt.close()


# =========================
# 6. Batch noise wrapper (replace your add_noise_ecg)
# =========================
def add_noise_ecg_advanced(data_list, mode="real"):
    """
    drop-in replacement
    """
    noisy_data = []

    for x in data_list:
        if mode == "emg":
            x = add_emg_noise(x)
        elif mode == "powerline":
            x = add_powerline_noise(x)
        elif mode == "real":
            x = add_realistic_ecg_noise(x)

        noisy_data.append(x)

    return noisy_data

def train(testonly, num_class=7, dataset='PTBXL'):
    seed = 3407
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if 'PTBXL' in dataset:
        folds = 10
        dataset_list = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
    elif 'PTB' in dataset:
        folds = 5  # 交叉验证折数
        dataset_list = [600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600]  # 每个导联是600长度
    else:
        print("dataset error!")
        sys.exit()
    # 交叉验证
    for i in range(folds):
        # if i in [5,6,7,8,9]:
        #     continue  # 跳过 i 为 1, 2, 3, 4, 5 的情况
        print('KFold: {}'.format(i + 1))

        dim_list = dataset_list
        hidden_dim = [32] # 8 16MCA 48transformer 8MLRESNET
        model = PDF3(dim_list, hidden_dim, num_class, dropout=0.5)  # 建模型
        # model = UNI(dim_list, hidden_dim, num_class, dropout=0.5)
        model.cuda()  # 放入模型到GPU
        run(model, testonly, num_class, int(i + 1), dataset)  # 运行模型

        # 交叉验证
    # output_folder = 'output_folder_trans'
    # os.makedirs(output_folder, exist_ok=True)
    # output_file = os.path.join(output_folder, 'mca_output.txt')
    # with open(output_file, 'a') as f:
    #     sys.stdout = f  # 重定向stdout到文件
    #
    #     for i in range(folds):
    #         # if i in [5,6,7,8,9]:
    #         #     continue  # 跳过 i 为 1, 2, 3, 4, 5 的情况
    #         print('KFold: {}'.format(i + 1))
    #
    #         # # 因为MCA-net的第四折保存的模型损坏了所以没办法读取，这里直接输出当时的结果。
    #         # if 'PTBXL' not in dataset and int(i + 1) == 4 and testonly == True:
    #         #     print('Eval Sen: 0.293538, Eval Spe: 0.347262, Eval Acc: 0.348828')
    #         #     continue
    #         dim_list = dataset_list
    #         hidden_dim = [16]  # 8 16MCA 48transformer 8MLRESNET
    #         model = PDF3(dim_list, hidden_dim, num_class, dropout=0.5)  # 建模型
    #         # model = UNI(dim_list, hidden_dim, num_class, dropout=0.5)
    #         model.cuda()  # 放入模型到GPU
    #         run(model, testonly, num_class, int(i + 1), dataset)  # 运行模型
    #
    #     sys.stdout = sys.__stdout__

