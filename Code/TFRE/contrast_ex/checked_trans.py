import os, sys
import numpy as np
from pytorchtools import EarlyStopping
import torch
import torch.nn.functional as F

from basemodel_cab import PDF3  # double_SE # MCA MLRESNET
# from basemodel_transformer import PDF3  # CBAM MRM ECGtrans TGLLnet
# from basemodel_origin import PDF3  # se_trans SAPTSTA
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
    num_epoch = 200 # 200
    lr = 1e-4  # MRM1e-4 ECGTRAN1e-3 CBAM-3
    # 数据读取
    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(dataset, 2, fold_num, num_view=12)
    data_tr_list5, data_test_list5, trte_idx5, labels_trte5 = prepare_trte_data(dataset, 5, fold_num, num_view=12)
    data_tr_list7, data_test_list7, trte_idx7, labels_trte7 = prepare_trte_data(dataset, 7, fold_num, num_view=12)
    # # PTBXL
    # # 标签转tensor
    # labels_tr_tensor5 = torch.LongTensor(labels_trte5[trte_idx["tr"]])
    # labels_te_tensor5 = torch.LongTensor(labels_trte5[trte_idx["te"]])
    # labels_tr_tensor7 = torch.LongTensor(labels_trte7[trte_idx["tr"]])
    # labels_te_tensor7 = torch.LongTensor(labels_trte7[trte_idx["te"]])
    # # 筛选出仅包含 [0, 2, 4, 5] 的标签
    # rare_labels = torch.tensor([0, 2, 4, 5])  # 定义少数类标签
    # # 使用布尔索引筛选出标签
    # labels_tr_tensor_rare = labels_tr_tensor7[:,rare_labels]
    # labels_te_tensor_rare = labels_te_tensor7[:,rare_labels]
    # # 转化为dataloader能识别的dataset
    # dataset_tr = matDataset(data_tr_list, labels_tr_tensor_rare, labels_tr_tensor5, labels_tr_tensor7)
    # dataset_te = matDataset(data_test_list, labels_te_tensor_rare, labels_te_tensor5, labels_te_tensor7)

    # PTB
    # 标签转tensor
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    labels_te_tensor = torch.LongTensor(labels_trte[trte_idx["te"]])
    labels_tr_tensor5 = torch.LongTensor(labels_trte5[trte_idx5["tr"]])
    labels_te_tensor5 = torch.LongTensor(labels_trte5[trte_idx5["te"]])
    labels_tr_tensor7 = torch.LongTensor(labels_trte7[trte_idx5["tr"]])
    labels_te_tensor7 = torch.LongTensor(labels_trte7[trte_idx5["te"]])
    # 转化为dataloader能识别的dataset
    dataset_tr = matDataset(data_tr_list, labels_tr_tensor, labels_tr_tensor5, labels_tr_tensor7)
    dataset_te = matDataset(data_test_list, labels_te_tensor, labels_te_tensor5, labels_te_tensor7)
    # train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=128)
    # test_loader = torch.utils.data.DataLoader(dataset_te, batch_size=512)

    train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=64)  # 在这里添加shuffle 64
    test_loader = torch.utils.data.DataLoader(dataset_te, batch_size=64)  # 64

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
                train_acc, train_auroc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2, class_accuracies = ptbxl_Result(
                    7,
                    tr_prob,
                    labels_trte7[
                        trte_idx[
                            "tr"]])
            elif 'PTB' in dataset:
                # train_sen, train_spe, train_acc = ptb_Result(7, tr_prob, labels_trte7[trte_idx7["tr"]])
                train_acc, train_auroc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2, class_accuracies = ptb_Result(
                    7,
                    tr_prob,
                    labels_trte7[
                        trte_idx[
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
                with open('result/PTBXL/PTBXL_TGLLnet_e4adamw.txt', 'a') as file:
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
                with open('result/PTB/PTB_ML_SGD.txt', 'a') as file:
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
        print("===== Running Noise Robustness Test =====")
        noise_results = test_noise_robustness(model, test_loader, dataset='PTBXL')
        print("Noise robustness summary:", noise_results)

        print("===== Running Few-Shot Performance Test =====")
        results = fewshot_experiment(PDF3, dataset_tr, dataset_te, dataset='PTBXL', num_class=7)
        print("Few-shot results:", results)

import seaborn as sns

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
def test_noise_robustness(model, test_loader, dataset, device='cuda', save_dir='result/PTB'):
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
                batch_noisy = add_noise_ecg(batch, noise_level=nl)
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
            with open('result/PTB/TGL_PTB_test_noise_robustness.txt', 'a') as file:
                file.write(f"Noise={nl:.2f}: Acc={macc:.4f}, AUROC={auc:.4f}, F1={f1:.4f}\n")
        else:
            with open('result/PTB/TGL_PTBXL_test_noise_robustness.txt', 'a') as file:
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

        model = model_class([5000]*12, [48], num_class, dropout=0.5).cuda() # change
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
            with open('result/PTB/TGL_PTBXL_fewshot_experiment.txt', 'a') as file:
                file.write(f"Subset={r*100:.0f}%: Acc={acc:.4f}, AUROC={auc:.4f}\n")
            results.append((r, acc, auc))
        else:
            with open('result/PTB/TGL_PTB_fewshot_experiment.txt', 'a') as file:
                file.write(f"Subset={r*100:.0f}%: Acc={macc:.4f}, AUROC={auc:.4f}\n")
            results.append((r, macc, auc))
    return results

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
        hidden_dim = [48]  # 8 16MCA 48transformer 8MLRESNET 32CBAM None_MRM  # change
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
