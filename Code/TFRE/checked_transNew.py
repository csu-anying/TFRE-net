import os, sys
import numpy as np
from pytorchtools import EarlyStopping
import torch
import torch.nn.functional as F
import random
from basemodel_transformer_test import PDF3  # ✅ 这里按你用的模型

from read_ptb import *
from read_ptbxl import *
from datasets import *
from Result_newnewnew import *
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cuda = True if torch.cuda.is_available() else False  # 使用GPU

# === 用来记录 fold 的目标 epoch 结果 ===
fold_metrics = {
    "Eval Acc": [],
    "Eval PR AUC": [],
    "Eval F1": [],
    "Eval Precision": [],
    "Eval Recall": [],
    "Eval MACRO AUC": [],
    "Eval MACRO ACC": [],
    "Eval F2": [],
    "Class Accuracies": [[] for _ in range(7)]  # ✅ 存每一类
}


def manual_isin(tensor, values):
    values = torch.tensor(values, dtype=tensor.dtype, device=tensor.device)
    return tensor.view(-1, 1).eq(values.view(1, -1)).any(dim=1)


def prepare_trte_data(data_folder, num_class, num_fold, num_view=12):
    if 'PTBXL' in data_folder:
        data_tr_list, labels_tr, data_te_list, labels_te = read_ptbxl(num_class, num_fold)
    elif 'PTB' in data_folder:
        data_tr_list, labels_tr, data_te_list, labels_te = read_ptb(num_class, num_fold)
    else:
        raise ValueError("Unsupported dataset")

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]

    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))

    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))

    idx_dict = {"tr": list(range(num_tr)), "te": list(range(num_tr, (num_tr + num_te)))}

    data_train_list = []
    data_test_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]])
        data_test_list.append(data_tensor_list[i][idx_dict["te"]])

    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_test_list, idx_dict, labels


def run(model, testonly, num_class=7, fold_num=10, dataset='PTBXL'):
    num_epoch = 200
    lr = 1e-3

    # === 数据读取 ===
    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(dataset, 2, fold_num, num_view=12)
    data_tr_list5, data_test_list5, trte_idx5, labels_trte5 = prepare_trte_data(dataset, 5, fold_num, num_view=12)
    data_tr_list7, data_test_list7, trte_idx7, labels_trte7 = prepare_trte_data(dataset, 7, fold_num, num_view=12)

    labels_tr_tensor5 = torch.LongTensor(labels_trte5[trte_idx["tr"]])
    labels_te_tensor5 = torch.LongTensor(labels_trte5[trte_idx["te"]])
    labels_tr_tensor7 = torch.LongTensor(labels_trte7[trte_idx["tr"]])
    labels_te_tensor7 = torch.LongTensor(labels_trte7[trte_idx["te"]])

    rare_labels = torch.tensor([0, 2, 4, 5])
    labels_tr_tensor_rare = labels_tr_tensor7[:, rare_labels]
    labels_te_tensor_rare = labels_te_tensor7[:, rare_labels]

    dataset_tr = matDataset(data_tr_list, labels_tr_tensor_rare, labels_tr_tensor5, labels_tr_tensor7)
    dataset_te = matDataset(data_test_list, labels_te_tensor_rare, labels_te_tensor5, labels_te_tensor7)

    train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=64)
    test_loader = torch.utils.data.DataLoader(dataset_te, batch_size=64)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    # === 测试模式 ===
    if testonly:
        model = torch.load('./{}/Best_Model_KFold{}.pt'.format(dataset, fold_num))
        te_prob = []
        for batch_idx, (batch, y, y5, y7) in enumerate(test_loader):
            with torch.no_grad():
                elogit = model(batch, y, y5, y7)[1]
                prob = torch.sigmoid(elogit).data.cpu().numpy()
                te_prob = np.concatenate((te_prob, prob), axis=0) if len(te_prob) else prob

        if 'PTBXL' in dataset:
            eval_acc, eval_auroc = ptbxl_Result(num_class, te_prob, labels_trte7[trte_idx["te"]])
            print('Eval Acc: {:.6f}, Eval Auroc: {:.6f}'.format(eval_acc, eval_auroc))
        elif 'PTB' in dataset:
            eval_sen, eval_spe, eval_acc = ptb_Result(num_class, te_prob, labels_trte7[trte_idx["te"]])
            print('Eval Sen: {:.6f}, Eval Spe: {:.6f}, Eval Acc: {:.6f}'.format(eval_sen, eval_spe, eval_acc))
        return

    # === 训练模式 ===
    print("\nTraining...")
    best_epoch = -1
    best_loss = float("inf")
    best_score = 0
    best_state = None

    early_stopping = EarlyStopping(patience=15, verbose=True, path='model.pth')

    for epoch in range(num_epoch + 1):
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
            tr_prob = np.concatenate((tr_prob, prob), axis=0) if len(tr_prob) else prob

        train_acc, train_auroc, _, _, _, _, _, _, _ = ptbxl_Result(7, tr_prob, labels_trte7[trte_idx["tr"]])

        # === eval ===
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
                te_prob = np.concatenate((te_prob, prob), axis=0) if len(te_prob) else prob

        eval_acc, eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2, class_accuracies = ptbxl_Result(
            7, te_prob, labels_trte7[trte_idx["te"]])

        print(
            'epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Auroc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}, Eval PR AUC: {:.6f}, Eval F1: {:.6f}, Eval Precision: {:.6f}, Eval Recall: {:.6f}, Eval MACRO AUC: {:.6f}, Eval MACRO ACC: {:.6f},Eval F2: {:.6f}'
            .format(epoch, train_loss / cnt, train_acc, train_auroc, eval_loss / ecnt, eval_acc,
                    eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2))

        # === 更新 best_epoch ===
        if (eval_loss / ecnt) < best_loss or best_score < eval_pr_auc + eval_f1:
            best_loss = eval_loss / ecnt
            best_epoch = epoch
            best_state = model.state_dict()
            best_score = eval_pr_auc + eval_f1

        auc_max = early_stopping(eval_loss / ecnt, model, eval_pr_auc, eval_f1)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # === 回到 best_epoch - 15 重新评估 ===
    target_epoch = max(0, best_epoch)
    print(f"\n[Fold {fold_num}] Best epoch: {best_epoch}, Evaluating epoch: {target_epoch}")

    model.load_state_dict(best_state)
    te_prob = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch, y4, y5, y7) in enumerate(test_loader):
            eloss, elogit = model(batch, y4, y5, y7, dataset)
            prob = torch.sigmoid(elogit).data.cpu().numpy()
            te_prob = np.concatenate((te_prob, prob), axis=0) if len(te_prob) else prob

        eval_acc, eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc, eval_f2, class_accuracies = ptbxl_Result(
            7, te_prob, labels_trte7[trte_idx["te"]])

        fold_metrics["Eval Acc"].append(eval_acc)
        fold_metrics["Eval PR AUC"].append(eval_pr_auc)
        fold_metrics["Eval F1"].append(eval_f1)
        fold_metrics["Eval Precision"].append(eval_precision)
        fold_metrics["Eval Recall"].append(eval_recall)
        fold_metrics["Eval MACRO AUC"].append(macro_auc)
        fold_metrics["Eval MACRO ACC"].append(avg_acc)
        fold_metrics["Eval F2"].append(eval_f2)

        # ✅ 保存每个类别的准确率
        for i, acc in enumerate(class_accuracies):
            fold_metrics["Class Accuracies"][i].append(acc)

        # ✅ 每个 fold 结果写到文件
        os.makedirs("TResult", exist_ok=True)
        with open("TResult/fold_summary.txt", "a") as f:
            f.write(f"\n--- Fold {fold_num} Results (best_epoch-15) ---\n")
            f.write(f"Eval Acc: {eval_acc * 100:.2f}\n")
            f.write(f"Eval PR AUC: {eval_pr_auc * 100:.2f}\n")
            f.write(f"Eval F1: {eval_f1 * 100:.2f}\n")
            f.write(f"Eval Precision: {eval_precision * 100:.2f}\n")
            f.write(f"Eval Recall: {eval_recall * 100:.2f}\n")
            f.write(f"Eval MACRO AUC: {macro_auc * 100:.2f}\n")
            f.write(f"Eval MACRO ACC: {avg_acc * 100:.2f}\n")
            f.write(f"Eval F2: {eval_f2 * 100:.2f}\n")
            for j, acc in enumerate(class_accuracies):
                f.write(f"Class {j} Acc: {acc * 100:.2f}\n")
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
        dataset_list = [5000] * 12
    elif 'PTB' in dataset:
        folds = 5
        dataset_list = [600] * 12
    else:
        print("dataset error!")
        sys.exit()

    for i in range(folds):
        print('KFold: {}'.format(i + 1))
        dim_list = dataset_list
        hidden_dim = [32]
        model = PDF3(dim_list, hidden_dim, num_class, dropout=0.5)
        model.cuda()
        run(model, testonly, num_class, int(i + 1), dataset)

    # === 所有 fold 结束后计算均值±标准差 ===
    results_file = "TResult/fold_summary.txt"
    with open(results_file, "a") as f:
        f.write("\n=== KFold Summary (best_epoch) ===\n")
        for key, values in fold_metrics.items():
            if key == "Class Accuracies":
                for i, class_vals in enumerate(values):
                    class_vals = np.array(class_vals) * 100
                    mean = class_vals.mean()
                    std = class_vals.std()
                    f.write(f"Class {i} Acc: {mean:.2f} ± {std:.2f}\n")
            else:
                values = np.array(values) * 100
                mean = values.mean()
                std = values.std()
                f.write(f"{key}: {mean:.2f} ± {std:.2f}\n")
    print(f"\n>>> Summary saved to {results_file}")

