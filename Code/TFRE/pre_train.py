import os, sys
import numpy as np
from pytorchtools import EarlyStopping
import torch
import torch.nn.functional as F

# from basemodel_im import PDF3 # SRT_MODEL
from basemodel_cab import PDF3  # double_SE # MCA MLRESNET
# from basemodel_im_second import PDF3 # TB-net
# from basemodel_transformer import PDF3  # se_trans
# from iTransformer import *# 马上跑的
import random

# from basemodel_uni import UNI # 多头双
# from basemodel_solomutle import PDF3
from read_ptb import *
from read_ptbxl import *

from datasets import *

# from Result_new import *
from Result_newnewnew import *
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cuda = True if torch.cuda.is_available() else False  # 使用GPU


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
    num_epoch = 100
    lr = 1e-4 # 1e-4
    # 数据读取
    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(dataset, 2, fold_num, num_view=12)
    data_tr_list5, data_test_list5, trte_idx5, labels_trte5 = prepare_trte_data(dataset, 5, fold_num, num_view=12)
    data_tr_list7, data_test_list7, trte_idx7, labels_trte7 = prepare_trte_data(dataset, 7, fold_num, num_view=12)
    # 标签转tensor
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    labels_te_tensor = torch.LongTensor(labels_trte[trte_idx["te"]])
    labels_tr_tensor5 = torch.LongTensor(labels_trte5[trte_idx["tr"]])
    labels_te_tensor5 = torch.LongTensor(labels_trte5[trte_idx["te"]])
    labels_tr_tensor7 = torch.LongTensor(labels_trte7[trte_idx["tr"]])
    labels_te_tensor7 = torch.LongTensor(labels_trte7[trte_idx["te"]])
    # 转化为dataloader能识别的dataset
    dataset_tr = matDataset(data_tr_list, labels_tr_tensor, labels_tr_tensor5, labels_tr_tensor7)
    dataset_te = matDataset(data_test_list, labels_te_tensor, labels_te_tensor5, labels_te_tensor7)

    # train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=128)
    # test_loader = torch.utils.data.DataLoader(dataset_te, batch_size=512)

    train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=64, shuffle=True)  # 在这里添加shuffle 64
    test_loader = torch.utils.data.DataLoader(dataset_te, batch_size=64) # 128

    # train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=128,shuffle = True)#在这里添加shuffle
    # test_loader = torch.utils.data.DataLoader(dataset_te, batch_size=512)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

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
            for batch_idx, (batch, y, y5, y7) in enumerate(train_loader):
                cnt += 1
                optimizer.zero_grad()
                loss, logit = model(batch, y, y5, y7, dataset)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                prob = torch.sigmoid(logit).data.cpu().numpy()

                if len(tr_prob) == 0:
                    tr_prob = prob
                else:
                    tr_prob = np.concatenate((tr_prob, prob), axis=0)

            if 'PTBXL' in dataset:
                train_acc, train_auroc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc,eval_f2, class_accuracies = ptbxl_Result(
                    7,
                    tr_prob,
                    labels_trte7[
                        trte_idx[
                            "tr"]])
            elif 'PTB' in dataset:
                # train_sen, train_spe, train_acc = ptb_Result(7, tr_prob, labels_trte7[trte_idx7["tr"]])
                train_acc, train_auroc, eval_f1, eval_precision, eval_recall, class_accuracies = ptb_Result(7,
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
            for batch_idx, (batch, y, y5, y7) in enumerate(test_loader):
                ecnt += 1
                with torch.no_grad():
                    eloss, elogit = model(batch, y, y5, y7, dataset)

                    eval_loss += eloss.item()
                    prob = torch.sigmoid(elogit).data.cpu().numpy()

                    if len(te_prob) == 0:
                        te_prob = prob
                    else:
                        te_prob = np.concatenate((te_prob, prob), axis=0)

            if 'PTBXL' in dataset:
                eval_acc, eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc,eval_f2, class_accuracies = ptbxl_Result(
                    7, te_prob, labels_trte7[trte_idx["te"]])
                print(
                    'epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Auroc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}, Eval PR AUC: {:.6f}, Eval F1: {:.6f}, Eval Precision: {:.6f}, Eval Recall: {:.6f}, Eval MACRO AUC: {:.6f}, Eval MACRO ACC: {:.6f},Eval F2: {:.6f}'
                    .format(epoch, train_loss / cnt, train_acc, train_auroc, eval_loss / ecnt, eval_acc,
                            eval_pr_auc, eval_f1, eval_precision, eval_recall, macro_auc, avg_acc,eval_f2))
                # 打印每个类别的准确率
                for i, class_acc in enumerate(class_accuracies):
                    print('Class {} Acc: {:.6f}'.format(i, class_acc))
                # with open('output_trans.txt', 'a') as file:
                #     # 写入训练和评估的指标
                #     file.write(
                #         'epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Auroc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}, Eval PR AUC: {:.6f}, Eval F1: {:.6f}, Eval Precision: {:.6f}, Eval Recall: {:.6f}\n'
                #         .format(epoch, train_loss / cnt, train_acc, train_auroc, eval_loss / ecnt, eval_acc,
                #                 eval_pr_auc, eval_f1, eval_precision, eval_recall))
                #     # 写入每个类别的准确率
                #     for i, class_acc in enumerate(class_accuracies):
                #         file.write('Class {} Acc: {:.6f}\n'.format(i, class_acc))
                auc_max = early_stopping(eval_loss / ecnt, model, eval_pr_auc, eval_f1)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

                # if 'PTBXL' in dataset:
                #     eval_acc, eval_auroc = ptbxl_Result(num_classes, te_prob, labels_trte7[trte_idx["te"]])
                #     print(
                #         'epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Auroc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}, Eval Auroc: {:.6f}'
                #         .format(epoch, train_loss / cnt, train_acc, train_auroc, eval_loss / ecnt, eval_acc, eval_auroc))
                # auc_max = early_stopping(eval_loss / ecnt, model, eval_auroc)
                # auc_max = early_stopping(eval_loss / ecnt, model, eval_pr_auc, eval_f1)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

            # elif 'PTB' in dataset:
            #     eval_sen, eval_spe, eval_acc = ptb_Result(7, te_prob, labels_trte7[trte_idx["te"]])
            #     print(
            #         'epoch: {}, Train Loss: {:.6f}, Train Sen : {:.6f}, Train Spe: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Sen: {:.6f}, Eval Spe: {:.6f}, Eval Acc: {:.6f}'
            #         .format(epoch, train_loss / cnt, train_sen, train_spe, train_acc, eval_loss / ecnt, eval_sen,
            #                 eval_spe, eval_acc))
            #     evalloss_min = early_stopping(eval_loss / ecnt, model, eval_acc)
            #     if early_stopping.early_stop:
            #         print("Early stopping triggered.")
            #         break
            elif 'PTB' in dataset:
                eval_acc, eval_pr_auc, eval_f1, eval_precision, eval_recall, class_accuracies = ptb_Result(
                    7, te_prob,
                    labels_trte7[
                        trte_idx["te"]], prin=True)
                print(
                    'epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Train Auroc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}, Eval PR AUC: {:.6f}, Eval F1: {:.6f}, Eval Precision: {:.6f}, Eval Recall: {:.6f}'
                    .format(epoch, train_loss / cnt, train_acc, train_auroc, eval_loss / ecnt, eval_acc,
                            eval_pr_auc, eval_f1, eval_precision, eval_recall))
                auc_max = early_stopping(eval_loss / ecnt, model, eval_pr_auc, eval_f1)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

        # early_stopping.load_best_model(model)

        # def print_model_parameters(model):
        #     for name, param in model.named_parameters():
        #         print(f"Layer: {name}, Size: {param.size()}")
        #         print(
        #             f"Mean: {param.mean().item():.4f}, Std: {param.std().item():.4f}, Min: {param.min().item():.4f}, Max: {param.max().item():.4f}")
        #         print("----------")


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
    output_folder = 'output_folder_contrast'
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'mca_output_2.txt')
    with open(output_file, 'a') as f:
        sys.stdout = f  # 重定向stdout到文件

        for i in range(folds):
            # if i in [5,6,7,8,9]:
            #     continue  # 跳过 i 为 1, 2, 3, 4, 5 的情况
            print('KFold: {}'.format(i + 1))

            # # 因为MCA-net的第四折保存的模型损坏了所以没办法读取，这里直接输出当时的结果。
            # if 'PTBXL' not in dataset and int(i + 1) == 4 and testonly == True:
            #     print('Eval Sen: 0.293538, Eval Spe: 0.347262, Eval Acc: 0.348828')
            #     continue
            dim_list = dataset_list
            hidden_dim = [16] # 8 16MCA 48transformer 8MLRESNET
            model = PDF3(dim_list, hidden_dim, num_class, dropout=0.5)  # 建模型
            # model = UNI(dim_list, hidden_dim, num_class, dropout=0.5)
            model.cuda()  # 放入模型到GPU
            run(model, testonly, num_class, int(i + 1), dataset)  # 运行模型

        sys.stdout = sys.__stdout__
