""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from models import *
from CBAM.CBAM import *
from MRM.duibi import *
from TGLLnet import Mymodel
import sys



import torch.nn.functional as F

# 正态分布初始化函数
def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class PDF3(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):  # in_dim是有12个5000的列表，hidden_dim是[16],num_class是7
        super().__init__()
        self.views = len(in_dim)  # 12��12������
        self.classes = num_class  # ��������Ĭ��Ϊ7
        # self.dropout = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.embding = hidden_dim[-1]

        # 创建配置对象和超参数字典
        configs = Configs()
        hparams = {"feature_dim": 8}  # 根据实际情况调整 feature_dim
        # self.model = ecgTransForm(configs, hparams)
        # self.model = Dui1(num_classes=7, input_channels=12)#MRM
        self.model = ECGNet()#CBAM
        # self.model = Mymodel(num_classes=num_class)#TGLLnet

        self.MMClasifier7 = []
        self.MMClasifier7.append(LinearLayer(self.views * hidden_dim[-1], 7))  # ��������3
        self.MMClasifier7 = nn.Sequential(*self.MMClasifier7)

    def forward(self, data_list, label4=None, label5=None, label7=None, dataset='PTBXL',inference_only=False):
        # �����
        FeatureInfo, feature, feature_lead, TCPLogit, TCPConfidence, lead_weight, TCPpreLogit = dict(), dict(), dict(), dict(), dict(), dict(), dict()


        stacked_data = torch.stack(data_list)  # [12, 64, 5000]
        stacked_data = torch.transpose(stacked_data, 1, 0)  # [64, 12, 5000]
        # MMlogit7 = self.model(stacked_data)  # [64, 7]

        MMlogit7 = self.model(stacked_data) # other
        # for view in range(self.views):
        #     feature[view] = self.FeatureEncoder[view](data_list[view]) #  64,5000->64,32,1
        #     feature[view] = feature[view].flatten(start_dim=1, end_dim=2) #  64,32,1->64,32

        # MIfeature, MIfeature_1 = dict(), dict()
        # MMfeature = torch.stack([i for i in feature.values()], dim=1) #  ->64,12,32
        # # Now concatenate the cross-encoded features with the original features
        # MMfeature = MMfeature.view(-1,12*self.embding).clone()
        # if self.training:  # 只在训练阶段启用 R-drop
        #     MMfeature = self.dropout(MMfeature)  # 添加 dropout
        #
        # MMlogit7 = self.MMClasifier7(MMfeature)

        # normal loss
        # criterion0 = torch.nn.CrossEntropyLoss(reduction='none')
        # criterion = torch.nn.BCEWithLogitsLoss()  # BCEloss���ڶ����ཻ���أ��ú���������BCE Loss�Լ�Sigmoid�����㣬
        # # print(type(MMlogit7))  # 应该输出 <class 'torch.Tensor'>
        # # print(MMlogit7.shape)  # 输出张量的形状
        # if 'PTBXL' in dataset:
        #     weights = torch.tensor(
        #     [1, 1, 1, 1, 1, 1, 1]).cuda()
        #     Loss7 = torch.mean(weights * criterion(MMlogit7, label7.to(torch.float)))
        #
        # else:
        #     Loss7 = torch.mean(criterion0(MMlogit7, label7))
        # MMLoss = Loss7
        # MMLoss = Loss7
        if inference_only:
            # 只走 forward，不算 loss
            return MMlogit7
        #reconstruct loss
        criterion0 = nn.CrossEntropyLoss(reduction='none')
        criterion = nn.BCEWithLogitsLoss()

        if 'PTBXL' in dataset:
            weights = torch.tensor([1, 1, 1, 1, 1, 1, 1]).cuda()
            Loss7 = torch.mean(weights * criterion(MMlogit7, label7.to(torch.float)))
        else:
            Loss7 = torch.mean(criterion0(MMlogit7, label7))


        MMLoss = Loss7

        return MMLoss, MMlogit7
