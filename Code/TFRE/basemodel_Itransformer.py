""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from iTransformer import *
import sys


# from unireplknet import UniRepLKNetBlock

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
        self.dropout = dropout
        self.configs = Configs()
        self.itrans = Model(self.configs)

        self.MMClasifier = []
        self.MMClasifier.append(LinearLayer(100, 2))  # ��������1
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

        self.MMClasifier5 = []
        self.MMClasifier5.append(LinearLayer(100, 5))  # ��������2
        self.MMClasifier5 = nn.Sequential(*self.MMClasifier5)

        self.MMClasifier7 = []
        self.MMClasifier7.append(LinearLayer(100, 7))  # ��������3
        self.MMClasifier7 = nn.Sequential(*self.MMClasifier7)

    # def forward(self, data_list, label2=None, label5=None, label7=None, dataset='PTB'):
    def forward(self, data_list, label2=None, label5=None, label7=None, dataset='PTBXL'):
        # �����
        FeatureInfo, feature, feature_lead, TCPLogit, TCPConfidence, lead_weight, TCPpreLogit = dict(), dict(), dict(), dict(), dict(), dict(), dict()
        # 导联间特征
        bothfeature = torch.stack(data_list, dim=-1)  # [64,5000,12]
        # SOLOfeature = torch.transpose(SOLOfeature, 1, 2)  # [64,12,5000]
        # downfeature = torch.squeeze(downfeature, dim=1)  # [64,5000]

        # for view in range(self.views):
        #     feature[view] = self.FeatureEncoder[view](data_list[view])
        #     #将data_list[view]融合为单导联
        #     feature[view] = feature[view].flatten(start_dim=1, end_dim=2)
        # ���ڿ�����ط�=
        MIfeature, MIfeature_1 = dict(), dict()

        # MMfeature = torch.cat([i for i in feature.values()], dim=1)
        # MMfeature = F.dropout(MMfeature, self.dropout, training=self.training)


        MMlogit = self.MMClasifier(MMfeature)
        MMlogit5 = self.MMClasifier5(MMfeature)
        MMlogit7 = self.MMClasifier7(MMfeature)

        criterion0 = torch.nn.CrossEntropyLoss(reduction='none')
        criterion = torch.nn.BCEWithLogitsLoss()  # BCEloss���ڶ����ཻ���أ��ú���������BCE Loss�Լ�Sigmoid�����㣬
        if 'PTBXL' in dataset:
            Loss2 = torch.mean(criterion0(MMlogit, label2))
            Loss5 = torch.mean(criterion(MMlogit5, label5.to(torch.float)))
            Loss7 = torch.mean(criterion(MMlogit7, label7.to(torch.float)))
        else:
            Loss2 = torch.mean(criterion0(MMlogit, label2))
            Loss5 = torch.mean(criterion0(MMlogit5, label5))
            Loss7 = torch.mean(criterion0(MMlogit7, label7))

        MMLoss = 0.013 * Loss2 + 0.19 * Loss5 + 0.79 * Loss7

        return MMLoss, MMlogit7


