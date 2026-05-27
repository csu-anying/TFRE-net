""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from SRT_model import *
# from SEnet import *
# from CA import *
# from DKR import *
import sys


# from unireplknet import UniRepLKNetBlock

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCE With Logits Loss + Sigmoid Activation
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)  # Convert logits to probabilities

        # Focal Loss Modulation
        focal_weight = self.alpha * (1 - probs) ** self.gamma * targets + \
                       (1 - self.alpha) * probs ** self.gamma * (1 - targets)
        focal_loss = focal_weight * BCE_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # No reduction

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

        self.FeatureEncoder = SRTNet()

        self.MMClasifier = []
        self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], 2))  # ��������1
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

        self.MMClasifier5 = []
        self.MMClasifier5.append(LinearLayer(self.views * hidden_dim[-1], 5))  # ��������2
        self.MMClasifier5 = nn.Sequential(*self.MMClasifier5)

        self.MMClasifier7 = []
        self.MMClasifier7.append(LinearLayer(84, 7))  # ��������3
        self.MMClasifier7 = nn.Sequential(*self.MMClasifier7)

    # def forward(self, data_list, label2=None, label5=None, label7=None, dataset='PTB'):
    def forward(self, data_list, label2=None, label5=None, label7=None, dataset='PTBXL'):
        # �����
        FeatureInfo, feature, feature_lead, TCPLogit, TCPConfidence, lead_weight, TCPpreLogit = dict(), dict(), dict(), dict(), dict(), dict(), dict()
        view_weight_sigmoid = torch.empty(3, 3)

        feature = torch.stack(data_list, dim=-1)  # [64,5000,12]
        feature = torch.transpose(feature, 1, 2)  # [64,12,5000]

        MMfeature = self.FeatureEncoder(feature)

        # ���ڿ�����ط�=
        MIfeature, MIfeature_1 = dict(), dict()

        # MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMfeature = F.dropout(MMfeature, self.dropout, training=self.training)


        # MMlogit = self.MMClasifier(MMfeature)
        # MMlogit5 = self.MMClasifier5(MMfeature)
        MMlogit7 = self.MMClasifier7(MMfeature)

        criterion0 = torch.nn.CrossEntropyLoss(reduction='none')
        criterion = torch.nn.BCEWithLogitsLoss()  # BCEloss���ڶ����ཻ���أ��ú���������BCE Loss�Լ�Sigmoid�����㣬
        criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)
        if 'PTBXL' in dataset:
            # Loss2 = torch.mean(criterion0(MMlogit, label2))
            # Loss5 = torch.mean(criterion(MMlogit5, label5.to(torch.float)))
            # Loss7 = torch.mean(criterion(MMlogit7, label7.to(torch.float)))
            Loss7 = criterion_focal(MMlogit7, label7.to(torch.float))
        else:
            # Loss2 = torch.mean(criterion0(MMlogit, label2))
            # Loss5 = torch.mean(criterion0(MMlogit5, label5))
            Loss7 = torch.mean(criterion0(MMlogit7, label7))

        # MMLoss = 0.013 * Loss2 + 0.19 * Loss5 + 0.79 * Loss7
        MMLoss = Loss7
        return MMLoss, MMlogit7

