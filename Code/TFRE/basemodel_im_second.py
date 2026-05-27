""" Componets of the model
"""
import math

import torch.nn as nn
import torch
import torch.nn.functional as F
# from SEnet import *
from SEresnet import *
# from iTransformer import *
# from SEinception import *

# from CA import *
# from DKR_second import *
# from SE_tran import *
# from resnet_500 import *
import sys
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch

# from unireplknet import UniRepLKNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)  # 启用异常检测

# Transformer特征提取器
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Initialize relative positional encodings
        self.relative_positions = nn.Parameter(torch.randn(2 * max_len - 1, d_model))

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.d_model, "Input embedding dimension must match the model dimension"

        pos_encoding = torch.zeros((batch_size, seq_len, d_model), device=x.device)

        for i in range(seq_len):
            for j in range(seq_len):
                pos_index = i - j + self.max_len - 1
                pos_encoding[:, i, :] += self.relative_positions[pos_index, :]

        x = x + pos_encoding
        return x


class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim, max_len=600, dropout=0.5):
        super(TransformerFeatureExtractor, self).__init__()

        # Input and output projections
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)

        # Positional Encoding
        self.positional_encoding = RelativePositionalEncoding(hidden_dim, max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=hidden_dim * 4  # Increase FFN layer size (optional)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Input projection and positional encoding
        x = self.linear_in(x)
        x = self.positional_encoding(x)

        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Transformer encoder with optional mask for variable-length inputs
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Output projection
        x = self.linear_out(x)
        return x

class SimpleBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleBiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)  # *2 because of bidirectional

    def forward(self, x):
        batch_size, seq_types, seq_length = x.size()
        x = x.view(batch_size * seq_types, seq_length, -1)  # Reshape to [batch_size * seq_types, seq_length, input_dim]
        lstm_out, _ = self.bilstm(x)  # LSTM forward pass
        lstm_out_last = lstm_out[:, -1, :]  # Get last time step
        linear_out = self.linear(lstm_out_last)  # Linear layer
        output = linear_out.view(batch_size, seq_types, -1)  # Reshape back to [batch_size, seq_types, output_dim]
        return output

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


def create_classifiers(input_dim):
    classifiers = {
        '2': nn.ModuleList([nn.Sequential(LinearLayer(input_dim, 1)) for _ in range(2)]),
        '5': nn.ModuleList([nn.Sequential(LinearLayer(input_dim, 1)) for _ in range(5)]),
        '7': nn.ModuleList([nn.Sequential(LinearLayer(input_dim, 1)) for _ in range(7)])
    }
    return nn.ModuleDict(classifiers)


# 全连接图生成
def generate_complete_graph_edges(num_nodes):
    # 生成全连接图的边索引
    edge_index = torch.combinations(torch.arange(num_nodes), 2)
    return edge_index.t().contiguous()
# 焦点损失
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class PDF3(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):  # in_dim是有12个5000的列表，hidden_dim是[16],num_class是7
        super().__init__()
        self.views = len(in_dim)  # 12��12������
        self.classes = num_class  # ��������Ĭ��Ϊ7
        self.dropout = dropout
        # self.FeatureEncoder_DKR = mSEnet_DKR()

        # Transformer 特征提取器
        self.transformer_extractor_2 = TransformerFeatureExtractor(input_dim=32, num_heads=4, num_layers=4,
                                                                   hidden_dim=128, output_dim=32)
        self.transformer_extractor_3 = TransformerFeatureExtractor(input_dim=32, num_heads=32, num_layers=16,
                                                                   hidden_dim=128, output_dim=32)
        self.tr_2 = nn.Sequential(
            nn.BatchNorm1d(12),
            nn.LeakyReLU(0.3)
        )
        self.tr_3 = nn.Sequential(
            nn.BatchNorm1d(12),
            nn.LeakyReLU(0.3)
        )
        self.FeatureEncoder_trans = nn.ModuleList(
            [TransformerFeatureExtractor(input_dim=32, num_heads=4, num_layers=4,
                                                                   hidden_dim=128, output_dim=32) for view in range(self.views)])
        # BiLSTM 特征提取器
        self.BLSTM_extractor_2 = SimpleBiLSTM(input_dim=1, hidden_dim=64, output_dim=32)
        self.BL_2 = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.3)
        )
        self.BLSTM_extractor_3 = SimpleBiLSTM(input_dim=1, hidden_dim=64, output_dim=32)
        self.BL_3 = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.3)
        )

        # # CNN 特征提取器
        self.cn_1 = mSEnet()
        self.db_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.3)
        )
        self.cn_2 = mSEnet()
        self.db_2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.3)
        )

        # self.configs = Configs()
        # self.itransformer = Model(self.configs)
        self.FeatureEncoder1 = nn.ModuleList(
            [mSEnet() for view in range(self.views)])
        self.FeatureEncoder2 = nn.ModuleList(
            [mSEnet() for view in range(self.views)])

        self.SE_gcn = SE_Module(12, dim=1)
        self.SE_gat = SE_Module(12, dim=1)
        self.BN_gcn = nn.BatchNorm1d(12)
        self.BN_gat = nn.BatchNorm1d(12)

        self.gcn1 = pyg_nn.GraphConv(32, 128)

        self.gcn1_0 = nn.Sequential(
            pyg_nn.BatchNorm(12),
            nn.LeakyReLU(0.3)
        )

        self.gcn2 = pyg_nn.GraphConv(128, 512)

        self.gcn2_0 = nn.Sequential(
            pyg_nn.BatchNorm(12),
            nn.LeakyReLU(0.3)
        )

        self.gcn3 = pyg_nn.GraphConv(512, 32)

        self.gcn3_0 = nn.Sequential(
            pyg_nn.BatchNorm(12),
            nn.LeakyReLU(0.3)
        )

        self.gat1 = pyg_nn.GATConv(32, 32, heads=12, concat=True)

        self.gat1_0 = nn.Sequential(
            pyg_nn.BatchNorm(32 * 12),
            nn.LeakyReLU(0.5)
        )

        self.gat2 = pyg_nn.GATConv(32 * 12, 32, heads=1, concat=True)

        self.gat2_0 = nn.Sequential(
            pyg_nn.BatchNorm(32),
            nn.LeakyReLU(0.5)
        )

        self.classifiers = create_classifiers(12 * hidden_dim[-1])

        self.Part_Clasifier = nn.ModuleDict({
            'classifier_1': nn.Linear(12 * 32, 1),  # 对应第一个分类器
            'classifier_2': nn.Linear(12 * 32, 1),  # 对应第二个分类器
            'classifier_3': nn.Linear(12 * 32, 1),  # 对应第三个分类器
            'classifier_4': nn.Linear(12 * 32, 1),  # 对应第四个分类器
            'classifier_5': nn.Linear(12 * 32, 1),  # 对应第五个分类器
            'classifier_6': nn.Linear(12 * 32, 1),  # 对应第六个分类器
            'classifier_7': nn.Linear(12 * 32, 1)   # 对应第七个分类器
        })

        self.MMClasifier = []
        self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], 2))  # ��������1
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

        self.MMClasifier5 = []
        self.MMClasifier5.append(LinearLayer(self.views * hidden_dim[-1], 5))  # ��������2
        self.MMClasifier5 = nn.Sequential(*self.MMClasifier5)

        self.MMClasifier7 = []
        self.MMClasifier7.append(LinearLayer(self.views * hidden_dim[-1], 7))  # ��������3
        self.MMClasifier7 = nn.Sequential(*self.MMClasifier7)

        self.MMClasifier_Dout = []
        self.MMClasifier_Dout.append(LinearLayer(self.views * hidden_dim[-1], 1))  # ��������1
        self.MMClasifier_Dout = nn.Sequential(*self.MMClasifier_Dout)
        self.leakyrule = nn.LeakyReLU(0.3)
    # def forward(self, data_list, label2=None, label5=None, label7=None, dataset='PTB'):
    def forward(self, batch, num_classes=7, label7=None, dataset='PTBXL', criter='CE', return_only_output=False):
        # def forward(self, data_list, num_classes, labels=None, dataset='PTBXL'):
        # �����
        FeatureInfo, feature, feature_lead, TCPLogit, TCPConfidence, lead_weight, TCPpreLogit = dict(), dict(), dict(), dict(), dict(), dict(), dict()
        feature_less = dict()
        if isinstance(batch, list):
            data_list = batch
        elif isinstance(batch, torch.Tensor):
            dim = batch.dim()
            if dim == 3:
                data_list = torch.unbind(batch, dim=1)
            elif dim == 4:
                data_list = batch.squeeze(0)
                data_list = torch.unbind(data_list, dim=0)
        # PTBXL shap图构图最好的构图
        Anterior_septal_edges = [(6, 7)]
        Anterolateral_edges = [(10, 11)]
        Inferior_edges = [(3,0)]
        # 合并所有边
        all_edges = Anterior_septal_edges + Anterolateral_edges + Inferior_edges

        # 去重（因为某些边可能在不同的类型中被重复定义）
        all_edges = list(set(all_edges))

        # 转换为PyTorch tensor格式
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_index = edge_index.to(device)
        for view in range(self.views):
            feature_less[view] = self.FeatureEncoder2[view](data_list[view]) # origin now
            feature_less[view] = feature_less[view].flatten(start_dim=1, end_dim=2)
        MIfeature, MIfeature_1 = dict(), dict()
        # MINOR OR MORE
        MMfeature = torch.cat([i for i in feature_less.values()], dim=1)
        # #9 10
        # MMfeature = self.MMClasifier_Dout(MMfeature)
        stacked_data = torch.stack([i for i in feature_less.values()])  # [12, 64, 32]
        stacked_data = stacked_data.permute(1, 0, 2)  # [64, 12, 32]
        stacked_data = stacked_data.to(device)
        # 使用 Transformer 特征提取器进行编码
        tr_tensor_2 = self.transformer_extractor_2(stacked_data)
        tr_tensor_2 = self.tr_2(tr_tensor_2)

        # GCN
        x = self.gcn1(stacked_data, edge_index)
        x = self.gcn1_0(x)
        x = self.gcn2(x, edge_index)
        x = self.gcn2_0(x)
        x = self.gcn3(x, edge_index)
        x = self.gcn3_0(x)
        x = self.SE_gcn(x)
        # # 应用全连接层降维
        # # 重塑输出以匹配所需的形状[批次大小, 节点数, 特征数]
        x_1 = x.view(-1, 12, 32)
        # stacked_data += x_1
        #GAT
        gat_edge = generate_complete_graph_edges(12)
        gat_edge = gat_edge.to(device)

        x = stacked_data.reshape(-1, 32)
        x = self.gat1(x, gat_edge)
        x = self.gat1_0(x)
        x = self.gat2(x, gat_edge)
        x = self.gat2_0(x)
        x = x.view(-1, 12, 32)
        x = self.BN_gat(x)
        # x = self.SE_gat(x)
        # # 10
        # MMfeature = self.leakyrule(MMfeature)
        # # 9
        # stacked_data += (x+x_1+ tr_tensor_2)*(MMfeature.unsqueeze(-1))
        # 8 11
        x += x_1+tr_tensor_2+stacked_data
        # 12
        # stacked_data += x_1
        # MMfeature = stacked_data.flatten(start_dim=1, end_dim=2)
        # MMfeature = self.MMClasifier_Dout(MMfeature)
        # MMfeature = self.leakyrule(MMfeature)
        # stacked_data += (x + tr_tensor_2) * (MMfeature.unsqueeze(-1))

        selected_lead = []
        # origin 8 11
        selected_lead.append(x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        selected_lead.append(x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        selected_lead.append(x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        selected_lead.append(x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        # 9 10 12
        # selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        # selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        # selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        # selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        # selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        # selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        # selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))


        for i in range(7):
            selected_lead[i] = F.dropout(selected_lead[i], self.dropout, training=self.training).to(device)

        selected_class = []
        # 两个等长的列表
        indices1 = [0, 1, 2, 3, 4, 5, 6]
        indices2 = [0, 1, 2, 3, 4, 5, 6]
        classifier_names = list(self.Part_Clasifier.keys())
        # 使用 zip 函数并行迭代
        for i, j in zip(indices1, indices2):
            classifier_name = classifier_names[j]  # 使用索引 j 从名称列表中获取对应的键名
            pred_class = self.Part_Clasifier[classifier_name](selected_lead[i]).to(device)
            pred_class = pred_class.to(device)
            selected_class.append(pred_class)
        selected_class = torch.cat(selected_class, dim=1)
        selected_class_normalized=selected_class
        MMlogit7 = selected_class_normalized
        # end

        if return_only_output:
            return MMlogit7

        if criter == 'CE':  # 常规
            criterion0 = torch.nn.CrossEntropyLoss(reduction='none')
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')  # BCEWithLogitsLoss 用于多标签分类
        elif criter == 'FL':  # 焦点损失
            criterion0 = FocalLoss(reduction='none')
            criterion = FocalLoss(reduction='none')
        elif criter == 'WCE':  # 加权损失
            # 假设你有一个用于权重的张量 weights
            weights = torch.tensor(
                [1 / 3.48, 1 / 20.37, 1 / 2.45, 1 / 23.62, 1 / 4.16, 1 / 2.78, 1 / 89.0]).cuda()  # 根据你的数据分布调整权重
            criterion0 = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')  # BCEWithLogitsLoss 也可以添加权重参数

        if label7 is not None:
            if 'PTBXL' in dataset:
                weights = torch.tensor(
                    [1, 2, 1, 1.25, 1, 1, 1]).cuda()  # TB-net-new_13
                    # [1, 1.7, 1, 1.25, 1, 1, 1]).cuda()  # TB-net-new_11
                    # [1, 1.5, 1, 1.25, 1, 1, 1]).cuda()  # TB-net-new_8 12
                    # [1.2, 1.25, 1.2, 1.25, 1, 1, 1]).cuda()  # TB-net-new_7
                    # [1, 1.25, 1, 1.25, 1, 1, 1]).cuda()  # TB-net-new_6
                    # [1, 1.5, 1, 1.5, 1, 1, 1]).cuda()  # TB-net-new_5
                    # [1.5, 1, 1.5, 1, 1, 1, 1]).cuda()  # TB-net-new_4
                    # [1.5, 1, 1.5, 1, 1.5, 1.5, 0.9]).cuda()# TB-net-new_3
                # [1, 1, 1, 1, 1, 1, 1]).cuda()  # TB-net-new_2
                    # [3, 1, 3, 1 , 3,3, 1 ]).cuda()# TB-net
                    # [2, 1, 2, 1, 1.5, 3, 1]).cuda()
                Loss7 = torch.mean(weights*criterion(MMlogit7, label7.to(torch.float)))
                # Loss7 = torch.mean(criterion(MMlogit7, label7.to(torch.float)))
            else:
                Loss7 = torch.mean(criterion0(MMlogit7, label7))
        else:
            Loss7 = torch.tensor(0.0, device=device)  # 如果没有标签，则损失为0
        # MMLoss = 0.013 * Loss2 + 0.19 * Loss5 + 0.79 * Loss7

        MMLoss = Loss7
        # return MMLoss, MMlogit7
        return MMLoss, MMlogit7

    def get_only_output(self, batch, num_classes=7, label7=None, dataset='PTBXL', criter='CE'):
        return self.forward(batch, num_classes=num_classes, label7=label7, dataset=dataset, criter=criter,
                            return_only_output=True)
