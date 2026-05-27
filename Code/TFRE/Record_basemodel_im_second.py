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

# class RelativePositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(RelativePositionalEncoding, self).__init__()
#         self.d_model = d_model
#         self.max_len = max_len
#         self.relative_positions = nn.Parameter(torch.randn(2 * max_len - 1, d_model))
#
#     def forward(self, x):
#         batch_size, seq_len, d_model = x.size()
#         assert d_model == self.d_model, "输入的嵌入维度必须与模型维度匹配"
#         pos_encoding = torch.zeros((batch_size, seq_len, d_model), device=x.device)
#         for i in range(seq_len):
#             for j in range(seq_len):
#                 pos_index = i - j + self.max_len - 1
#                 pos_encoding[:, i, :] += self.relative_positions[pos_index, :]
#         x = x + pos_encoding
#         return x
#
# class TransformerFeatureExtractor(nn.Module):
#     def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim, max_len=5000, dropout=0.5):
#         super(TransformerFeatureExtractor, self).__init__()
#         self.linear_in = nn.Linear(input_dim, hidden_dim)
#         self.positional_encoding = RelativePositionalEncoding(hidden_dim, max_len)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dropout=dropout,
#             dim_feedforward=hidden_dim * 4
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.linear_out = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x, mask=None):
#         x = x.unsqueeze(-1)  # 将形状从 (64, 5000) 转换为 (64, 5000, 1)
#         x = self.linear_in(x)  # 投影到 hidden_dim 维度: (64, 5000, hidden_dim)
#         x = self.positional_encoding(x)
#         x = self.layer_norm(x)
#         x = self.dropout(x)
#         x = self.transformer_encoder(x, src_key_padding_mask=mask)
#         x = self.linear_out(x)  # 输出形状: (64, 5000, output_dim)
#         x = x.view(x.size(0), 32, -1)  # 重塑输出为目标形状 (64, 32, 1)
#         return x
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
#
#
# # 在多个批次生成边
# def repeat_edge_index(edge_index, batch_size, num_nodes):
#     edge_index = edge_index.repeat(1, batch_size)
#     for i in range(batch_size):
#         start = i * edge_index.size(1) // batch_size
#         end = (i + 1) * edge_index.size(1) // batch_size
#         edge_index[:, start:end] += i * num_nodes
#
#     # 检查边索引是否超出范围
#     max_index = edge_index.max().item()
#     min_index = edge_index.min().item()
#     if max_index >= batch_size * num_nodes:
#         raise ValueError(f"边索引超出范围: 最大索引 {max_index}, 最大允许值 {batch_size * num_nodes - 1}")
#     if min_index < 0:
#         raise ValueError(f"边索引小于零: 最小索引 {min_index}")
#
#     return edge_index
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
            [mSEnet() for view in range(self.views)])  # ������������������views��mSE���磨ÿ������ƥ��һ��������
        # self.FeatureEncoder1 = nn.ModuleList(
        #     [TransformerFeatureExtractor(input_dim=5000, num_heads=4, num_layers=10,
        #                                                            hidden_dim=128, output_dim=32) for view in range(self.views)])

        # self.FeatureEncoder1 = nn.ModuleList(
        #     [mSEnet_trans() for view in range(self.views)])
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

        # 多分枝输出
        # self.MMClassifiers_2 = nn.ModuleList([
        #     nn.Sequential(LinearLayer(self.views * hidden_dim[-1], 1)) for _ in range(2)
        # ])
        # self.MMClassifiers_5 = nn.ModuleList([
        #     nn.Sequential(LinearLayer(self.views * hidden_dim[-1], 1)) for _ in range(5)
        # ])
        # self.MMClassifiers_7 = nn.ModuleList([
        #     nn.Sequential(LinearLayer(self.views * hidden_dim[-1], 1)) for _ in range(7)
        # ])

        # self.Part_Clasifier = nn.ModuleList([
        #     nn.Linear(12 * 32, 1),  # 对应 [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]
        #     nn.Linear(12 * 32, 1),  # 对应 [0, 1, 3, 4, 6, 7, 8, 9, 10, 11]
        #     nn.Linear(12 * 32, 1),  # 对应 [0, 1, 3, 4, 7, 8, 9, 10, 11]
        #     nn.Linear(12 * 32, 1),  # 对应 [1, 2, 3, 5, 6, 7, 9]
        #     nn.Linear(12 * 32, 1),  # 对应 [1, 2, 3, 4, 5, 10, 11]
        #     nn.Linear(12 * 32, 1),  # 对应 [0, 1, 3, 4, 5, 7, 9, 10, 11]
        #     nn.Linear(12 * 32, 1)  # 对应所有导联
        # ])
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
        self.MMClasifier_Dout = nn.Sequential(*self.MMClasifier)

    # def forward(self, data_list, label2=None, label5=None, label7=None, dataset='PTB'):
    def forward(self, batch, num_classes=7, label7=None, dataset='PTBXL', criter='CE', return_only_output=False):
        # def forward(self, data_list, num_classes, labels=None, dataset='PTBXL'):
        # �����
        FeatureInfo, feature, feature_lead, TCPLogit, TCPConfidence, lead_weight, TCPpreLogit = dict(), dict(), dict(), dict(), dict(), dict(), dict()
        feature_less = dict()
        view_weight_sigmoid = torch.empty(3, 3)
        # if isinstance(batch, list):
        #     # 训练期间
        #     data_list = batch
        #     if all(isinstance(x, torch.Tensor) for x in data_list):
        #         stacked_data = torch.stack(data_list)  # [12, 64, 5000]
        #     else:
        #         raise ValueError("data_list must be a list of Tensors")
        # elif isinstance(batch, torch.Tensor):
        #     # 使用 IntegratedGradients 方法时
        #     stacked_data = batch  # 假设 batch 本身就是需要的张量
        #
        # Ensure batch is in the correct format
        if isinstance(batch, list):
            data_list = batch
        elif isinstance(batch, torch.Tensor):
            dim = batch.dim()
            if dim == 3:
                # print(f"batch.shape: {batch.shape}")
                data_list = torch.unbind(batch, dim=1)
                # print(f"data_list[0].shape: {data_list[0].shape}")
            elif dim == 4:
                # print(f"batch.shape: {batch.shape}")
                data_list = batch.squeeze(0)
                data_list = torch.unbind(data_list, dim=0)
                # print(f"data_list[0].shape: {data_list[0].shape}")
        # 打印每个 data_list 元素的形状
        # for i, data in enumerate(data_list):
        #     print(f"Shape of data_list[{i}]: {data.shape}")
        # ... (rest of your forward function)

        # 导联间特征
        # SOLOfeature = torch.stack(data_list, dim=-1)  # [128,5000,12]
        # SOLOfeature = torch.transpose(SOLOfeature, 1, 2)  # [128,12,5000]
        # downfeature = self.FeatureEncoder_DKR(SOLOfeature)  # [128,1,5000]
        # downfeature = torch.squeeze(downfeature, dim=1)  # [128,5000]

        # print(downfeature.shape)#torch.Size([64, 16, 12, 1])

        # downfeature = downfeature.flatten(start_dim=1, end_dim=3)

        # 导联间特征
        # output_list = []
        # for view in range(self.views - 3):
        #     output, _ = self.attention(feature[view], feature[view], feature[view])
        #     output_list.append(output)

        # MIfeature = dict()
        # for view in range(4):
        #     if view == 0:  # AMI/ASMI:7-10
        #         MIfeature[view] = torch.stack([data_list[6], data_list[7], data_list[8], data_list[9]],
        #                                       dim=1)  # [64,4,5000]
        #     elif view == 1:  # ALMI:9-12/1/5
        #         MIfeature[view] = torch.stack(
        #             [data_list[8], data_list[9], data_list[10], data_list[11], data_list[0], data_list[4]],
        #             dim=1)  # [64,6,5000]
        #     elif view == 2:  # ILMI:2/3/5/11/12
        #         MIfeature[view] = torch.stack([data_list[1], data_list[2], data_list[4], data_list[10], data_list[11]],
        #                                       dim=1)  # [64,5,5000]
        #     else:  # IMI:2/3/6
        #         MIfeature[view] = torch.stack([data_list[1], data_list[2], data_list[5]], dim=1)  # [64,3,5000]

        # for view in range(4):
        #     value = MIfeature[view].permute(0,2,1)  # [64,32,12]
        #     output = self.itransformer(value, None, None, None)  # [64,32,12]
        #     output = output.permute(0, 2, 1)  # [64,12,32]
        #     MIfeature[view] = output
        # Ⅰ、Ⅱ、Ⅲ、aVR、 aVL、aVF、V1、V2、V3、V4、V5、V6
        # AMI/ASMI:V1-V4,ALMI:V3-V6/I/aVL,IMI:II/III/aVF,ILMI:II/III/aVF/V5/V6,
        # AMI/ASMI:7-10,ALMI:9-12/1/5,IMI:2/3/6,ILMI:2/3/5/11/12,

        # gcn图
        # 定义导联之间的连接
        # Anterior_septal_edges = [(6, 7), (7, 8), (8, 9)]
        # Anterolateral_edges = [(9, 10), (10, 11), (5, 9)]
        # Inferior_edges = [(0, 3), (3, 8)]
        #心脏分区
        # Anterior_septal_edges = [(6, 7),  (8, 9)]
        # Anterolateral_edges = [(0, 10), (10, 11), (0, 4),(0,11),(10,4)]
        # Inferior_edges = [(1,2),(2,5)]
        #shap图构图1
        # Anterior_septal_edges = [(6, 7)]
        # Anterolateral_edges = [(9, 10), (10, 11), (2, 11),(3,10)]
        # Inferior_edges = [(3, 7), (4, 7)]
        # PTBXL shap图构图最好的构图
        Anterior_septal_edges = [(6, 7)]
        Anterolateral_edges = [(10, 11)]
        Inferior_edges = [(3,0)]
        # PTB shap图构图最好的构图
        # Anterior_septal_edges = [(6, 9)]
        # Anterolateral_edges = [(10, 0)]
        # Inferior_edges = [(5, 7)]
        # 合并所有边
        all_edges = Anterior_septal_edges + Anterolateral_edges + Inferior_edges

        # 去重（因为某些边可能在不同的类型中被重复定义）
        all_edges = list(set(all_edges))

        # 转换为PyTorch tensor格式
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_index = edge_index.to(device)
        # for view in range(4):
        #     if view == 0:  # AMI/ASMI:7-10
        #         # print(stacked_data[:, [6, 7, 8, 9], :].shape)
        #         # print(MIfeature[view].shape)
        #         stacked_data[:, [6, 7, 8, 9], :] += MIfeature[view]  # torch.Size([64, 4, 32])
        #     elif view == 1:  # ALMI:9-12/1/5
        #         stacked_data[:, [8, 9, 10, 11, 0, 4], :] += MIfeature[view]
        #     elif view == 2:  # ILMI:2/3/5/11/12
        #         stacked_data[:, [1, 2, 4, 10, 11], :] += MIfeature[view]
        #     else:  # IMI:2/3/6
        #         stacked_data[:, [1, 2, 5], :] += MIfeature[view]
        # print(stacked_data.shape)  # [64,12,32]
        # stacked_data = stacked_data.permute(0, 2, 1)  # torch.Size([64, 32, 12])
        # stacked_data_resized = stacked_data.flatten(start_dim=1, end_dim=2)  # torch.Size([64, 32 * 12])
        # # # 应用全连接层降维
        # # # 重塑输出以匹配所需的形状[批次大小, 节点数, 特征数]
        # x = x.view(-1, 12, 32)

        # if return_only_output:
        #     for view in range(self.views):
        #         print(data_list[view].shape)
        #         data_list[view] = torch.squeeze(data_list[view], dim=0)
        #         # # data_list[view] = data_list[view].permute(1, 0, 2)
        #         # print(" 1")
        #         # print(data_list[view].shape)

        for view in range(self.views):
            # feature[view] = self.FeatureEncoder1[view](data_list[view])  # [64,32,1]
            feature_less[view] = self.FeatureEncoder2[view](data_list[view]) # origin now
            # feature_less[view] = self.FeatureEncoder_trans[view](data_list[view])
            # feature_lead[view] = self.FeatureEncoder1[view](data_list[view])
            # feature_lead[view] = self.FeatureEncoder2[view](data_list[view])
            # 将data_list[view]融合为单导联
            # feature[view] = feature[view] + feature_less[view]
            feature_less[view] = feature_less[view].flatten(start_dim=1, end_dim=2)
            # feature[view] = feature[view].flatten(start_dim=1, end_dim=2)  # [64,32]
        #  feature[12,64,32]
        MIfeature, MIfeature_1 = dict(), dict()
        # 从data_list中提取特定的张量并组合
        # tensor_2 = torch.stack([data_list[10], data_list[11]], dim=1)  # (64, 2, 5000)
        # tensor_3 = torch.stack([data_list[7], data_list[8]], dim=1)  # (64, 3, 5000)
        # 使用 Transformer 特征提取器进行编码
        # tr_tensor_2 = self.transformer_extractor_2(tensor_2)
        # tr_tensor_3 = self.transformer_extractor_3(tensor_3)
        # tr_tensor_2 = self.tr_2(tr_tensor_2)
        # tr_tensor_3 = self.tr_3(tr_tensor_3)
        # 使用 SEresnet 特征提取器进行编码
        # encoded_tensor_2 = self.cn_1(tensor_2)  # (64, 2, 32)
        # encoded_tensor_3 = self.cn_2(tensor_3)  # (64, 3, 32)
        # encoded_tensor_2 = encoded_tensor_2.permute(0, 2, 1)
        # encoded_tensor_3 = encoded_tensor_3.permute(0, 2, 1)
        # encoded_tensor_2 = self.db_1(encoded_tensor_2)
        # encoded_tensor_3 = self.db_2(encoded_tensor_3)

        # encoded_tensor_2 += tr_tensor_2
        # encoded_tensor_3 += tr_tensor_3
        # encoded_tensor_2 = encoded_tensor_2.to(device)
        # encoded_tensor_3 = encoded_tensor_3.to(device)


        # stacked_data_all = torch.stack([i for i in feature.values()])
        # stacked_data_all = stacked_data_all.permute(1, 0, 2).to(device)
        # stacked_data_all = stacked_data_all.flatten(start_dim=1, end_dim=2)
        # stacked_data_all = F.dropout(stacked_data_all, self.dropout, training=self.training).to(device)
        # MMlogit_all = self.MMClasifier7(stacked_data_all).to(device)
        # MINOR OR MORE
        MMfeature = torch.cat([i for i in feature.values()], dim=1)

        stacked_data = torch.stack([i for i in feature_less.values()])  # [12, 64, 32]
        stacked_data = stacked_data.permute(1, 0, 2)  # [64, 12, 32]
        stacked_data = stacked_data.to(device)
        # 使用 Transformer 特征提取器进行编码
        tr_tensor_2 = self.transformer_extractor_2(stacked_data)
        tr_tensor_2 = self.tr_2(tr_tensor_2)
        # tr_tensor_3 = self.transformer_extractor_3(stacked_data)
        # tr_tensor_3 = self.tr_3(tr_tensor_3)
        # stacked_data = stacked_data+tr_tensor_2
        #下面四行是加上几个并联导联的独特编码
        # stacked_data[:, 10, :] += encoded_tensor_2[:, 0, :]
        # stacked_data[:, 11, :] += encoded_tensor_2[:, 1, :]
        # stacked_data[:, 7, :] += encoded_tensor_3[:, 0, :]
        # stacked_data[:, 8, :] += encoded_tensor_3[:, 1, :]


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
        # stacked_data += x+x_1+ tr_tensor_2
        x += x_1+tr_tensor_2+stacked_data

        # stacked_data = stacked_data + tr_tensor_2 + tr_tensor_3

        selected_lead = []
        # stacked_data += x
        # # GAT,这里stacker_data不包含GCN信息
        # Anterior_septal_edges = [(6, 7), (7, 8)]
        # Anterolateral_edges = [(10, 11)]
        # # Inferior_edges = [(0, 3), (3, 4),(4,7)]
        # # 合并所有边
        # # all_edges = Anterior_septal_edges + Anterolateral_edges + Inferior_edges
        # all_edges = Anterior_septal_edges + Anterolateral_edges
        # # 去重（因为某些边可能在不同的类型中被重复定义）
        # all_edges = list(set(all_edges))
        #
        # # 转换为PyTorch tensor格式
        # edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        # edge_index = edge_index.to(device)
        # # 构建全连接图
        # gat_edge = torch.tensor([[i, j] for i in range(12) for j in range(12) if i != j],
        #                         dtype=torch.long).t().contiguous()
        # gat_edge = gat_edge.to(device)
        #
        # # 将每个图的数据转换为 PyTorch Geometric 的 Data 对象
        # data_list = []
        # for i in range(stacked_data.shape[0]):
        #     data = Data(x=stacked_data[i], edge_index=edge_index)
        #     data_list.append(data)
        #
        # # 创建批次数据
        # batch = Batch.from_data_list(data_list)
        #
        # batch.x = self.gat1(batch.x, batch.edge_index)
        # batch.x = self.gat1_0(batch.x)
        # batch.x = self.gat2(batch.x, batch.edge_index)
        # batch.x = self.gat2_0(batch.x)
        # x = batch.x.view(-1, 12, 32)
        # x_2 = self.BN_gat(x)
        # stacked_data += x_1 + x_2
        # stacked_data += x_1
        # x = self.SE_gat(x)
        # # output += x
        # selected_lead.append(stacked_data[:, [6, 7], :].flatten(start_dim=1, end_dim=2))  # torch.Size([64, 32 * 2])
        # selected_lead.append(stacked_data[:, [7, 9], :].flatten(start_dim=1, end_dim=2))  # torch.Size([64, 32 * 3])
        # selected_lead.append(stacked_data[:, [0, 3, 4, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))  # torch.Size([64, 32 * 7])
        # selected_lead.append(stacked_data[:, [1, 2, 5, 6], :].flatten(start_dim=1, end_dim=2))  # torch.Size([64, 32 * 5])
        # selected_lead.append(stacked_data[:, [2, 5, 10, 11], :].flatten(start_dim=1, end_dim=2)) # torch.Size([64, 32 * 4])
        # selected_lead.append(stacked_data[:, [0, 1, 3, 4, 7, 9, 11], :].flatten(start_dim=1, end_dim=2))  # torch.Size([64, 32 * 7])
        # selected_lead.append(stacked_data[:, [1, 6, 7, 9, 10], :].flatten(start_dim=1, end_dim=2))
        # # selected_lead.append(stacked_data[:, [:], :])

        # origin
        selected_lead.append(x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        selected_lead.append(x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        selected_lead.append(x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        selected_lead.append(x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))
        selected_lead.append(stacked_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], :].flatten(start_dim=1, end_dim=2))



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

        # pred_class_pre6 = torch.cat(selected_class, dim=1).to(device)
        # pred_class_7 = self.Part_Clasifier1[0](pred_class_pre6)
        # selected_class.append(pred_class_7)
        # # 对 selected_class 进行 L2 正则化
        # norm_selected = torch.norm(selected_class, p=2, dim=1, keepdim=True)
        # selected_class_normalized = selected_class / norm_selected.clamp(min=1e-8)
        #
        # # 对 MMlogit_all 进行 L2 正则化
        # norm_MMlogit = torch.norm(MMlogit_all, p=2, dim=1, keepdim=True)
        # MMlogit_all_normalized = MMlogit_all / norm_MMlogit.clamp(min=1e-8)

        # selected_class_normalized[:, [0, 2, 4, 5]] = 0.5+selected_class_normalized[:, [0, 2, 4, 5]]
        # selected_class_normalized=selected_class
        # MMlogit_all_normalized=MMlogit_all
        # 假设 selected_class_normalized 和 MMlogit_all_normalized 是你的两个张量

        # # 使用门控机制
        # threshold = 0.8  # 置信度阈值，可以根据具体需求调整
        # confidence_scores = torch.max(selected_class_normalized, dim=1, keepdim=True)[0]  # 假设这是置信度
        # MMlogit7 = torch.where(confidence_scores > threshold, selected_class_normalized, MMlogit_all_normalized)
        #赋予少数类分类器更高的权重
        weight_minor = 0.7  # 设定少数类的权重
        weight_major = 0.3  # 设定多数类的权重
        # MMlogit7 = weight_minor * selected_class_normalized + weight_major * MMlogit_all_normalized
        # MMlogit7 = selected_class_normalized
        # #使用带有温度缩放的 Softmax
        # temperature = 0.5  # 温度参数，小于 1 表示放大影响
        # selected_class_normalized = torch.softmax(selected_class_normalized / temperature, dim=1)
        # MMlogit7 = selected_class_normalized + MMlogit_all_normalized
        #三种方法组合使用
        # # 1. 温度缩放
        # temperature_minor = 0.5  # 少数类温度参数
        # temperature_major = 1.0  # 多数类温度参数
        # selected_class_normalized = torch.softmax(selected_class_normalized / temperature_minor, dim=1)
        # MMlogit_all_normalized = torch.softmax(MMlogit_all_normalized / temperature_major, dim=1)
        # # 2. 赋予少数类更高的权重
        # weight_minor = 0.7  # 少数类权重
        # weight_major = 0.3  # 多数类权重
        # MMlogit7 = weight_minor * selected_class_normalized + weight_major * MMlogit_all_normalized
        # # 3. 使用门控机制
        # threshold = 0.8  # 置信度阈值
        # confidence_scores = torch.max(selected_class_normalized, dim=1, keepdim=True)[0]  # 计算少数类分类器的置信度
        # # 使用门控机制保留少数类的结果
        # MMlogit7 = torch.where(confidence_scores > threshold, selected_class_normalized, MMlogit7)



        # 以下是原七分类过程，可在if return_only_output:前一并解除注释
        # output = stacked_data
        # # 融合策略
        # # output = torch.cat((x, output), dim=2) # 拼接维度
        #
        # # gcn_fea = self.gcn_clf(x) # 线性变换
        # # output = output + x
        # # stacked_data = stacked_data.to(device)
        # # #
        # # # # 重塑输出以匹配所需的形状[批次大小, 节点数, 特征数]
        # # x = x.view(-1, 12, 108)
        # # x = self.BN_gat(x)
        # # x = self.SE_gcn(x)
        # # output += x
        #
        # # gat_edge = generate_complete_graph_edges(12)
        # # gat_edge = repeat_edge_index(gat_edge, stacked_data.size(0), 12)
        # # gat_edge = gat_edge.to(device)
        #
        # # x = stacked_data.reshape(-1, 108)
        # # x = self.gat1(x, gat_edge)
        # # x = self.gat1_0(x)
        # # x = self.gat2(x, gat_edge)
        # # x = self.gat2_0(x)
        # # x = x.view(-1, 12, 108)
        # # x = self.BN_gat(x)
        # # # x = self.SE_gat(x)
        # # output += x
        #
        # stacked_data_resized = output.flatten(start_dim=1, end_dim=2)  # torch.Size([64, 32 * 12])
        # MMfeature = stacked_data_resized
        # MMfeature = torch.cat([i for i in feature.values()], dim=1)  # [64,16*12]
        # # MMfeature = downfeature + MMfeature # 双网络
        # # MMfeature_7 = MMfeature + stacked_data_resized


        # # DBW w/o
        # MMfeature = stacked_data.flatten(start_dim=1, end_dim=2)
        # MMfeature = F.dropout(MMfeature, self.dropout, training=self.training)
        # MMlogit7 = self.MMClasifier7(MMfeature)
        # MMfeature_7 =  F.dropout(MMfeature_7, self.dropout, training=self.training)

        # MMlogit = self.MMClasifier(MMfeature)
        # MMlogit5 = self.MMClasifier5(MMfeature)
        # MMlogit7 = self.MMClasifier7(MMfeature)

        if return_only_output:
            return MMlogit7
        # 原函数
        # criterion0 = torch.nn.CrossEntropyLoss(reduction='none')
        # criterion = torch.nn.BCEWithLogitsLoss()  # BCEloss���ڶ����ཻ���أ��ú���������BCE Loss�Լ�Sigmoid�����㣬
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

        # if 'PTBXL' in dataset:
        #     # Loss2 = torch.mean(criterion0(MMlogit, label2))
        #     # Loss5 = torch.mean(criterion(MMlogit5, label5.to(torch.float)))
        #     Loss7 = torch.mean(criterion(MMlogit7, label7.to(torch.float)))
        # else:
        #     # Loss2 = torch.mean(criterion0(MMlogit, label2))
        #     # Loss5 = torch.mean(criterion0(MMlogit5, label5))
        #     Loss7 = torch.mean(criterion0(MMlogit7, label7))

        if label7 is not None:
            if 'PTBXL' in dataset:
                weights = torch.tensor(
                    [1, 1.5, 1, 1.25, 1, 1, 1]).cuda()  # TB-net-new_8
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

        # 多分支输出
        # classifier = self.classifiers[str(num_classes)]
        # logits = torch.cat([classifiers(MMfeature) for classifiers in classifier], dim=1)
        # MMlogit = torch.cat([classifier(MMfeature) for classifier in self.MMClassifiers_2], dim=1)
        # MMlogit5 = torch.cat([classifier(MMfeature) for classifier in self.MMClassifiers_5], dim=1)
        # MMlogit7 = torch.cat([classifier(MMfeature) for classifier in self.MMClassifiers_7], dim=1)
        # if 'PTBXL' in dataset:
        #     if num_classes == 2:
        #         Loss2 = torch.mean(criterion(logits, labels.to(torch.float)))
        #         MMLoss = Loss2
        #         return MMLoss, logits
        #     elif num_classes == 5:
        #         Loss5 = torch.mean(criterion(logits, labels.to(torch.float)))
        #         MMlogit7 = Loss5
        #         MMLoss = Loss5
        #         return MMLoss, logits
        #     else:
        #         Loss7 = torch.mean(criterion(logits, labels.to(torch.float)))
        #         MMlogit7 = Loss7
        #         MMLoss = Loss7
        #         return MMLoss, logits
        # else:
        #     if num_classes == 2:
        #         Loss2 = torch.mean(criterion0(logits, labels))
        #         MMlogit7 = Loss2
        #         MMLoss = Loss2
        #         return MMLoss, logits
        #     elif num_classes == 5:
        #         Loss5 = torch.mean(criterion0(logits, labels))
        #         MMlogit7 = Loss5
        #         MMLoss = Loss5
        #         return MMLoss, logits
        #     else:
        #         Loss7 = torch.mean(criterion0(logits, labels))
        #         MMlogit7 = Loss7
        #         MMLoss = Loss7
        #         return MMLoss, logits

    def get_only_output(self, batch, num_classes=7, label7=None, dataset='PTBXL', criter='CE'):
        return self.forward(batch, num_classes=num_classes, label7=label7, dataset=dataset, criter=criter,
                            return_only_output=True)
