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
        self.views = len(in_dim)
        self.dropout = dropout

        self.FeatureEncoder2 = nn.ModuleList([mSEnet() for _ in range(self.views)])
        # Transformer 特征提取器
        self.transformer_extractor_2 = TransformerFeatureExtractor(input_dim=32, num_heads=4, num_layers=4,
                                                                   hidden_dim=128, output_dim=32)
        self.tr_2 = nn.Sequential(
            nn.BatchNorm1d(12),
            nn.LeakyReLU(0.3)
        )
        self.tr = nn.Sequential(nn.BatchNorm1d(12), nn.LeakyReLU(0.3))

        self.SE_gcn = SE_Module(12, dim=1)
        self.BN_gcn = nn.BatchNorm1d(12)
        self.BN_gat = nn.BatchNorm1d(12)
        self.learnable_adj = nn.Parameter(torch.randn(12, 12))

        self.gcn1 = pyg_nn.GraphConv(32, 128)
        self.gcn1_0 = nn.Sequential(pyg_nn.BatchNorm(12), nn.LeakyReLU(0.3))
        self.gcn2 = pyg_nn.GraphConv(128, 512)
        self.gcn2_0 = nn.Sequential(pyg_nn.BatchNorm(12), nn.LeakyReLU(0.3))
        self.gcn3 = pyg_nn.GraphConv(512, 32)
        self.gcn3_0 = nn.Sequential(pyg_nn.BatchNorm(12), nn.LeakyReLU(0.3))

        # >=29
        self.conv1 = pyg_nn.SAGEConv(32, 128)
        self.conv2 = pyg_nn.SAGEConv(128, 512)
        self.conv3 = pyg_nn.SAGEConv(512, 32)
        self.bn1 = pyg_nn.BatchNorm(12)
        self.bn2 = pyg_nn.BatchNorm(12)
        self.bn3 = pyg_nn.BatchNorm(12)
        self.act = nn.LeakyReLU(0.3)

        self.gat1 = pyg_nn.GATConv(32, 32, heads=12, concat=True)
        self.gat1_0 = nn.Sequential(pyg_nn.BatchNorm(32 * 12), nn.LeakyReLU(0.5))
        self.gat2 = pyg_nn.GATConv(32 * 12, 32, heads=1, concat=True)
        self.gat2_0 = nn.Sequential(pyg_nn.BatchNorm(32), nn.LeakyReLU(0.5))

        self.MMClassifier2 = nn.Linear(12 * 32, 2)
        self.GCNClassifier2 = nn.Linear(12 * 32, 2)
        self.MMClassifier_1_3 = nn.Linear(12 * 32, 2)
        self.FourClassClassifier = nn.Linear(12 * 32, 4)

        self.ensemble_head = nn.Linear(12 * 32, 7)
        self.leakyrelu = nn.LeakyReLU(0.3)

        self.fusion_gate_tr = nn.Sequential(
            nn.Linear(32, 32),
            nn.Sigmoid()
        )

        self.fusion_gate_gat = nn.Sequential(
            nn.Linear(32, 32),
            nn.Sigmoid()
        )

        self.fusion_logits = nn.Linear(12 * 32, 7)

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
        Inferior_edges = [(3, 0)]
        # PTB shap图构图最好的构图 46
        # Anterior_septal_edges = [(6, 9)]
        # Anterolateral_edges = [(10, 0)]
        # Inferior_edges = [(5, 7)]
        # changjian心脏分区 39 40
        # Anterior_septal_edges = [(6, 7),  (8, 9)]
        # Anterolateral_edges = [(0, 10), (10, 11), (0, 4),(0,11),(10,4)]
        # Inferior_edges = [(1,2),(2,5)]
        # 合并所有边
        all_edges = Anterior_septal_edges + Anterolateral_edges + Inferior_edges

        # 去重（因为某些边可能在不同的类型中被重复定义）
        all_edges = list(set(all_edges))

        # 转换为PyTorch tensor格式
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        for view in range(self.views):
            feature_less[view] = self.FeatureEncoder2[view](data_list[view])  # origin now
            feature_less[view] = feature_less[view].flatten(start_dim=1, end_dim=2)
        MIfeature, MIfeature_1 = dict(), dict()
        # MINOR OR MORE
        MMfeature = torch.cat([i for i in feature_less.values()], dim=1)
        # #9 10
        # MMfeature = self.MMClasifier_Dout(MMfeature)
        stacked_data = torch.stack([i for i in feature_less.values()])  # [12, 64, 32]
        stacked_data = stacked_data.permute(1, 0, 2)  # [64, 12, 32]
        device = next(self.parameters()).device
        stacked_data = stacked_data.to(device)
        # 使用 Transformer 特征提取器进行编码
        tr_tensor = self.transformer_extractor_2(stacked_data)
        tr_tensor = self.tr_2(tr_tensor)


        # GCN
        edge_index = edge_index.to(stacked_data.device) #  flopcount change
        x = self.gcn1(stacked_data, edge_index)
        x = self.gcn1_0(x)
        x = self.gcn2(x, edge_index)
        x = self.gcn2_0(x)
        x = self.gcn3(x, edge_index)
        x = self.gcn3_0(x)
        x = self.SE_gcn(x)
        # # 应用全连接层降维
        # # 重塑输出以匹配所需的形状[批次大小, 节点数, 特征数]
        x = x.view(-1, 12, 32)
        # stacked_data += x_1
        # GAT
        # 21 >=29 start
        soft_adj = F.softmax(self.learnable_adj, dim=-1)
        gat_edge = soft_adj.nonzero(as_tuple=False).t().contiguous()
        gat = self.act(self.bn1(self.conv1(stacked_data, gat_edge)))
        gat = self.act(self.bn2(self.conv2(gat, gat_edge)))
        gat = self.act(self.bn3(self.conv3(gat, gat_edge)))
        gat = gat.view(-1, 12, 32)
        #
        # gat_edge = generate_complete_graph_edges(12).to(device)
        # gat = self.gat1(stacked_data.reshape(-1, 32), gat_edge)
        # gat = self.gat1_0(gat)
        # gat = self.gat2(gat, gat_edge)
        # gat = self.gat2_0(gat)
        # gat = gat.view(-1, 12, 32)
        # gat = self.BN_gat(gat)

        # 22 >=29
        gate_tr = self.fusion_gate_tr(tr_tensor)
        gate_gat = self.fusion_gate_gat(gat)
        x_fused = x * (gate_tr + gate_gat) + tr_tensor * (1 - gate_tr) + gat * (1 - gate_gat) + stacked_data
        # x_fused =  tr_tensor * (1 - gate_tr) + gat * (1 - gate_gat) + stacked_data # nogcn 43
        # x_fused = x * (gate_tr + gate_gat) + gat * (1 - gate_gat) + stacked_data # notr 44
        # x_fused = x * (gate_tr + gate_gat) + tr_tensor * (1 - gate_tr) + stacked_data # nosage 45
        # x_fused = x  + stacked_data  # onlygcn 47
        # x_fused = tr_tensor + stacked_data  # onlytr 48
        # x_fused = gat + stacked_data  # onlytr 49
        # x_fused = stacked_data # 31 37xiaorong
        # x_fused = x + gat + tr_tensor  # [B, 12, 32]

        x_flat = x_fused.flatten(start_dim=1)  # [B, 12*32]

        # 15
        # Stage 1: MMfeature 2-way classification (label 6 vs rest)
        # 合并所有预测路径 → 构建 final_logits
        logit6 = self.MMClassifier2(MMfeature)  # 二分类第2类即认为是label=6
        # logit6 = self.MMClassifier2(x_flat)  # 32
        logit13 = self.MMClassifier_1_3(MMfeature)  # 输出 [B, 2] → label 1, 3
        logit0245 = self.FourClassClassifier(x_flat)  # 输出 [B, 4] → label 0, 2, 4, 5
        # >=29
        logitNum = self.GCNClassifier2(x_flat)
        # 构建 final_logits，按列位置拼接
        final_logits = torch.full((stacked_data.size(0), 7), -1e9, device=stacked_data.device)

        # logits 路由合并
        final_logits[:, 6] = logit6[:, 1]  # label 6
        final_logits[:, 1] = logit13[:, 0]  # label 1
        final_logits[:, 3] = logit13[:, 1]  # label 3
        final_logits[:, 0] = logit0245[:, 0]  # label 0
        final_logits[:, 2] = logit0245[:, 1]  # label 2
        final_logits[:, 4] = logit0245[:, 2]  # label 4
        final_logits[:, 5] = logit0245[:, 3]  # label 5

        # final_logits = 1.5 * final_logits + 0 * self.ensemble_head(MMfeature) # PTBcontrast
        # final_logits = 0.7 * final_logits + 0.3 * self.ensemble_head(x_flat) # 22
        # final_logits = 0.3 * final_logits + 0.7 * self.ensemble_head(x_flat)  # 23
        # final_logits = 0.2 * final_logits + 0.8 * self.ensemble_head(x_flat)  # 24
        # final_logits = 0.5 * final_logits + 0.5 * self.ensemble_head(x_flat)  # 25
        final_logits = 0.5 * final_logits + 0.5 * self.ensemble_head(MMfeature)  # 29 38+
        # final_logits = 0.7 * final_logits + 0.3 * self.ensemble_head(MMfeature)  # 33
        # final_logits = final_logits  # 35 xiaorong 1
        # final_logits = self.ensemble_head(x_flat)  # 36 xiaorong 2
        # final_logits = 0.3 * final_logits + 0.7 * self.ensemble_head(MMfeature)  # 34 37
        # gate_logits = self.fusion_logits(x_flat)
        # final_logits = gate_logits * final_logits + (1-gate_logits) * self.ensemble_head(MMfeature)  # 30 lose
        # logits 路由合并 20
        # final_logits[:, 6] = logit6 - (logit0245[:, 0]+logit0245[:, 1]+logit0245[:, 2]+logit0245[:, 3]) # label 6
        # final_logits[:, 1] = logit13[:, 0] - (logit0245[:, 0]+logit0245[:, 1]+logit0245[:, 2]+logit0245[:, 3]) # label 1
        # final_logits[:, 3] = logit13[:, 1] - (logit0245[:, 0]+logit0245[:, 1]+logit0245[:, 2]+logit0245[:, 3]) # label 3
        # final_logits[:, 0] = logit0245[:, 0]  # label 0
        # final_logits[:, 2] = logit0245[:, 1]  # label 2
        # final_logits[:, 4] = logit0245[:, 2]  # label 4
        # final_logits[:, 5] = logit0245[:, 3]  # label 5

        # logits 路由合并 19
        # final_logits[:, 6] = logit6  # label 6
        # final_logits[:, 1] = logit13[:, 0] - logit6 - (logit0245[:, 0]+logit0245[:, 1]+logit0245[:, 2]+logit0245[:, 3])/4# label 1
        # final_logits[:, 3] = logit13[:, 1] - logit6 - (logit0245[:, 0]+logit0245[:, 1]+logit0245[:, 2]+logit0245[:, 3])/4# label 3
        # final_logits[:, 0] = logit0245[:, 0] - logit6 - (logit13[:, 0]+logit13[:, 1])/2  # label 0
        # final_logits[:, 2] = logit0245[:, 1] - logit6 - (logit13[:, 0]+logit13[:, 1])/2  # label 2
        # final_logits[:, 4] = logit0245[:, 2] - logit6 - (logit13[:, 0]+logit13[:, 1])/2  # label 4
        # final_logits[:, 5] = logit0245[:, 3] - logit6 - (logit13[:, 0]+logit13[:, 1])/2  # label 5
        if return_only_output:
            # 只走 forward，不算 loss
            return final_logits

        if criter == 'CE':  # 常规
            criterion0 = torch.nn.CrossEntropyLoss(reduction='none')
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')  # BCEWithLogitsLoss 用于多标签分类

        if label7 is not None:
            if 'PTBXL' in dataset:
                weights = torch.tensor(
                    # [1, 1, 1, 1, 1, 1, 1]).cuda()  # TB-net-new_14 38no dbw
                    [1, 1.5, 1, 1.25, 1, 1, 1]).cuda()  # TB-net-new_15 29+
                # [1, 2, 1, 1, 1, 1, 1]).cuda()  # TB-net-new_16
                # [1, 1.7, 1, 1.25, 1, 1, 1]).cuda()  # TB-net-new_17 19 20 21 22
                # [1, 2, 1, 1.25, 1, 1, 1]).cuda()  # TB-net-new_18
                # [1, 1.7, 1, 1, 1, 1, 1]).cuda()  # TB-net-new_
                Loss7 = torch.mean(weights * criterion(final_logits, label7.to(torch.float)))
                # Loss7 = torch.mean(criterion(MMlogit7, label7.to(torch.float)))
            else:
                Loss7 = torch.mean(criterion0(final_logits, label7))
        else:
            Loss7 = torch.tensor(0.0, device=device)  # 如果没有标签，则损失为0

        MMLoss = Loss7
        # return MMLoss, MMlogit7
        return MMLoss, final_logits

    def get_only_output(self, batch, num_classes=7, label7=None, dataset='PTBXL', criter='CE'):
        return self.forward(batch, num_classes=num_classes, label7=label7, dataset=dataset, criter=criter,
                            return_only_output=True)
