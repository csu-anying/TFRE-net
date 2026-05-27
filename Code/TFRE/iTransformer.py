import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from SEresnet import *

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

#
class Configs:
    def __init__(self):
        # 模型相关参数
        self.seq_len = 32 # 序列长度
        self.pred_len = 32  # 预测长度
        self.output_attention = False  # 是否输出注意力权重
        self.use_norm = True  # 是否使用归一化
        self.class_strategy = "cls_token"  # 类别策略

        # 嵌入层参数
        self.d_model = 512  # 隐藏层维度
        self.embed = 512  # 嵌入层维度 没用
        self.freq = None  # 频率信息（如果有的话） 没用
        self.dropout = 0.5  # Dropout 概率
        self.factor = 8  # MultiHeadAttention 中的缩放因子 没用
        self.n_heads = 64 # 注意力头数

        # 编码器和解码器层数
        self.e_layers = 16  # 编码器层数
        self.d_layers = 16  # 解码器层数

        # FeedForward 层参数
        self.d_ff = 1024  # FeedForward 层的隐藏层维度
        self.activation = 'gelu'  # 激活函数


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# class PDF3(nn.Module):
#     def __init__(self, in_dim, hidden_dim, num_class, dropout):  # in_dim是有12个5000的列表，hidden_dim是[16],num_class是7
#
#         super().__init__()
#         self.config = Configs()
#         self.dropout = dropout
#         self.views = 12
#
#         self.FeatureEncoder = nn.ModuleList(
#             [mSEnet() for view in range(self.views)])
#         self.itransformer = Model(self.config)
#         self.lstm = nn.LSTM(input_size=12, hidden_size=64, num_layers=64, batch_first=True)
#         self.avp = nn.AvgPool1d(4)
#         self.trans = nn.Linear(64, 12)
#
#         self.MMClasifier = []
#         self.MMClasifier.append(LinearLayer(32 * 12, 2))  # ��������1
#         self.MMClasifier = nn.Sequential(*self.MMClasifier)
#
#         self.MMClasifier5 = []
#         self.MMClasifier5.append(LinearLayer(32 * 12, 5))  # ��������2
#         self.MMClasifier5 = nn.Sequential(*self.MMClasifier5)
#
#         self.MMClasifier7 = []
#         self.MMClasifier7.append(LinearLayer(32 * 12, 7))  # ��������3
#         self.MMClasifier7 = nn.Sequential(*self.MMClasifier7)
#
#     def forward(self, data_list, label2=None, label5=None, label7=None, dataset='PTBXL'):
#         # �����
#         FeatureInfo, feature, feature_lead, TCPLogit, TCPConfidence, lead_weight, TCPpreLogit = dict(), dict(), dict(), dict(), dict(), dict(), dict()
#         MMfeature = torch.stack(data_list, dim=-1)  # [64,5000,12]
#
#         for view in range(self.views):
#             feature[view] = self.FeatureEncoder[view](data_list[view])
#             feature[view] = feature[view].flatten(start_dim=1, end_dim=2)
#         ALL_feature = torch.cat([i for i in feature.values()], dim=1)  # [64,16*12]
#         MMfeature = self.itransformer(MMfeature, None, None, None)  # [64,16,12]
#         LSTM_feature,_ = self.lstm(MMfeature)
#         # print(LSTM_feature.shape)
#         LSTM_feature = self.trans(LSTM_feature)
#         LSTM_feature = LSTM_feature.flatten(start_dim=1, end_dim=2)
#         MMfeature = MMfeature.flatten(start_dim=1, end_dim=2)  # [64,16*12]
#         MMfeature += ALL_feature + LSTM_feature
#
#
#         MMfeature = F.dropout(MMfeature, self.dropout, training=self.training)
#
#         MMlogit = self.MMClasifier(MMfeature)
#         MMlogit5 = self.MMClasifier5(MMfeature)
#         MMlogit7 = self.MMClasifier7(MMfeature)
#
#         criterion0 = torch.nn.CrossEntropyLoss(reduction='none')
#         criterion = torch.nn.BCEWithLogitsLoss()  # BCEloss���ڶ����ཻ���أ��ú���������BCE Loss�Լ�Sigmoid�����㣬
#
#         if 'PTBXL' in dataset:
#             Loss2 = torch.mean(criterion0(MMlogit, label2))
#             Loss5 = torch.mean(criterion(MMlogit5, label5.to(torch.float)))
#             Loss7 = torch.mean(criterion(MMlogit7, label7.to(torch.float)))
#         else:
#             Loss2 = torch.mean(criterion0(MMlogit, label2))
#             Loss5 = torch.mean(criterion0(MMlogit5, label5))
#             Loss7 = torch.mean(criterion0(MMlogit7, label7))
#
#         MMLoss = 0.013 * Loss2 + 0.19 * Loss5 + 0.79 * Loss7
#
#         return MMLoss, MMlogit7
