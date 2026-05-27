""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
# from SEnet import *
# from SE_tran import *
# from MLResnet import *
from SEresnet import *
# from CA import *
# from IM import *
import sys
# from unireplknet import UniRepLKNetBlock

import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 用于调整类别不平衡
        self.gamma = gamma  # 焦点参数，用来控制难易样本的权重
        self.reduction = reduction  # 损失的聚合方式，'mean' or 'sum'

    def forward(self, logits, target):
        # 使用 sigmoid 激活 logits 得到预测概率
        prob = torch.sigmoid(logits)

        # 计算交叉熵损失
        cross_entropy_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

        # 计算 Focal Loss
        p_t = prob * target + (1 - prob) * (1 - target)  # 目标类别的预测概率
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * cross_entropy_loss

        if self.reduction == 'mean':
            return focal_loss.mean()  # 对损失取平均
        elif self.reduction == 'sum':
            return focal_loss.sum()  # 对损失求和
        else:
            return focal_loss  # 返回每个样本的损失
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


class ConvolutionalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, kernel_size=3):
        super(ConvolutionalPositionalEncoding, self).__init__()
        # 动态 padding，确保输出序列长度与输入一致
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,  # 自动计算 padding
            groups=embed_dim
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_windows, embed_dim)
        Returns:
            x + conv_pe: 添加了卷积位置编码的张量
        """
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.conv.in_channels, "Embedding dimension mismatch"

        # 转换为 (batch_size, embed_dim, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)

        # 卷积操作
        conv_pe = self.conv(x)  # (batch_size, embed_dim, seq_len)

        # 转回原始形状 (batch_size, seq_len, embed_dim)
        conv_pe = conv_pe.permute(0, 2, 1)

        # 保证输出与输入形状一致
        return x.permute(0, 2, 1) + conv_pe

class GlobalConvolutionalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, kernel_size=3):
        super(GlobalConvolutionalPositionalEncoding, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,  # padding 保持 seq_len 不变
            groups=embed_dim
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            x + conv_pe + global_feature: 添加了卷积和全局特征的张量
        """
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.conv.in_channels, "Embedding dimension mismatch"

        # 转换为 (batch_size, embed_dim, seq_len)
        x = x.permute(0, 2, 1)

        # 卷积操作 (保持 seq_len 不变)
        conv_pe = self.conv(x)  # (batch_size, embed_dim, seq_len)

        # 全局特征池化
        global_feature = self.global_pool(x)  # (batch_size, embed_dim, 1)
        global_feature = global_feature.expand(-1, -1, seq_len)  # 扩展为 (batch_size, embed_dim, seq_len)

        # 转回原始形状
        conv_pe = conv_pe.permute(0, 2, 1)  # (batch_size, seq_len, embed_dim)

        # 添加全局特征到卷积位置编码
        return x.permute(0, 2, 1) + conv_pe + global_feature.permute(0, 2, 1)


class MultiScaleConvolutionalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, kernel_sizes=[3, 7, 15]):
        super(MultiScaleConvolutionalPositionalEncoding, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=k,
                padding=(k - 1) // 2,  # 确保输出与输入长度一致
                groups=embed_dim
            )
            for k in kernel_sizes
        ])

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_windows, embed_dim)
        Returns:
            x + multi_scale_conv_pe: 添加了多尺度卷积位置编码的张量
        """
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.convs[0].in_channels, "Embedding dimension mismatch"

        # 转换为 (batch_size, embed_dim, seq_len)
        x = x.permute(0, 2, 1)

        # 多尺度卷积
        conv_results = [conv(x) for conv in self.convs]  # 每个卷积的结果
        multi_scale_conv_pe = sum(conv_results)  # 聚合多尺度结果

        # 转回原始形状 (batch_size, seq_len, embed_dim)
        multi_scale_conv_pe = multi_scale_conv_pe.permute(0, 2, 1)

        # 添加位置编码
        return x.permute(0, 2, 1) + multi_scale_conv_pe

class RelativePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(RelativePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # 定义相对位置嵌入矩阵
        self.relative_positions = nn.Embedding(2 * max_len - 1, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_windows, embed_dim)
        Returns:
            x + relative_pe: 添加了相对位置编码的张量
        """
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"

        # 计算相对位置索引
        indices = torch.arange(seq_len, device=x.device).unsqueeze(1) - torch.arange(seq_len, device=x.device).unsqueeze(0)
        indices = indices + self.max_len - 1  # 偏移到正索引范围
        relative_pe = self.relative_positions(indices)  # (seq_len, seq_len, embed_dim)

        # 对相对位置编码求平均以适配输入形状
        return x + relative_pe.mean(dim=1)

class AttentionGuidance(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionGuidance, self).__init__()
        # 一个简单的网络来生成导联的重要性评分
        self.attention_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),  # 将导联特征映射到一个潜在空间
            nn.ReLU(),
            nn.Linear(embed_dim, 1)  # 输出一个标量，表示导联的重要性评分
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征，形状为 (batch_size, num_windows, embed_dim)
        Returns:
            导联的权重，形状为 (batch_size, num_windows, 1)
        """
        importance_scores = self.attention_network(x)  # 计算每个导联的权重
        return importance_scores


class CrossAttentionWithRelativePosition(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len=100):
        super(CrossAttentionWithRelativePosition, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.position_embedding = RelativePositionalEncoding(embed_dim, max_len) #  2
        # self.position_embedding = GlobalConvolutionalPositionalEncoding(embed_dim) #  1
        # self.position_embedding = MultiScaleConvolutionalPositionalEncoding(embed_dim) #  3
        # self.position_embedding = ConvolutionalPositionalEncoding(embed_dim) #  1
    def forward(self, x, indices):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            indices: List of index pairs to cross-encode
        Returns:
            cross_attention_features: Tensor after cross-attention encoding with relative position embeddings
        """
        cross_attention_features = []
        for i in range(0, len(indices), 2):  # 每次处理一对交叉的维度
            idx1, idx2 = indices[i], indices[i + 1]
            # 选择对应维度的特征
            feature1 = x[:, idx1, :]  # shape (batch_size, embed_dim)
            feature2 = x[:, idx2, :]  # shape (batch_size, embed_dim)

            # 将这两个特征堆叠起来，供自注意力模块编码
            pair = torch.stack([feature1, feature2], dim=1)  # shape (batch_size, 2, embed_dim)

            # 生成相对位置编码
            relative_position_embeddings = self.position_embedding(pair)
            # 输入到注意力层，并加上相对位置编码
            attn_output, _ = self.attention(pair + relative_position_embeddings, pair + relative_position_embeddings,
                                            pair + relative_position_embeddings)
            cross_attention_features.append(attn_output)

        # 拼接所有交叉编码后的特征
        cross_attention_features = torch.cat(cross_attention_features,
                                             dim=1)  # shape (batch_size, num_pairs * embed_dim, seq_len)
        return cross_attention_features


# ==============================
# 损失自适应加权 (Uncertainty Weighting)
# ==============================
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_losses=3, debug=False, min_var=0.1):
        super().__init__()
        self.vars = nn.Parameter(torch.ones(num_losses) * 2.0)  # 初始化偏大
        self.debug = debug
        self.min_var = min_var

    def forward(self, losses):
        weighted_losses = 0
        debug_info = []
        for i, loss in enumerate(losses):
            # σ² 下界控制，避免过小
            var = F.softplus(self.vars[i]) + self.min_var
            precision = 1.0 / (2.0 * var)
            task_loss = precision * loss + 0.5 * torch.log(var)

            # ✅ 保证非负
            task_loss = torch.clamp(task_loss, min=0.0)

            weighted_losses += task_loss

            if self.debug:
                debug_info.append((loss.item(), var.item(), precision.item(), task_loss.item()))

        if self.debug:
            print(" [MultiTaskLoss Debug] ",
                  " | ".join([f"Task{i}: loss={l:.4f}, var={v:.4f}, prec={p:.4f}, contrib={c:.4f}"
                              for i, (l, v, p, c) in enumerate(debug_info)]))
        return weighted_losses





# ==============================
# 频域增强 (FFT 特征)
# ==============================
class FrequencyDomainLayer(nn.Module):
    def __init__(self, topk=50, init_scale=0.1):
        super(FrequencyDomainLayer, self).__init__()
        self.topk = topk
        # 可学习缩放参数，初始值很小
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        fft_result = torch.fft.rfft(x, dim=-1)   # (batch, freq_len)
        fft_mag = torch.abs(fft_result)
        freq_feature = fft_mag[:, :self.topk]
        return self.scale * freq_feature   # 缩放后的频域特征

# ==============================
# 时域+频域加权融合
# ==============================
class WeightedFusionTF(nn.Module):
    def __init__(self, time_dim, freq_dim):
        super(WeightedFusionTF, self).__init__()
        # 线性层保证维度一致
        self.proj_freq = nn.Linear(freq_dim, time_dim)
        # 可学习权重，初始值很小
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, time_feat, freq_feat):
        freq_feat = self.proj_freq(freq_feat)  # 频域映射到和时域相同的维度
        return time_feat + self.alpha * freq_feat
# ==============================
# 拒识机制 (Reject Option)
# ==============================
def reject_option(logits, threshold=0.6):
    """
    logits: (batch, num_classes)
    threshold: 拒识阈值
    return: (batch,) 预测类别，拒识为 -1
    """
    probs = torch.softmax(logits, dim=-1)
    max_probs, preds = torch.max(probs, dim=-1)
    preds[max_probs < threshold] = -1  # 低置信度样本标记为拒识
    return preds

class CrossAttentionWithGuidance(CrossAttentionWithRelativePosition):
    def __init__(self, embed_dim, num_heads, max_len=100):
        super(CrossAttentionWithGuidance, self).__init__(embed_dim, num_heads, max_len)
        self.attention_guidance = AttentionGuidance(embed_dim)  # 引入注意力引导模块

    def forward(self, x, indices, guidance_input=None):
        """
        Args:
            x: 输入特征，形状为 (batch_size, seq_len, embed_dim)
            indices: 跨维度编码的索引对
            guidance_input: 生成的引导信号，形状为 (batch_size, seq_len, embed_dim)
        Returns:
            加权后的交叉注意力特征
        """
        # 原始的交叉注意力计算
        cross_attention_features = super().forward(x, indices)

        # 生成导联重要性评分
        importance_scores = self.attention_guidance(cross_attention_features)  # 形状为 (batch_size, seq_len, 1)

        # 使用导联重要性评分来加权注意力
        # 对交叉注意力特征进行加权
        weighted_cross_attention_features = cross_attention_features * importance_scores

        return weighted_cross_attention_features

class WeightedFusion(nn.Module):
    def __init__(self):
        super(WeightedFusion, self).__init__()
        self.weight = nn.Parameter(torch.ones(1))  # 学习加权系数

    def forward(self, x, y):
        return x + self.weight * y  # 加权融合



class DWALoss(nn.Module):
    def __init__(self, tasks, T=2.0):
        super().__init__()
        self.T = T
        self.tasks = tasks
        self.loss_history = []

    def forward(self, losses):
        if len(self.loss_history) < 2:
            weights = torch.ones(self.tasks) / self.tasks
        else:
            w = []
            for i in range(self.tasks):
                r = self.loss_history[-1][i] / (self.loss_history[-2][i] + 1e-8)
                w.append(r)
            w = torch.tensor(w)
            weights = torch.softmax(w / self.T, dim=0)

        self.loss_history.append([l.item() for l in losses])
        return torch.sum(weights.to(losses[0].device) * torch.stack(losses))



class PDF3(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):  # in_dim是有12个5000的列表，hidden_dim是[16],num_class是7
        super().__init__()
        self.views = len(in_dim)  # 12��12������
        self.classes = num_class  # ��������Ĭ��Ϊ7
        # self.dropout = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.embding = hidden_dim[-1]

        self.FeatureEncoder = nn.ModuleList(
            [mSEnet() for view in range(self.views)])  # ������������������views��mSE���磨ÿ������ƥ��һ��������
        # 频域增强层
        self.freq_layer = FrequencyDomainLayer(topk=50, init_scale=0.1)

        # 时域+频域融合
        self.fusion = WeightedFusionTF(time_dim=hidden_dim[-1], freq_dim=50)
        # 分类器
        self.feature_dim = hidden_dim[-1]  # 因为融合后维度回到 hidden_dim[-1]
        self.MMClasifier4 = nn.Sequential(nn.Linear(self.views * self.feature_dim, 4))
        self.MMClasifier7 = nn.Sequential(nn.Linear(self.views * self.feature_dim, 7))

        # 引入虚拟类别预测层
        self.virtual_output = nn.Linear(self.views * self.feature_dim, 1)  # 用于预测虚拟类别（0或1）

        self.cross_attention = CrossAttentionWithRelativePosition(embed_dim=self.feature_dim, num_heads=2)
        self.cross_attention_with_guidance = CrossAttentionWithGuidance(embed_dim=self.feature_dim, num_heads=2)
        self.fusion1 = WeightedFusion()
        self.fusion2 = WeightedFusion()
        self.fusion3 = WeightedFusion()

        self.loss_wrapper = DWALoss(tasks=3)#2 good
        # 损失自适应加权
        # self.loss_wrapper = MultiTaskLossWrapper(num_losses=3, debug=True)

        # self.virtual_output = nn.Linear(self.views * hidden_dim[-1], 1)  # 用于预测虚拟类别概率
    def forward(self, data_list, label4=None, label5=None, label7=None, dataset='PTBXL'):
        # �����
        FeatureInfo, feature, feature_lead, TCPLogit, TCPConfidence, lead_weight, TCPpreLogit = dict(), dict(), dict(), dict(), dict(), dict(), dict()

        # Step1: 特征提取（时域 + 频域）
        for view in range(self.views):
            time_feat = self.FeatureEncoder[view](data_list[view])  # (B, C, 1)
            time_feat = time_feat.flatten(start_dim=1, end_dim=2)  # (B, hidden_dim)
            freq_feat = self.freq_layer(data_list[view].squeeze(1))  # (B, 50)
            # ✅ 用加权融合替代拼接
            feature[view] = self.fusion(time_feat, freq_feat)
        MIfeature, MIfeature_1 = dict(), dict()
        # MMfeature = torch.cat([i for i in feature.values()], dim=1)

        # Step2: 堆叠导联特征
        MMfeature = torch.stack([i for i in feature.values()], dim=1)  # (B, 12, feature_dim)

        # Perform cross-attention on specific positions: (6,7), (10,11), (3,0)
        indices_to_cross = [6, 7, 10, 11, 3, 0]  # Define the index pairs for cross-encoding
        # cross_encoded_features = self.cross_attention(MMfeature.clone(), indices_to_cross)
        # embed_code = self.cross_attention_with_guidance(MMfeature.clone(), list(range(12)),
        #                                                      guidance_input=MMfeature)
        cross_encoded_features = self.cross_attention_with_guidance(MMfeature.clone(), indices_to_cross,
                                                        guidance_input=MMfeature)
        # Now concatenate the cross-encoded features with the original features

        # MMfeature[:, 6:8, :] =self.fusion1(MMfeature[:, 6:8, :],cross_encoded_features[:, 0:2, :])
        # MMfeature[:, 10:12, :] =self.fusion2(MMfeature[:, 10:12, :],cross_encoded_features[:, 2:4, :])
        # MMfeature[:, [3,0], :] =self.fusion2(MMfeature[:, [3,0], :],cross_encoded_features[:, 4:6, :])
        MMfeature[:, 6:8, :] = MMfeature[:, 6:8, :]+cross_encoded_features[:, 0:2, :]
        MMfeature[:, 10:12, :] = MMfeature[:, 10:12, :]+ cross_encoded_features[:, 2:4, :]
        MMfeature[:, [3, 0], :] = MMfeature[:, [3, 0], :] +cross_encoded_features[:, 4:6, :]
        MMfeature = MMfeature.view(-1,12*self.feature_dim).clone()
        # MMfeature = F.dropout(MMfeature, self.dropout, training=self.training)
        if self.training:  # 只在训练阶段启用 R-drop
            MMfeature = self.dropout(MMfeature)  # 添加 dropout
        MMlogit4 = self.MMClasifier4(MMfeature)
        MMlogit7 = self.MMClasifier7(MMfeature)

        # 计算虚拟类别预测
        virtual_logits = self.virtual_output(MMfeature)  # 用于虚拟类别的预测

        # 计算 loss
        criterion0 = torch.nn.CrossEntropyLoss(reduction='none')
        criterion = torch.nn.BCEWithLogitsLoss()

        if 'PTBXL' in dataset:
            weights7 = torch.tensor([1, 1, 1, 1, 1, 1, 1]).cuda()
            Loss7 = torch.mean(weights7 * criterion(MMlogit7, label7.to(torch.float)))

            weights4 = torch.tensor([1, 1, 1, 1]).cuda()
            Loss4 = torch.mean(weights4 * criterion(MMlogit4, label4.to(torch.float)))
        else:
            Loss7 = torch.mean(criterion0(MMlogit7, label7))
            Loss4 = torch.mean(criterion0(MMlogit4, label4))

        virtual_loss = torch.mean(criterion(virtual_logits, torch.ones_like(virtual_logits)))

        MMLoss = self.loss_wrapper([Loss7, Loss4, virtual_loss])#2,4

        return MMLoss, MMlogit7
