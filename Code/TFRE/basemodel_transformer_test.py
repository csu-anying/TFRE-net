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
import torch_geometric.nn as pyg_nn
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
        indices = torch.arange(seq_len, device=x.device).unsqueeze(1) - torch.arange(seq_len,
                                                                                     device=x.device).unsqueeze(0)
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
        self.position_embedding = RelativePositionalEncoding(embed_dim, max_len)  # 2
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
        fft_result = torch.fft.rfft(x, dim=-1)  # (batch, freq_len)
        fft_mag = torch.abs(fft_result)
        freq_feature = fft_mag[:, :self.topk]
        return self.scale * freq_feature  # 缩放后的频域特征


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


class GatedFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.1))  # 初始值小，避免过强干扰

    def forward(self, base_feat, cross_feat):
        return base_feat + self.gate * cross_feat


class FrequencySELayer(nn.Module):
    def __init__(self, freq_dim, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(freq_dim, freq_dim * reduction, bias=False),
            nn.ReLU(),
            nn.Linear(freq_dim * reduction, freq_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, freq_dim)
        weights = self.fc(x)
        return x * weights


class ViewAttention(nn.Module):
    def __init__(self, num_views, feature_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)  # 每个 view 一个权重
        )

    def forward(self, features):
        # features: (B, V, D)
        scores = self.attn(features)  # (B, V, 1)
        weights = torch.softmax(scores, dim=1)
        return features * weights


class ResidualCrossAttention(nn.Module):
    """ 残差 cross-attention 增强 """

    def __init__(self, alpha=0.5, learnable=True):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, base_feat, cross_feat):
        return base_feat + self.alpha * cross_feat


def contrastive_loss(features, temperature=0.5):
    """
    features: (B, V, D)
    """
    B, V, D = features.shape
    features = F.normalize(features, dim=-1)
    sim_matrix = torch.matmul(features, features.transpose(1, 2)) / temperature
    labels = torch.arange(V).repeat(B, 1).to(features.device)  # 每个 view 是正样本
    loss = F.cross_entropy(sim_matrix.view(-1, V), labels.view(-1))
    return loss


class ResidualTaskAlignment(nn.Module):
    """ 用 7分类 logit 残差校正 4分类 logit """

    def __init__(self, num_class7, num_class4, gamma=0.1, learnable=True):
        super().__init__()
        self.proj = nn.Linear(num_class7, num_class4, bias=False)
        if learnable:
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, logit7, logit4):
        # 残差修正：logit4 ← logit4 + γ * (Proj(logit7) - logit4)
        correction = self.proj(logit7) - logit4
        return logit4 + self.gamma * correction


class ResidualConfidenceCalibration(nn.Module):
    """ 用虚拟分支输出残差校正主任务 logit """

    def __init__(self, num_virtual, num_class, lambda_conf=0.05, learnable=True):
        super().__init__()
        self.proj = nn.Linear(num_virtual, num_class, bias=False)
        if learnable:
            self.lambda_conf = nn.Parameter(torch.tensor(lambda_conf))
        else:
            self.register_buffer("lambda_conf", torch.tensor(lambda_conf))

    def forward(self, virtual_output, logit):
        # 残差修正：logit ← logit + λ * (Proj(virtual) - logit)
        correction = self.proj(virtual_output) - logit
        return logit + self.lambda_conf * correction


class ResidualUncertaintyRefinement(nn.Module):
    """ 基于预测不确定性（熵）的残差校正 """

    def __init__(self, num_class, alpha=0.05, hidden_dim=16, learnable=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_class)
        )
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, logit):
        # 计算 softmax 熵
        prob = F.softmax(logit, dim=-1)
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1, keepdim=True)  # [B,1]
        correction = self.mlp(entropy)
        return logit + self.alpha * correction


# ============ 四个残差模块 ============

class ResidualDilatedConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=2):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(dim, dim, kernel_size,
                              padding=padding, dilation=dilation, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()

    def forward(self, x):  # (B, V, D)
        B, V, D = x.shape
        x_in = x.permute(0, 2, 1)  # (B, D, V)
        out = self.conv(x_in)
        out = self.bn(out)
        out = self.relu(out)
        out = out.permute(0, 2, 1)
        return x + out


class ResidualSelfGating(nn.Module):
    def __init__(self, dim, beta=0.5):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        gate = self.gate(x)
        return x + self.beta * (x * gate)


class ResidualGraphProjection(nn.Module):
    def __init__(self, num_views, dim, gamma=0.1):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.gamma = nn.Parameter(torch.tensor(gamma))
        adj = torch.ones(num_views, num_views) - torch.eye(num_views)
        self.register_buffer("adj", adj / adj.sum(1, keepdim=True))

    def forward(self, x):  # (B, V, D)
        graph_feat = torch.matmul(self.adj, x)  # (B, V, D)
        proj_feat = self.proj(graph_feat)
        return x + self.gamma * proj_feat


class ResidualChannelAttention(nn.Module):
    def __init__(self, num_views, dim, beta=0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):  # (B, V, D)
        scores = self.attn(x)
        weights = torch.sigmoid(scores)
        return x + self.beta * (x * weights)


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


class PDF3(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout,
                 use_gate=False, use_freq_se=True,
                 use_virtual_loss_weight=False, lambda_virtual=0.03,
                 use_view_attn=False, use_residual_ca=True,
                 use_contrastive=False, contrastive_weight=0.1,
                 use_residual_task_align=True, gamma_task_align=0.1,  # 0.1
                 use_residual_conf_calib=True, lambda_conf=0.05,  # 0.05
                 use_residual_uncertainty=True, alpha_uncertainty=0.05,  # 0.05
                 # 🔹 新增四个残差方法
                 use_residual_dconv=False,
                 use_residual_sg=True,
                 use_residual_gp=True,
                 use_residual_channel_attn=False
                 ):
        # need {use_freq_se,use_residual_ca,use_residual_task_align,use_residual_conf_calib,use_residual_uncertainty,
        # use_residual_sg,use_residual_gp}
        # contrastive bad view_attn :bad freq_se residual_ca 3R :good
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = nn.Dropout(p=dropout)
        self.embding = hidden_dim[-1]
        self.FeatureEncoder = nn.ModuleList([mSEnet() for _ in range(self.views)])

        # 频域增强
        self.freq_layer = FrequencyDomainLayer(topk=50, init_scale=0.1)  # topk=50, init_scale=0.1
        self.use_freq_se = use_freq_se
        if use_freq_se:
            self.freq_se = FrequencySELayer(freq_dim=50)  # 50

        # 时域+频域融合
        self.fusion = WeightedFusionTF(time_dim=hidden_dim[-1], freq_dim=50)  # 50
        self.feature_dim = hidden_dim[-1]

        # old
        # sage
        self.learnable_adj = nn.Parameter(torch.randn(12, 12))
        self.conv1 = pyg_nn.SAGEConv(32, 128)
        self.conv2 = pyg_nn.SAGEConv(128, 512)
        self.conv3 = pyg_nn.SAGEConv(512, 32)
        self.bn1 = pyg_nn.BatchNorm(12)
        self.bn2 = pyg_nn.BatchNorm(12)
        self.bn3 = pyg_nn.BatchNorm(12)
        self.act = nn.LeakyReLU(0.3)

        # 分类器
        self.MMClasifier4 = nn.Sequential(nn.Linear(self.views * self.feature_dim, 4))
        self.MMClasifier7 = nn.Sequential(nn.Linear(self.views * self.feature_dim, 7))
        self.virtual_output = nn.Linear(self.views * self.feature_dim, 1)

        # cross-attention
        self.cross_attention_with_guidance = CrossAttentionWithGuidance(embed_dim=self.feature_dim, num_heads=2)
        self.use_gate = use_gate
        if use_gate:
            self.gated_fusion1 = GatedFusion(self.feature_dim)
            self.gated_fusion2 = GatedFusion(self.feature_dim)
            self.gated_fusion3 = GatedFusion(self.feature_dim)

        # 🔹 新增：view attention
        self.use_view_attn = use_view_attn
        if use_view_attn:
            self.view_attn = ViewAttention(self.views, self.feature_dim)

        # 🔹 新增：残差 cross-attention
        self.use_residual_ca = use_residual_ca
        if use_residual_ca:
            self.residual_ca1 = ResidualCrossAttention(alpha=0.5, learnable=True)
            self.residual_ca2 = ResidualCrossAttention(alpha=0.5, learnable=True)
            self.residual_ca3 = ResidualCrossAttention(alpha=0.5, learnable=True)

        # 🔹 新增：残差任务对齐
        self.use_residual_task_align = use_residual_task_align
        if use_residual_task_align:
            self.task_aligner = ResidualTaskAlignment(num_class7=7, num_class4=4, gamma=gamma_task_align)
        # 🔹 残差置信度校正
        self.use_residual_conf_calib = use_residual_conf_calib
        if use_residual_conf_calib:
            self.conf_calib4 = ResidualConfidenceCalibration(num_virtual=1, num_class=4, lambda_conf=lambda_conf)
            self.conf_calib7 = ResidualConfidenceCalibration(num_virtual=1, num_class=7, lambda_conf=lambda_conf)
        # 🔹 残差不确定性修正
        self.use_residual_uncertainty = use_residual_uncertainty
        if use_residual_uncertainty:
            self.uncert_refine4 = ResidualUncertaintyRefinement(num_class=4, alpha=alpha_uncertainty)
            self.uncert_refine7 = ResidualUncertaintyRefinement(num_class=7, alpha=alpha_uncertainty)

        # 🔹 四个残差开关
        self.use_residual_dconv = use_residual_dconv
        self.use_residual_sg = use_residual_sg
        self.use_residual_gp = use_residual_gp
        self.use_residual_channel_attn = use_residual_channel_attn

        if self.use_residual_dconv:
            self.res_dconv = ResidualDilatedConvBlock(self.feature_dim)
        if self.use_residual_sg:
            self.res_sg = ResidualSelfGating(self.feature_dim)
        if self.use_residual_gp:
            self.res_gp = ResidualGraphProjection(self.views, self.feature_dim)
        if self.use_residual_channel_attn:
            self.res_channel_attn = ResidualChannelAttention(self.views, self.feature_dim)

        # 🔹 新增：对比损失
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight

        # loss wrapper
        self.loss_wrapper = DWALoss(tasks=3)
        self.use_virtual_loss_weight = use_virtual_loss_weight
        self.lambda_virtual = lambda_virtual

    def forward(self, data_list, label4=None, label5=None, label7=None, dataset='PTBXL',return_only_output=False):
        FeatureInfo, feature, feature_lead, TCPLogit, TCPConfidence, lead_weight, TCPpreLogit = dict(), dict(), dict(), dict(), dict(), dict(), dict()

        # Step1: 提取特征
        for view in range(self.views):
            time_feat = self.FeatureEncoder[view](data_list[view])
            time_feat = time_feat.flatten(start_dim=1, end_dim=2)
            freq_feat = self.freq_layer(data_list[view].squeeze(1))
            if self.use_freq_se:
                freq_feat = self.freq_se(freq_feat)
            feature[view] = self.fusion(time_feat, freq_feat)

        MMfeature = torch.stack([i for i in feature.values()], dim=1)
        stacked_data = MMfeature  # [64, 12, 32]
        device = next(self.parameters()).device
        stacked_data = stacked_data.to(device)
        # old
        soft_adj = F.softmax(self.learnable_adj, dim=-1)
        gat_edge = soft_adj.nonzero(as_tuple=False).t().contiguous()
        gat = self.act(self.bn1(self.conv1(stacked_data, gat_edge)))

        gat = self.act(self.bn2(self.conv2(gat, gat_edge)))
        gat = self.act(self.bn3(self.conv3(gat, gat_edge)))
        gat = gat.view(-1, 12, 32)
        MMfeature = gat + stacked_data  # gcn bad tranc bad

        # 🔹 导联注意力
        if self.use_view_attn:
            MMfeature = self.view_attn(MMfeature)

        # Step2: cross-attention 融合
        indices_to_cross = [6, 7, 10, 11, 3, 0]
        cross_encoded_features = self.cross_attention_with_guidance(MMfeature.clone(), indices_to_cross,
                                                                    guidance_input=MMfeature)

        if self.use_gate:
            MMfeature[:, 6:8, :] = self.gated_fusion1(MMfeature[:, 6:8, :], cross_encoded_features[:, 0:2, :])
            MMfeature[:, 10:12, :] = self.gated_fusion2(MMfeature[:, 10:12, :], cross_encoded_features[:, 2:4, :])
            MMfeature[:, [3, 0], :] = self.gated_fusion3(MMfeature[:, [3, 0], :], cross_encoded_features[:, 4:6, :])
        elif self.use_residual_ca:  # 🔹 残差 cross-attention
            MMfeature[:, 6:8, :] = self.residual_ca1(MMfeature[:, 6:8, :], cross_encoded_features[:, 0:2, :])
            MMfeature[:, 10:12, :] = self.residual_ca2(MMfeature[:, 10:12, :], cross_encoded_features[:, 2:4, :])
            MMfeature[:, [3, 0], :] = self.residual_ca3(MMfeature[:, [3, 0], :], cross_encoded_features[:, 4:6, :])
        else:  # 原始方式
            MMfeature[:, 6:8, :] += cross_encoded_features[:, 0:2, :]
            MMfeature[:, 10:12, :] += cross_encoded_features[:, 2:4, :]
            MMfeature[:, [3, 0], :] += cross_encoded_features[:, 4:6, :]
        # # 1+a消融
        # self.a = nn.Parameter(torch.tensor(0.0))
        # self.b = nn.Parameter(torch.tensor(0.0))
        # self.c = nn.Parameter(torch.tensor(0.0))
        # MMfeature[:, 6:8, :] += self.a * MMfeature[:, 6:8, :].clone()  # 避免原地操作冲突
        # MMfeature[:, 10:12, :] += self.b * MMfeature[:, 10:12, :].clone()
        # MMfeature[:, [3, 0], :] += self.c * MMfeature[:, [3, 0], :]  # 确保c正确广播

        # 🔹 应用四个残差模块
        if self.use_residual_dconv:
            MMfeature = self.res_dconv(MMfeature)
        if self.use_residual_sg:
            MMfeature = self.res_sg(MMfeature)
        if self.use_residual_gp:
            MMfeature = self.res_gp(MMfeature)
        if self.use_residual_channel_attn:
            MMfeature = self.res_channel_attn(MMfeature)

        MMfeature = MMfeature.view(-1, 12 * self.feature_dim).clone()
        if self.training:  # 只在训练阶段启用 R-drop
            MMfeature = self.dropout(MMfeature)

        # Step3: 分类
        MMlogit4 = self.MMClasifier4(MMfeature)
        MMlogit7 = self.MMClasifier7(MMfeature)
        if return_only_output:
            # 只走 forward，不算 loss
            return MMlogit7
        virtual_logits = self.virtual_output(MMfeature)
        # 🔹 残差任务对齐
        if self.use_residual_task_align:
            MMlogit4 = self.task_aligner(MMlogit7, MMlogit4)
        # 🔹 残差置信度校正
        if self.use_residual_conf_calib:
            MMlogit4 = self.conf_calib4(virtual_logits, MMlogit4)
            MMlogit7 = self.conf_calib7(virtual_logits, MMlogit7)
        # 残差不确定性修正
        if self.use_residual_uncertainty:
            MMlogit4 = self.uncert_refine4(MMlogit4)
            MMlogit7 = self.uncert_refine7(MMlogit7)
        # Loss
        criterion0 = torch.nn.CrossEntropyLoss(reduction='none')
        criterion = torch.nn.BCEWithLogitsLoss()

        if 'PTBXL' in dataset:
            if label7 is None:
                # 推理模式
                return 1, MMlogit7
            weights7 = torch.tensor([2, 1, 2, 1, 2, 2, 1]).cuda()
            Loss7 = torch.mean(weights7 * criterion(MMlogit7, label7.to(torch.float)))

            weights4 = torch.tensor([1, 1, 1, 1]).cuda()
            Loss4 = torch.mean(weights4 * criterion(MMlogit4, label4.to(torch.float)))
        else:
            # ⚠️ 保证标签是 int64 (long) 且是一维索引
            Loss7 = torch.mean(criterion0(MMlogit7, label7.long().view(-1)))
            Loss4 = torch.mean(criterion0(MMlogit4, label4.long().view(-1)))

        virtual_loss = torch.mean(criterion(virtual_logits, torch.ones_like(virtual_logits)))
        if self.use_virtual_loss_weight:
            virtual_loss = self.lambda_virtual * virtual_loss

        MMLoss = self.loss_wrapper([Loss7, Loss4, virtual_loss])

        # 🔹 加 contrastive loss
        if self.use_contrastive:
            cl_loss = contrastive_loss(MMfeature.view(-1, self.views, self.feature_dim))
            MMLoss = MMLoss + self.contrastive_weight * cl_loss

        return MMLoss, MMlogit7
