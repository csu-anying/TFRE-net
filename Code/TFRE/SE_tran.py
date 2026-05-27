import torch
import torch.nn as nn
# from torchsummary import summary
import numpy as np
import sys

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


import math
class DynamicConvolutionalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, kernel_size=3):
        super(DynamicConvolutionalPositionalEncoding, self).__init__()
        self.kernel_size = kernel_size
        self.conv_weight_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * kernel_size),
            nn.ReLU(),
            nn.Linear(embed_dim * kernel_size, embed_dim * kernel_size)
        )
        self.conv_bias = nn.Parameter(torch.zeros(embed_dim))  # 动态卷积的偏置

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_windows, embed_dim)
        Returns:
            x + dynamic_conv_pe: 添加了动态卷积位置编码的张量
        """
        batch_size, seq_len, embed_dim = x.size()

        # 转换为 (batch_size, embed_dim, seq_len)
        x = x.permute(0, 2, 1)

        # 动态生成卷积核
        conv_weights = self.conv_weight_generator(x.mean(dim=2))  # (batch_size, embed_dim * kernel_size)
        conv_weights = conv_weights.view(batch_size, embed_dim, self.kernel_size)  # (batch_size, embed_dim, kernel_size)

        # 确保卷积核的形状符合要求 (embed_dim, 1, kernel_size)
        # conv_weights = conv_weights.permute(1, 0, 2)  # (embed_dim, batch_size, kernel_size)

        # 应用动态卷积
        dynamic_conv_pe = torch.zeros_like(x)
        for b in range(batch_size):
            dynamic_conv_pe[b] = nn.functional.conv1d(
                x[b:b+1], weight=conv_weights[b:b+1],  padding=(self.kernel_size - 1) // 2
            )

        # 转回原始形状 (batch_size, seq_len, embed_dim)
        dynamic_conv_pe = dynamic_conv_pe.permute(0, 2, 1)

        return x.permute(0, 2, 1) + dynamic_conv_pe


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



class RelativePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=600):
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


class TransformerEncoderWithPosition(nn.Module):
    def __init__(self,window_size, embed_dim, num_heads, ff_dim, max_len=5000):
        super(TransformerEncoderWithPosition, self).__init__()
        self.embedding = nn.Linear(window_size, embed_dim)
        self.positional_encoding4 = DynamicConvolutionalPositionalEncoding(embed_dim) #  4 4/2 4/3
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim),
            num_layers=2 #  2
        )
        # self.output_layer = nn.Linear(embed_dim, 32)
        self.gap = nn.AdaptiveMaxPool1d(1)
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_windows, window_size)
        Returns:
            Encoded output with shape (batch_size, num_windows, 32)
        """
        x = self.embedding(x)  # (batch_size, num_windows, embed_dim)
        x = self.positional_encoding4(x)  # 添加位置编码
        x = x.permute(1, 0, 2)  # Transformer 需要输入为 (seq_len, batch_size, embed_dim)
        x = self.transformer(x)  # (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # 恢复为 (batch_size, num_windows, embed_dim)
        # x = self.output_layer(x)  # (batch_size, num_windows, 32)
        x = x.permute(0, 2, 1)  # 恢复为 (batch_size, embed_dim, num_windows)
        x = self.gap(x)  # 恢复为 (batch_size, embed_dim, 1)
        return x




class mSEnet(nn.Module):
    """docstring for mSEnet"""

    def __init__(self):
        super(mSEnet, self).__init__()
        self.window_size = 400  # 16 60 80 200 300 400
        self.stride = 40 # 6 40 up and good for memory
        self.embed_dim = 48  # 16 12 32 up and good
        self.num_heads = 24  # 4 6 16 up and good
        self.ff_dim = 16  # 128 32  3  8 12 low and good 16best
        self.tr = TransformerEncoderWithPosition(window_size = self.window_size,embed_dim=self.embed_dim, num_heads=self.num_heads, ff_dim=self.ff_dim)


    def forward(self, x):
        x = torch.unsqueeze(x, dim=1) #64 1 5000
        # 滑动窗口分割
        x = x.unfold(dimension=-1, size=self.window_size, step=self.stride).squeeze(1)  # (64, num_windows, window_size)

        # Transformer 参数
        x = self.tr(x)


        return x
