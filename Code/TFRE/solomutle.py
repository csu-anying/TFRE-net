import math

import torch
import torch.nn as nn
# from torchsummary import summary
import numpy as np
import sys

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

if torch.cuda.is_available():
   device = torch.device("cuda")
else:
   device = torch.device("cpu")

class SE_Module(nn.Module):

    def __init__(self, in_channels, ratio=4, dim=1):
        super(SE_Module, self).__init__()
        self.dim = dim
        if self.dim == 1:
            self.squeeze = nn.AdaptiveAvgPool1d(1)
        else:
            self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_channels // ratio, out_features=in_channels),
            nn.Sigmoid()
        )


    def forward(self, x):
        identity = x

        out = self.squeeze(x)
        out = out.reshape(out.shape[0], out.shape[1])
        scale = self.excitation(out)
        if self.dim == 1:
            scale = scale.reshape(scale.shape[0], scale.shape[1], 1)
        else:
            scale = scale.reshape(scale.shape[0], scale.shape[1], 1, 1)

        return identity * scale.expand_as(identity)

class ResBlock2d(nn.Module):
    """docstring for ResBlock2d"""
    def __init__(self, filters):
        super(ResBlock2d, self).__init__()

        self.conv2d0 = nn.Sequential(
            nn.Conv2d(1, filters // 2, kernel_size = (21, 1), padding = ((21 - 1) // 2, 0) ),
            nn.BatchNorm2d(filters // 2),
            nn.LeakyReLU(0.3)
        )

        self.conv2d1 = nn.Sequential(
            nn.Conv2d(filters // 2, filters, kernel_size = (17, 1), padding = ((17 - 1) // 2, 0) ),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.3)
        )

        self.conv2d2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size = (11, 1), padding = ((11 - 1) // 2, 0) ),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.3)
        )

        self.conv2d3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size = (5, 1), padding = ((5 - 1) // 2, 0) ),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.3)
        )

        self.SE = SE_Module(filters, dim=2)

        self.conv2d4 = nn.Sequential(
            nn.Conv2d(filters // 2, filters, kernel_size = (1, 1), padding = (0, 0) ),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.3)
        )

        self.leakyrule = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.conv2d0(x)
        output = x
        output = self.conv2d1(output)
        output = self.conv2d2(output)
        output = self.conv2d3(output)
        output = self.SE(output)

        x = self.conv2d4(x)

        output = output + x

        output = self.leakyrule(output)

        return output

class ResBlock1d(nn.Module):
    """docstring for ResBlock2d"""
    def __init__(self, filters):
        super(ResBlock1d, self).__init__()
        filters0 = filters
        if filters != 4:
            filters0 = filters // 2

        self.conv1d1 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size = 17, padding = (17 - 1) // 2),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        self.conv1d2 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_size = 11, padding = (11 - 1) // 2),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        self.conv1d3 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_size = 5, padding = (5 - 1) // 2),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        self.SE = SE_Module(filters, dim=1)

        self.conv1d4 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size = 1, padding = 0),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        self.leakyrule = nn.LeakyReLU(0.3)
        self.avp = nn.AvgPool1d(3)

    def forward(self, x):
        output = x
        output = self.conv1d1(output)
        output = self.conv1d2(output)
        output = self.conv1d3(output)
        output = self.SE(output)

        x = self.conv1d4(x)

        output = output + x

        output = self.leakyrule(output)
        output = self.avp(output)

        return output

class SEnet(nn.Module):
    """docstring for SEnet"""
    def __init__(self):
        super(SEnet, self).__init__()
        self.PE = PositionalEmbedding(92)
        self.attention = nn.MultiheadAttention(92,2)
        self.blk1 = ResBlock2d(4)
        self.blk2 = ResBlock1d(4)
        self.blk3 = ResBlock1d(8)
        self.blk4 = ResBlock1d(16)
        self.avp = nn.AvgPool1d(2)
        self.gap = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # output = torch.sin(x)
        # x += output
        x = self.blk1(x)
        x = torch.squeeze(x, dim = 3)

        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = self.avp(x)
        out = x
        x += self.PE(x)
        x = x.permute(1, 0, 2)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)
        x += out

        x = self.gap(x)
        return x

class mSEnet(nn.Module):
    """docstring for mSEnet"""
    def __init__(self):
        super(mSEnet, self).__init__()
        self.senet = SEnet()

    def forward(self, x):
        x = torch.unsqueeze(x, dim = 1)
        x = torch.unsqueeze(x, dim = -1)
        x = self.senet(x)

        return x