import torch
import torch.nn as nn
# from torchsummary import summary
import numpy as np
import sys

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
        for layer in self.excitation:
            if isinstance(layer, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)

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


class ResBlock1d(nn.Module):
    """docstring for ResBlock2d"""

    def __init__(self, filters):
        super(ResBlock1d, self).__init__()
        filters0 = filters
        if filters != 4:
            filters0 = filters // 2

        self.conv1d1 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size=17, padding=(17 - 1) // 2),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        nn.init.xavier_uniform_(self.conv1d1[0].weight)  # Xavier初始化卷积层权重
        self.conv1d1[0].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d1[1].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d1[1].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零

        self.conv1d2 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_size=11, padding=(11 - 1) // 2),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        nn.init.xavier_uniform_(self.conv1d2[0].weight)  # Xavier初始化卷积层权重
        self.conv1d2[0].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d2[1].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d2[1].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零

        self.conv1d3 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_size=5, padding=(5 - 1) // 2),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        nn.init.xavier_uniform_(self.conv1d3[0].weight)  # Xavier初始化卷积层权重
        self.conv1d3[0].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d3[1].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d3[1].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零

        self.SE = SE_Module(filters, dim=1)

        self.conv1d4 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size=1, padding=0),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        nn.init.xavier_uniform_(self.conv1d4[0].weight)  # Xavier初始化卷积层权重
        self.conv1d4[0].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d4[1].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d4[1].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零

        self.leakyrule = nn.LeakyReLU(0.3)
        self.avp = nn.AvgPool1d(2)

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
        # self.blk1 = ResBlock2d(4)
        self.conv2d = nn.Sequential(
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), padding=(2, 0)),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.3)
        )  # [64, 4, 5000]
        # self.con = nn.Sequential(
        #     nn.Conv1d(2, 4, kernel_size=1, padding=0),
        #     nn.BatchNorm1d(4),
        #     nn.LeakyReLU(0.3)
        # )  # [64, 4, 5000]
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size= 1, padding= 0),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.3)
        )  # [64, 4, 5000]
        # self.tanh = nn.Tanh()
        # self.prelu = nn.PReLU()
        # self.leakyrelu = nn.LeakyReLU(0.3)
        self.blk2 = ResBlock1d(4)  # [64, 4, 2500]
        self.blk3 = ResBlock1d(8)  # [64, 8, 1250]
        self.blk4 = ResBlock1d(16)  # [64, 16, 625]
        self.blk5 = ResBlock1d(32)  # [64, 32, 312]
        self.conv1d2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size= 1, padding= 0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3),
            nn.AvgPool1d(2)
        )  # [64, 32, 156]
        self.conv1d3 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size= 1, padding= 0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3),
            nn.AvgPool1d(2)
        )  # [64, 32, 78]
        self.avp = nn.AvgPool1d(2)  # [64, 32, 39]
        self.gap = nn.AdaptiveMaxPool1d(1)


    def forward(self, x):
        # output = torch.sin(x)
        # x += output
        # output = x
        # output = torch.cat([x, x,x,x], dim=1)
        # x = self.blk1(x)
        if x.dim() == 4:
            x = self.conv2d(x)
            x = x.squeeze(1)
            # x = self.con(x)
        else:
            x = self.conv1d(x)
        # x1 = self.leakyrelu(x)
        # x2 = self.tanh(x)
        # x3 = self.prelu(x)
        # x = torch.cat([x, x1,x2,x3], dim=1)
        # output = torch.squeeze(output, dim=3)

        # x = torch.squeeze(x, dim=3)

        # output = torch.cat([output, output], dim=1)
        # x += output
        #
        # output = x
        # print(x.shape)
        x = self.blk2(x)
        # print(("x:"))
        # print(x.shape)
        # x += output
        # print("x+:")
        # print((x.shape))

        # output = x
        x = self.blk3(x)
        # output =  torch.cat([output, output], dim=1)
        # x += output
        # output = x
        x = self.blk4(x)
        # output = torch.cat([output, output], dim=1)
        # x += output
        x = self.blk5(x)

        # x = self.conv1d2(x)
        # x = self.conv1d3(x)
        x = self.gap(x)
        return x


class mSEnet(nn.Module):
    """docstring for mSEnet"""

    def __init__(self):
        super(mSEnet, self).__init__()
        self.senet = SEnet()

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        # x = torch.unsqueeze(x, dim=-1)
        x = self.senet(x)

        return x
