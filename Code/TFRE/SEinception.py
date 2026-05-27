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

    def __init__(self, filters0, filters):
        super(ResBlock1d, self).__init__()

        self.conv1d1 = nn.Sequential(
            nn.Conv1d(filters0, filters0 * 2, kernel_size=3, padding=(3 - 1) // 2),
            nn.BatchNorm1d(filters0 * 2),
            nn.LeakyReLU(0.3),
            nn.Conv1d(filters0 * 2, filters0 * 2, kernel_size=7, padding=(7 - 1) // 2),
            nn.BatchNorm1d(filters0 * 2),
            nn.LeakyReLU(0.3),
            nn.Conv1d(filters0 * 2, filters0 * 2, kernel_size=11, padding=(11 - 1) // 2),
            nn.BatchNorm1d(filters0 * 2),
            nn.LeakyReLU(0.3),
            nn.Conv1d(filters0 * 2, filters, kernel_size=17, padding=(17 - 1) // 2),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3),
            nn.AvgPool1d(2)
        )

        nn.init.xavier_uniform_(self.conv1d1[0].weight)  # Xavier初始化卷积层权重
        self.conv1d1[0].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d1[1].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d1[1].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零
        nn.init.xavier_uniform_(self.conv1d1[3].weight)  # Xavier初始化卷积层权重
        self.conv1d1[3].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d1[4].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d1[4].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零
        nn.init.xavier_uniform_(self.conv1d1[6].weight)  # Xavier初始化卷积层权重
        self.conv1d1[6].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d1[7].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d1[7].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零
        nn.init.xavier_uniform_(self.conv1d1[9].weight)  # Xavier初始化卷积层权重
        self.conv1d1[9].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d1[10].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d1[10].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零

        self.conv1d2 = nn.Sequential(
            nn.Conv1d(filters0, filters0 * 2, kernel_size=17, padding=(17 - 1) // 2),
            nn.BatchNorm1d(filters0 * 2),
            nn.LeakyReLU(0.3),
            nn.Conv1d(filters0 * 2, filters0 * 2, kernel_size=11, padding=(11 - 1) // 2),
            nn.BatchNorm1d(filters0 * 2),
            nn.LeakyReLU(0.3),
            nn.Conv1d(filters0 * 2, filters0 * 2, kernel_size=7, padding=(7 - 1) // 2),
            nn.BatchNorm1d(filters0 * 2),
            nn.LeakyReLU(0.3),
            nn.Conv1d(filters0 * 2, filters, kernel_size=3, padding=(3 - 1) // 2),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3),
            nn.AvgPool1d(2)
        )

        nn.init.xavier_uniform_(self.conv1d2[0].weight)  # Xavier初始化卷积层权重
        self.conv1d2[0].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d2[1].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d2[1].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零
        nn.init.xavier_uniform_(self.conv1d2[3].weight)  # Xavier初始化卷积层权重
        self.conv1d2[3].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d2[4].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d2[4].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零
        nn.init.xavier_uniform_(self.conv1d2[6].weight)  # Xavier初始化卷积层权重
        self.conv1d2[6].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d2[7].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d2[7].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零
        nn.init.xavier_uniform_(self.conv1d2[9].weight)  # Xavier初始化卷积层权重
        self.conv1d2[9].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d2[10].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d2[10].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零

        self.conv1d3 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size=1),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3),
            nn.AvgPool1d(2)
        )

        nn.init.xavier_uniform_(self.conv1d3[0].weight)  # Xavier初始化卷积层权重
        self.conv1d3[0].bias.data.fill_(0.0)  # 初始化卷积层偏置为零
        self.conv1d3[1].weight.data.fill_(1)  # 批归一化层权重初始化为1
        self.conv1d3[1].bias.data.fill_(0.0)  # 批归一化层偏置初始化为零

        self.SE = SE_Module(3 * filters, dim=1)
        self.SE1 = SE_Module(filters, dim=1)
        self.SE2 = SE_Module(filters, dim=1)
        self.SE3 = SE_Module(filters, dim=1)
        self.leakyrule = nn.LeakyReLU(0.3)
        self.avp = nn.AvgPool1d(2)

    def forward(self, x):
        output = x
        branch1 = self.conv1d1(output)
        branch2 = self.conv1d2(output)
        branch3 = self.conv1d3(output)
        branch1 = self.SE1(branch1)
        branch2 = self.SE2(branch2)
        branch3 = self.SE3(branch3)
        output = [branch1, branch2, branch3]
        output = torch.cat(output, 1)
        x = self.SE(output)

        output = output + x

        output = self.avp(output)

        return output


class SEnet(nn.Module):
    """docstring for SEnet"""

    def __init__(self):
        super(SEnet, self).__init__()
        # self.blk1 = ResBlock2d(4)  # [64, 1, 5000]
        self.blk2 = ResBlock1d(1, 4)  # [64, 12, 1250]
        self.blk3 = ResBlock1d(12, 36)  # [64, 108, 312]
        # self.blk4 = ResBlock1d(36,36)
        # self.blk5 = ResBlock1d(32)
        self.avp1 = nn.AvgPool1d(2)  # [64, 108, 156]
        self.avp2 = nn.AvgPool1d(2)
        self.avp3 = nn.AvgPool1d(2)
        self.gap = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # output = torch.cat([x, x,x,x], dim=1)

        # x = self.blk1(x)
        # output = torch.squeeze(output, dim=3)

        # x = torch.squeeze(x, dim=3)

        # output = torch.cat([output, output], dim=1)
        # x += output
        #
        # output = x
        # print(output.shape)
        # branch1 = self.blk2_1(x)
        # branch2 = self.blk2_2(x)
        x = self.blk2(x)
        # print(("x:"))
        # print(x.shape)
        # x += output
        # print("x+:")
        # print((x.shape))
        # x = x + branch1 + branch2  # [64, 12, 1250]

        # output = x
        x = self.blk3(x)
        # output =  torch.cat([output, output], dim=1)
        # x += output
        # output = x
        # x = self.blk4(x)
        # output = torch.cat([output, output], dim=1)
        # x += output
        # x = self.blk5(x)
        x = self.avp1(x)

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
