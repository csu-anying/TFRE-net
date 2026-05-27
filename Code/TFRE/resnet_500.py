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


class ResBlock1d0(nn.Module):
    """docstring for ResBlock2d"""

    def __init__(self, filters):
        super(ResBlock1d0, self).__init__()
        filters0 = filters
        if filters != 4:
            filters0 = filters // 2

        self.conv1d0 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size=3, padding=(3 - 1) // 2),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        self.SE = SE_Module(filters, dim=1)

        self.conv1d5 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size=1, padding=0),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        self.leakyrule = nn.LeakyReLU(0.3)
        self.avp = nn.AvgPool1d(2)

    def forward(self, x):
        output = x
        output = self.conv1d0(output)
        output = self.SE(output)

        x = self.conv1d5(x)

        output = output + x

        output = self.leakyrule(output)
        output = self.avp(output)

        return output

class ResBlock1d1(nn.Module):
    """docstring for ResBlock2d"""

    def __init__(self, filters):
        super(ResBlock1d1, self).__init__()
        filters0 = filters
        if filters != 4:
            filters0 = filters // 2

        self.conv1d0 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size=5, padding=(5 - 1) // 2),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        self.SE = SE_Module(filters, dim=1)

        self.conv1d5 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size=1, padding=0),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        self.leakyrule = nn.LeakyReLU(0.3)
        self.avp = nn.AvgPool1d(2)

    def forward(self, x):
        output = x
        output = self.conv1d0(output)
        output = self.SE(output)

        x = self.conv1d5(x)

        output = output + x

        output = self.leakyrule(output)
        output = self.avp(output)

        return output

class ResBlock1d2(nn.Module):
    """docstring for ResBlock2d"""

    def __init__(self, filters):
        super(ResBlock1d2, self).__init__()
        filters0 = filters
        if filters != 4:
            filters0 = filters // 2

        self.conv1d0 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size=7, padding=(7 - 1) // 2),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        self.SE = SE_Module(filters, dim=1)

        self.conv1d5 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size=1, padding=0),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        self.leakyrule = nn.LeakyReLU(0.3)
        self.avp = nn.AvgPool1d(2)

    def forward(self, x):
        output = x
        output = self.conv1d0(output)
        output = self.SE(output)

        x = self.conv1d5(x)

        output = output + x

        output = self.leakyrule(output)
        output = self.avp(output)

        return output

class ResBlock1d3(nn.Module):
    """docstring for ResBlock2d"""

    def __init__(self, filters):
        super(ResBlock1d3, self).__init__()
        filters0 = filters
        if filters != 4:
            filters0 = filters // 2

        self.conv1d0 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size=9, padding=(9 - 1) // 2),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        self.SE = SE_Module(filters, dim=1)

        self.conv1d5 = nn.Sequential(
            nn.Conv1d(filters0, filters, kernel_size=1, padding=0),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(0.3)
        )

        self.leakyrule = nn.LeakyReLU(0.3)
        self.avp = nn.AvgPool1d(2)

    def forward(self, x):
        output = x
        output = self.conv1d0(output)
        output = self.SE(output)

        x = self.conv1d5(x)

        output = output + x

        output = self.leakyrule(output)
        output = self.avp(output)

        return output

class SEnet(nn.Module):
    """docstring for SEnet"""

    def __init__(self):
        super(SEnet, self).__init__()
        # self.blk1 = ResBlock2d(4)
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size= 1, padding= 0),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.3)
        )
        self.blk2 = ResBlock1d0(4)
        self.blk3 = ResBlock1d1(8)
        self.blk4 = ResBlock1d2(16)
        self.blk5 = ResBlock1d3(32)
        self.avp = nn.AvgPool1d(2)
        self.gap = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # output = torch.sin(x)
        # x += output
        # output = x
        output = torch.cat([x, x,x,x], dim=1)
        # x = self.blk1(x)
        x = self.conv1d(x)
        x += output
        # output = torch.squeeze(output, dim=3)

        # x = torch.squeeze(x, dim=3)

        # output = torch.cat([output, output], dim=1)
        # x += output
        #
        # output = x
        # print(output.shape)
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
