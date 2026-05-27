import torch
import torch.nn as nn
# from torchsummary import summary
import numpy as np
import sys

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# class SE_Module(nn.Module):
#
#     def __init__(self, in_channels, ratio=4, dim=1):
#         super(SE_Module, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         if in_channels > ratio:
#             self.fc1 = nn.Linear(in_channels, in_channels // ratio)
#             self.relu = nn.ReLU(inplace=True)
#             self.fc2 = nn.Linear(in_channels // ratio, in_channels)
#         else:
#             self.fc1 = nn.Linear(in_channels, in_channels)
#             self.relu = nn.ReLU(inplace=True)
#             self.fc2 = nn.Linear(in_channels, in_channels)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         y = x.view(b*c,h,w)
#         # print("1:")
#         # print(y.shape)#torch.Size([128, 12, 5000])
#         y = self.avg_pool(y).squeeze(-1)
#         # print("2:")
#         # print(y.shape)#torch.Size([64, 2, 12])
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y).view(b, c, h, 1)
#         return x * y

class SE_Module(nn.Module):

    def __init__(self, in_channels, ratio=4, dim=1):
        super(SE_Module, self).__init__()
        self.in_channels = in_channels
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_channels*12, out_features=in_channels*12 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_channels*12//2, out_features=in_channels*12),
            nn.Sigmoid()
        )


    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        out = x.view(b,c*h,w)
        out = self.squeeze(out)
        out = out.reshape(out.shape[0], out.shape[1])
        scale = self.excitation(out)
        scale = scale.reshape(b, c, h, 1)

        return identity * scale.expand_as(identity)

class block1(nn.Module):
    """docstring for ResBlock2d"""

    def __init__(self, filters):
        super(block1, self).__init__()

        self.conv2d0 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(1, filters // 2, kernel_size=(3, 1), padding=(1, 0))
        )
        self.SE1 = SE_Module(filters // 2, dim=2)

        self.skipconv0 = nn.Sequential(
            nn.Conv2d(1, filters // 2, kernel_size=(1, 1), padding=(0,0)),
            nn.BatchNorm2d(filters // 2)
        )

        self.skipconv1 = nn.Sequential(
            nn.Conv2d(filters // 2,filters , kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(filters)
        )

        self.conv2d1 = nn.Sequential(
            nn.BatchNorm2d(filters // 2),
            nn.LeakyReLU(0.3),
            nn.Conv2d(filters // 2, filters, kernel_size=(5, 1), padding=(2, 0))
        )

        self.conv2d2 = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.3),
            nn.Conv2d(filters, filters, kernel_size=(7, 1), padding=(3, 0))
        )

        self.SE2 = SE_Module(filters, dim=2)
        # self.avp = nn.AvgPool1d(10)

    def forward(self, x):
        output0 = x
        output0 = self.conv2d0(output0)
        output0 = self.SE1(output0)
        x = self.skipconv0(x)
        output0 = output0 + x

        output1 = output0
        output0 = self.skipconv1(output0)
        output1 = self.conv2d1(output1)
        # print("1:")#torch.Size([128, 4, 12, 5000])
        # print(output1.shape)
        output1 = self.SE2(output1)
        output1 = output1 + output0

        output2 = output1
        output2 = self.conv2d2(output2)
        output2 = self.SE2(output2)
        output2 = output1 + output2

        output2 = torch.mean(output2.view(output2.size(0), output2.size(1), output2.size(2), -1, 10), dim=4)
        return output2


class block2(nn.Module):
    """docstring for ResBlock2d"""

    def __init__(self, filters):
        super(block2, self).__init__()

        filters0 = filters
        if filters != 4:
            filters0 = filters // 2

        self.conv2d0 = nn.Sequential(
            nn.BatchNorm2d(filters0),
            nn.LeakyReLU(0.3),
            nn.Conv2d(filters0, filters // 2, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(filters // 2),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.5),
            nn.Conv2d(filters // 2, filters // 2, kernel_size=(11, 1), padding=(5, 0))
        )

        self.skipconv0 = nn.Sequential(
            nn.Conv2d(filters0, filters // 2, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(filters // 2)
        )

        self.skipconv1 = nn.Sequential(
            nn.Conv2d(filters // 2, filters, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(filters)
        )

        self.conv2d1 = nn.Sequential(
            nn.BatchNorm2d(filters // 2),
            nn.LeakyReLU(0.3),
            nn.Conv2d(filters // 2, filters, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.5),
            nn.Conv2d(filters, filters, kernel_size=(11, 1), padding=(5, 0))
        )

        self.conv2d2 = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.3),
            nn.Conv2d(filters, filters, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.5),
            nn.Conv2d(filters, filters, kernel_size=(11, 1), padding=(5, 0))
        )

        self.SE0 = SE_Module(filters // 2, dim=2)
        self.SE1 = SE_Module(filters, dim=2)

        # self.avp = nn.AvgPool1d(10)
        self.leakyrule = nn.LeakyReLU(0.3)

    def forward(self, x):
        output0 = x
        x = self.skipconv0(x)
        output0 = self.conv2d0(output0)
        output0 = output0 + x
        output0 = self.SE0(output0)

        output1 = output0
        output0 = self.skipconv1(output0)
        output1 = self.conv2d1(output1)
        output1 = self.SE1(output1)
        output1 = output1 + output0

        output2 = output1
        output2 = self.conv2d2(output2)
        output2 = self.SE1(output2)
        output2 = output1 + output2

        output3 = output2
        output3 = self.conv2d2(output3)
        output3 = self.SE1(output3)
        output3 = output2 + output3

        # output1 = self.leakyrule(output1)
        output3 = torch.mean(output3.view(output3.size(0), output3.size(1), output3.size(2), -1, 10), dim=4)
        return output3

class block3(nn.Module):
    """docstring for ResBlock2d"""

    def __init__(self, filters,n):
        super(block3, self).__init__()

        filters0 = filters
        if filters != 4:
            filters0 = filters // 2

        self.conv2d0 = nn.Sequential(
            nn.BatchNorm2d(filters0),
            nn.LeakyReLU(0.3),
            nn.Conv2d(filters0, filters // 2, kernel_size=(n, 3), padding=( (n-1) // 2,1)),
            nn.BatchNorm2d(filters // 2),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.5),
            nn.Conv2d(filters // 2, filters // 2, kernel_size=(n+6, 1), padding=( (n+6-1) // 2,0))
        )

        self.skipconv0 = nn.Sequential(
            nn.Conv2d(filters0, filters // 2, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(filters // 2)
        )

        self.skipconv1 = nn.Sequential(
            nn.Conv2d(filters // 2, filters, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(filters)
        )

        self.conv2d1 = nn.Sequential(
            nn.BatchNorm2d(filters // 2),
            nn.LeakyReLU(0.3),
            nn.Conv2d(filters // 2, filters, kernel_size=(n, 3), padding=( (n-1) // 2,1)),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.5),
            nn.Conv2d(filters, filters, kernel_size=(n+2, 1), padding=( (n+2-1) // 2,0))
        )

        self.conv2d2 = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.3),
            nn.Conv2d(filters, filters,kernel_size=(n+4, 3), padding=( (n+4-1) // 2,1)),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.5),
            nn.Conv2d(filters, filters, kernel_size=(n+6, 1), padding=( (n+6-1) // 2,0))
        )

        self.SE0 = SE_Module(filters // 2, dim=2)
        self.SE1 = SE_Module(filters, dim=2)

        # self.avp = nn.AvgPool1d(25)
        self.leakyrule = nn.LeakyReLU(0.3)

    def forward(self, x):
        output0 = x
        x = self.skipconv0(x)
        output0 = self.conv2d0(output0)
        output0 = output0 + x
        output0 = self.SE0(output0)

        output1 = output0
        output0 = self.skipconv1(output0)
        output1 = self.conv2d1(output1)
        output1 = self.SE1(output1)
        output1 = output1 + output0

        output2 = output1
        output2 = self.conv2d2(output2)
        output2 = self.SE1(output2)
        output2 = output1 + output2

        output3 = output2
        output3 = self.conv2d2(output3)
        output3 = self.SE1(output3)
        output3 = output2 + output3

        output4 = output3
        output4 = self.conv2d2(output4)
        output4 = self.SE1(output4)
        output4 = output3 + output4

        output5 = output4
        output5 = self.conv2d2(output5)
        output5 = self.SE1(output5)
        output5 = output4 + output5
        # output1 = self.leakyrule(output1)

        output5 = torch.mean(output5.view(output5.size(0), output5.size(1), output5.size(2), -1, 5), dim=4)
        return output5

class SEnet(nn.Module):
    """docstring for SEnet"""

    def __init__(self):
        super(SEnet, self).__init__()
        self.blk1 = block1(4)
        self.blk2 = block2(4)
        self.blk3_1 = block3(8,3)
        self.blk3_2 = block3(8, 5)
        # self.blk3_3 = block3(8, 7)
        # self.gap = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # output = torch.sin(x)
        # x += output
        x = self.blk1(x)
        x = torch.squeeze(x, dim=3)
        x = self.blk2(x)
        x_1 = self.blk3_1(x)
        x_2 = self.blk3_2(x)
        # x_3 = self.blk3_3(x)
        # x = torch.cat([x_1,x_2,x_3], dim=1)
        x = torch.cat([x_1, x_2], dim=1)
        x = torch.max(x.view(x.size(0), x.size(1), x.size(2), -1, 10), dim=4)[0]
        # print("1:")
        # print(x.shape)
        return x


class mSEnet_DKR(nn.Module):
    """docstring for mSEnet"""

    def __init__(self):
        super(mSEnet_DKR, self).__init__()
        self.senet = SEnet()

    def forward(self, x):
        # print(x.shape)#torch.Size([128, 12, 5000])
        x = torch.unsqueeze(x, dim=1)
        # print(x.shape)#torch.Size([128, 1, 12, 5000])
        x = self.senet(x)
        # print(x.shape)#torch.Size([128, 4,1,1])
        return x