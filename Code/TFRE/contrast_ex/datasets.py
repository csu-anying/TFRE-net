from torch.utils.data import Dataset
import torch
import numpy as np

cuda = True if torch.cuda.is_available() else False

class matDataset(Dataset):#X是输入数据集，Y、Y5、Y7分别是三种分类的标签
    def __init__(self, X, Y, Y5, Y7):
        self.X = X
        self.Y = Y
        self.Y5 = Y5
        self.Y7 = Y7

    def __getitem__(self, index):#应该是拿出指定导联的数据？
        idx = index % len(self.Y)
        data = []
        for x in self.X:
            data.append(x[idx])
        x = data
        y = self.Y[idx]
        y5 = self.Y5[idx]
        y7 = self.Y7[idx]
        
        if cuda:
            for i in range(len(x)):
                x[i] = x[i].cuda()
            y = y.cuda()
            y5 = y5.cuda()
            y7 = y7.cuda()
        return x, y, y5, y7

    def __len__(self):
        return len(self.Y)