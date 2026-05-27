import torch
from torch import nn

class Configs:
    def __init__(self):
        self.input_channels = 12  # 输入通道数
        self.mid_channels = 3    # 中间特征图的通道数
        self.final_out_channels = 8  # 最终输出特征图的通道数
        self.stride = 1           # 卷积层的步长
        self.dropout = 0.5        # Dropout 概率
        self.trans_dim = 8     # Transformer 编码器的输入维度 self.final_out_channels=configs.trans_dim 调整 configs.trans_dim
        # 使其等于 x_transposed.shape[-1]
        self.num_heads = 8        # Transformer 编码器的多头注意力头数
        self.num_classes = 7     # 分类任务中的类别数（可以根据实际需求调整）

class ecgTransForm(nn.Module):
    def __init__(self, configs, hparams):
        super(ecgTransForm, self).__init__()

        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[0],
                               stride=configs.stride, bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[1],
                               stride=configs.stride, bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=filter_sizes[2],
                               stride=configs.stride, bias=False, padding=(filter_sizes[2] // 2))

        self.bn = nn.BatchNorm1d(configs.mid_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.do = nn.Dropout(configs.dropout)


        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        
        self.inplanes = configs.final_out_channels
        self.crm = self._make_layer(SEBasicBlock, configs.final_out_channels, 2)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=configs.trans_dim, nhead=configs.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.aap = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(hparams["feature_dim"], configs.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_in):

        # Multi-scale Convolutions
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)

        x_concat = torch.mean(torch.stack([x1, x2, x3],2), 2)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))

        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)

        # Channel Recalibration Module
        x = self.crm(x)

        # 调整维度顺序以适应 TransformerEncoderLayer
        x_transposed = x.permute(2, 0, 1)  # (sequence_length, batch_size, feature_dim)

        # Bi-directional Transformer
        x1 = self.transformer_encoder(x_transposed)
        x2 = self.transformer_encoder(torch.flip(x_transposed, [0]))
        x_transformed = (x1 + x2).permute(1, 2, 0)  # (batch_size, feature_dim, sequence_length)

        x = self.aap(x_transformed)
        x_flat = x.reshape(x.shape[0], -1)
        x_out = self.clf(x_flat)
        return x_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        reduction = min(reduction, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
