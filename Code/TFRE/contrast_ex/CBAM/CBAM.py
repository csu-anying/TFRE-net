import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=2, keepdim=False)
        max_pool, _ = torch.max(x, dim=2, keepdim=False)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = torch.sigmoid(avg_out + max_out).unsqueeze(2)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_attn = torch.sigmoid(self.conv(x_cat))
        return x * x_attn


class CBAMBlock(nn.Module):
    def __init__(self, in_channels):
        super(CBAMBlock, self).__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ECGNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ECGNet, self).__init__()

        self.dropout = nn.Dropout(p=0.5)

        self.input_conv = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
        )

        self.cbam = CBAMBlock(64)

        self.mobilenet_blocks = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1, groups=128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 12, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def augment(self, x):
        batch_size, channels, seq_len = x.size()
        factors = torch.rand(batch_size, 1, 1, device=x.device) * 2 - 1
        x_aug = x * (1 + 0.1 * factors)
        return x_aug

    def forward(self, x):
        x_aug = self.augment(x)
        x = self.input_conv(x_aug)
        x_enc = self.encoder(x)
        x_cbam = self.cbam(x_enc)

        class_feat = self.mobilenet_blocks(x_cbam)

        logits = self.global_pool(class_feat).squeeze(-1)

        logits = self.dropout(logits)
        class_out = self.classifier(logits)

        return class_out