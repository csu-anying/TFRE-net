import torch
import torch.nn as nn
import torch.nn.functional as F


class SRTNet(nn.Module):
    def __init__(self):
        super(SRTNet, self).__init__()
        # Scanning Module
        self.scanning_branches = nn.ModuleList([self._create_scanning_branch() for _ in range(12)])

        # Reading Module
        self.dense_block1 = self._create_dense_block(48, 24, 2)
        self.transition1 = self._create_transition_layer(96, 36)
        self.dense_block2 = self._create_dense_block(36, 24, 7)
        self.transition2 = self._create_transition_layer(204, 60)
        self.dense_block3 = self._create_dense_block(60, 24, 2)
        self.transition3 = self._create_transition_layer(108, 84)

        # Thinking Module
        self.self_attention = nn.MultiheadAttention(embed_dim=84, num_heads=3)
        self.ffn = nn.Sequential(
            nn.Linear(84, 200),
            nn.ReLU(),
            nn.Linear(200, 84)
        )
        self.norm1 = nn.LayerNorm(84)
        self.norm2 = nn.LayerNorm(84)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def _create_scanning_branch(self):
        return nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=17, stride=1, padding=8),
            nn.BatchNorm1d(2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(2, 4, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def _create_dense_block(self, in_channels, growth_rate, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1),
                nn.Dropout(p=0.5)
            ))
            in_channels += growth_rate
        return nn.Sequential(*layers)

    def _create_transition_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        batch_size, num_leads, length = x.shape
        assert num_leads == 12, "Input should have 12 leads"

        # Scanning Module
        outputs = []
        for i in range(num_leads):
            lead_input = x[:, i, :].unsqueeze(1)  # Shape: [batch_size, 1, length]
            output = self.scanning_branches[i](lead_input)  # Each branch processes one lead
            outputs.append(output)
        x = torch.cat(outputs, dim=1)  # Shape: [batch_size, 48, reduced_length]

        # Reading Module
        # Reading Module

        # Dense Block 1
        for layer in self.dense_block1:
            new_features = layer(x)  # 提取新特征
            x = torch.cat([x, new_features], dim=1)  # 拼接新特征
        x = self.transition1(x)  # Transition Layer 压缩特征

        # Dense Block 2
        for layer in self.dense_block2:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)
        x = self.transition2(x)  # Transition Layer 压缩特征

        # Dense Block 3
        for layer in self.dense_block3:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)
        x = self.transition3(x)  # Transition Layer 压缩特征

        # Thinking Module
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, seq_len, features] for attention
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ffn(x)
        x = self.norm2(x + ff_output) # [64, 156, 84]
        x = self.global_avg_pool(x.permute(0, 2, 1)).squeeze(-1) # [64, 84]

        return x

#
# # Example usage
# if __name__ == "__main__":
#     model = SRTNet()
#     sample_input = torch.randn(64, 12, 5000)  # Batch size 64, 12 leads, length 5000
#     output = model(sample_input)
#     print(f"Output shape: {output.shape}")
