import torch.nn as nn

class TrendHeadConv1x1(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)

    def forward(self, x):  # [B, T-1, D]
        x = x.unsqueeze(1)         # [B, 1, T-1, D]
        x = self.conv(x)           # [B, 3, T-1, D]
        return x.permute(0, 2, 3, 1)  # [B, T-1, D, 3]
