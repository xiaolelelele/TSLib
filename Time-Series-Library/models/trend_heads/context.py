# context.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrendContextHead(nn.Module):
    def __init__(self, d_in, context_size=3, hidden=16, dropout=0.2):
        super().__init__()
        self.context_size = context_size
        self.padding = context_size // 2
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=hidden,
            kernel_size=(context_size, 1),
            padding=(self.padding, 0)
        )
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3)
        )

    def forward(self, x):  # [B, T-1, D]
        x = x.unsqueeze(1)  # [B, 1, T-1, D]
        x = self.conv(x)    # [B, hidden, T-1, D]
        x = x.permute(0, 2, 3, 1)  # [B, T-1, D, hidden]
        return self.proj(x)        # [B, T-1, D, 3]
