import torch.nn as nn

class TrendHeadLinearPerDim(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.linear = nn.Linear(1, 3)

    def forward(self, x):  # [B, T-1, D]
        x = x.unsqueeze(-1)        # [B, T-1, D, 1]
        return self.linear(x)      # [B, T-1, D, 3]
