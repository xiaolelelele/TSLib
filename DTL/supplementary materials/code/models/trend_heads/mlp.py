import torch.nn as nn
import torch.nn.functional as F

class TrendHeadMLPPerDim(nn.Module):
    def __init__(self, d_in, hidden=8, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3)
        )

    def forward(self, x):  # [B, T-1, D]
        x = x.unsqueeze(-1)        # [B, T-1, D, 1]
        return self.mlp(x)         # [B, T-1, D, 3]
