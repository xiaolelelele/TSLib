import torch
import torch.nn as nn

class TrendHeadPerDimMLPSeparate(nn.Module):
    def __init__(self, d_in, hidden=8, dropout=0.1):
        super().__init__()
        self.mlp_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 3)
            ) for _ in range(d_in)
        ])

    def forward(self, x):  # [B, T-1, D]
        B, T, D = x.shape
        out = []
        for i in range(D):
            xi = x[:, :, i].unsqueeze(-1)     # [B, T-1, 1]
            out_i = self.mlp_list[i](xi)      # [B, T-1, 3]
            out.append(out_i.unsqueeze(2))    # [B, T-1, 1, 3]
        return torch.cat(out, dim=2)          # [B, T-1, D, 3]
