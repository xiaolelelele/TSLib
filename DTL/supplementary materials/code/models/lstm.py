import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

# -------- 模型定义 -------- #

class Model(nn.Module):
    def __init__(self,configs, hidden_dim=512, num_layers=3, horizon=96, num_classes=3):
        super().__init__()
        input_dim = configs.enc_in
        self.horizon = horizon
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes * horizon)

    def forward(self, x,padding_mask, h,t):
        out, _ = self.lstm(x)             # [batch, window, hidden*2]
        h_last = out[:, -1, :]            # [batch, hidden*2]
        logits = self.classifier(h_last)  # [batch, num_classes*horizon]
        logits = logits.view(-1, self.horizon, 3)
        return logits                     # [batch, horizon, num_classes]