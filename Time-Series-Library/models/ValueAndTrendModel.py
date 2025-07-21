import torch.nn as nn
import torch

class ValueAndTrendModel(nn.Module):
    def __init__(self, trend_hidden=16):
        super().__init__()
        self.trend_head = nn.Sequential(
            nn.Linear(1, trend_hidden),
            nn.Tanh(),
            nn.Linear(trend_hidden, 3)  # 三类：0=下降, 1=持平, 2=上升
        )
        self.value_loss = nn.MSELoss()
        self.trend_loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        """
        y_pred: 预测值, shape (B, T, D)
        y_true: 真实值, shape (B, T, D)
        """
        # 差分
        d_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]  # (B, T-1, D)
        d_true = y_true[:, 1:, :] - y_true[:, :-1, :]  # (B, T-1, D)

        # === 缩放差分值 ===
        # mean = d_pred.mean(dim=(0, 1), keepdim=True)  # shape (1, 1, D)
        std = d_pred.std(dim=(0, 1), keepdim=True) + 1e-6  # 防止除以0
        # d_pred_norm = (d_pred - mean) / std  # shape (B, T-1, D)
        d_pred_norm = d_pred / std  # shape (B, T-1, D)
        # 构造真实趋势标签 ∈ {0,1,2}
        tau = 0.0
        label_true = torch.where(d_true > tau, 2,
                        torch.where(d_true < -tau, 0, 1))  # (B, T-1, D)

        # 预测 logits
        logits = self.trend_head(d_pred_norm.unsqueeze(-1))  # (B, T-1, D, 3)
        # 查看logits学习情况
        print("d_pred", d_pred[0, 0, :10])
        print("d_pred_norm",d_pred_norm[0, 0, :10])
        print("logits",logits[0, 0, :10,:])
        print("label_true", label_true[0, 0, :10])
        # 计算交叉熵
        B, L, D, C = logits.shape
        logits_flat = logits.view(B * L * D, C)
        labels_flat = label_true.view(B * L * D).long()
        loss_tr = self.trend_loss(logits_flat, labels_flat)
        

        return loss_tr
