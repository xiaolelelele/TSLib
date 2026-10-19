import torch
import torch.nn as nn
import torch.nn.functional as F

class TrendLoss(nn.Module):
    def __init__(self, tau=0.05, alpha=0.1, warmup_epoch=3, reduction='mean'):
        """
        Args:
            tau: Temperature for soft label smoothing
            alpha: Weight for trend loss in total loss
            warmup_epoch: Delay epoch to start applying trend loss
            reduction: Reduction method for KLDivLoss
        """
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.warmup_epoch = warmup_epoch
        self.reduction = reduction

    def build_soft_trend_label(self, delta_true):
        """
        delta_true: [B, T-1, D]
        returns: [B, T-1, D, 3]
        """
        score_down = -delta_true
        score_flat = -torch.abs(delta_true)
        score_up = delta_true

        scores = torch.stack([score_down, score_flat, score_up], dim=-1)  # [B, T-1, D, 3]
        probs = F.softmax(scores / self.tau, dim=-1)
        return probs

    def forward(self, logits, y_pred, y_true, epoch,means, stdev):
        """
        logits: trend logits, shape [B, T-1, D, 3]
        y_pred: model prediction [B, T, D]
        y_true: ground truth [B, T, D]
        epoch: current training epoch (int)
        """
        if epoch < self.warmup_epoch:
            return torch.tensor(0.0, device=logits.device)
        # 归一化 y_true
        # print("Y True:", y_true[0,:10, -1])
        y_true_norm = (y_true - means) / stdev
        delta_true = y_true_norm[:, 1:, :] - y_true_norm[:, :-1, :]
        soft_labels = self.build_soft_trend_label(delta_true)

        log_probs = F.log_softmax(logits, dim=-1)  # [B, T-1, D, 3]
        # print("logits", logits.shape)
        # print("log_probs", log_probs.shape)   
        # print("y_true", y_true.shape)
        # print("delta_true", delta_true.shape)
        # print("soft_labels", soft_labels.shape)
        kl_loss = F.kl_div(log_probs, soft_labels, reduction=self.reduction)
        # print("KL Loss:", kl_loss.item())
        # print("Soft Labels:", soft_labels[0, :10, -1, :])
        # print("Logits:", logits[0, :10, -1, :])
        # print("Delta True:", delta_true[0, :10, -1])
        # print("Y True Norm:", y_true_norm[0,:10, -1])
        # print("Y Pred:", y_pred[0, :10, -1])

        return self.alpha * kl_loss
