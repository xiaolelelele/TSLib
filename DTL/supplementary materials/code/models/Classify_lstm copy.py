import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

# -------- 模型定义 -------- #

class TrendClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, forecast_horizon):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 3*forecast_horizon)
        self.forecast_horizon = forecast_horizon
        # self.labels = torch.tensor([-1.0, 0.0, 1.0], device='cpu', dtype=torch.float32)

    def forward(self, x):
        out, _ = self.encoder(x)
        context = out[:, -1, :]
        logits = self.fc(context)
        logits = logits.view(-1, self.forecast_horizon, 3)
        return logits

class MagnitudeRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, forecast_horizon):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, x):
        out, _ = self.encoder(x)
        final = out[:, -1, :]
        mag = F.relu(self.fc(final))
        return mag

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__() 
        # input_dim, hidden_dim=16, num_layers=1, forecast_horizon=5
        input_dim = configs.enc_in
        print(f"input_dim: {input_dim}, enc_in: {configs.enc_in}")
        hidden_dim = configs.d_model
        num_layers = configs.e_layers 
        forecast_horizon = configs.pred_len 
        self.trend_classifier = TrendClassifier(input_dim, hidden_dim, num_layers, forecast_horizon)
        self.magnitude_regressor = MagnitudeRegressor(input_dim, hidden_dim, num_layers, forecast_horizon)
        self.forecast_horizon = forecast_horizon

    def forward(self, x, y0):
        trend_logits = self.trend_classifier(x)         # (B, H, 3) （3，1）
        mag_pred = self.magnitude_regressor(x)         # (B, H)
        ######################   0--》1
        # 软标签
        # weights = torch.tensor([-1.0, 0.0, 1.0], device=x.device, dtype=x.dtype)
        # soft_sign = torch.matmul(trend_probs, weights)  # (B, H)
        # delta = soft_sign * mag_pred  #(batch_size, forecast_horizon)                 # (B, H)
        # 直接选择softmax最大值的类别标签作为系数
        # idx = torch.argmax(trend_probs, dim=-1)        # (B, H)
        # labels = torch.tensor([-1.0, 0.0, 1.0], device=x.device, dtype=x.dtype)
        # coef = labels[idx]                             # (B, H)
        # delta = coef * mag_pred                        # (B, H)
        # # pred还原需要反归一化mag，先忽略这部分loss
        # pred_seq = torch.cumsum(delta, dim=-1) + y0.unsqueeze(-1)
        #  [1,2,3]  1 3 6
        return trend_logits, mag_pred # Gradcheck 需要只返回一个 tensor
