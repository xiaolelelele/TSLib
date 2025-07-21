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
        self.fc = nn.Linear(hidden_dim * 2, 3)
        self.forecast_horizon = forecast_horizon
        # self.labels = torch.tensor([-1.0, 0.0, 1.0], device='cpu', dtype=torch.float32)

    def forward(self, x):
        out, _ = self.encoder(x)
        context = out[:, -1, :]
        # (batch_size, forecast_horizon, hidden_dim*2)
        repeated = context.unsqueeze(1).repeat(1, self.forecast_horizon, 1)
        logits = self.fc(repeated)
        probs = F.softmax(logits, dim=-1) # (batch_size, forecast_horizon, 3)
        return probs

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
        trend_probs = self.trend_classifier(x)         # (B, H, 3) （3，1）
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
        return trend_probs, mag_pred # Gradcheck 需要只返回一个 tensor

# -------- gradcheck 验证 + 输出检查 -------- #
# -------- gradcheck 验证 + 输出检查 -------- #
if __name__ == "__main__":
    # 构造模型配置
    class Config:
        enc_in = 4
        d_model = 8
        e_layers = 1
        pred_len = 5

    configs = Config()
    model = Model(configs)
    model = model.double()  # gradcheck 要求模型参数为 double 类型
    model.eval()


    # 构造输入数据（需要 double 类型并且 requires_grad=True 才能做 gradcheck）
    batch_size = 2
    seq_len = 10
    input_dim = configs.enc_in
    forecast_horizon = configs.pred_len

    x = torch.randn(batch_size, seq_len, input_dim, dtype=torch.double, requires_grad=True)
    y0 = torch.randn(batch_size, dtype=torch.double, requires_grad=True)

    # gradcheck 要求所有输入都是 double 且 requires_grad=True
    def func(*inputs):
        trend_probs, mag_pred, pred_seq = model(*inputs)
        # 只返回一个 tensor，取 pred_seq
        return pred_seq

    # gradcheck 验证
    try:
        gradcheck(func, (x, y0), eps=1e-6, atol=1e-4)
        print("Gradcheck passed!")
    except Exception as e:
        print("Gradcheck failed:", e)

    # 输出检查
    with torch.no_grad():
        trend_probs, mag_pred, pred_seq = model(x, y0)
        print("trend_probs shape:", trend_probs.shape)  # (B, H, 3)
        print("mag_pred shape:", mag_pred.shape)        # (B, H)
        print("pred_seq shape:", pred_seq.shape)        # (B, H)
        print("trend_probs[0]:", trend_probs[0])
        print("mag_pred[0]:", mag_pred[0])
        print("pred_seq[0]:", pred_seq[0])