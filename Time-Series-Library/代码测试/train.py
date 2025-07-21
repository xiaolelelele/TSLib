import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 1. 数据集定义
class TimeSeriesMultiStepDataset(Dataset):
    def __init__(self, csv_path, window_size=128, horizon=96):
        """
        csv_path: 带 ot, delta, 以及其它特征的 CSV
        window_size: 用多少过去步长作为输入
        horizon: 预测多少步，默认为 96
        """
        df = pd.read_csv(csv_path, parse_dates=True)
        # 排除 ot 和 delta 列
        feats = df.drop(columns=['date','OT', 'delta'])  # 假设已经有 label
        labels = df['delta'].values  # 三分类标签
        self.X = feats.values.astype('float32')
        self.y = labels.astype('int64')
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        # 最后能取到 y[idx + horizon - 1]
        return len(self.X) - self.horizon - self.window_size + 1

    def __getitem__(self, idx):
        # 输入窗口 [idx : idx+window_size]
        x = self.X[idx : idx + self.window_size]
        # 输出 horizon 步标签 [idx+window_size : idx+window_size+horizon]
        y = self.y[idx + self.window_size : idx + self.window_size + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)

# 2. 模型定义
class Seq2SeqClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, horizon=96, num_classes=3):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 双向 -> hidden_dim * 2
        self.classifier = nn.Linear(hidden_dim*2, num_classes * horizon)

    def forward(self, x):
        # x: [batch, window, feat_dim]
        out, _ = self.lstm(x)            # out: [batch, window, hidden_dim*2]
        # 只取最后一个时间步的隐藏状态
        h_last = out[:, -1, :]           # [batch, hidden_dim*2]
        logits = self.classifier(h_last) # [batch, num_classes*horizon]
        # reshape成 [batch, horizon, num_classes]
        logits = logits.view(-1, self.horizon, 3)
        return logits

# 3. 训练/验证循环示例
def train():
    # 超参
    csv_path = 'data_new.csv'
    window_size = 96
    horizon = 96
    batch_size = 32
    lr = 1e-3
    epochs = 10

    # 数据
    dataset = TimeSeriesMultiStepDataset(csv_path, window_size, horizon)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 模型
    feat_dim = dataset.X.shape[1]
    model = Seq2SeqClassifier(feat_dim, hidden_dim=64, num_layers=1,
                              horizon=horizon, num_classes=3)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # 会把最后维度作为类别

    # 训练
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            # x_batch: [B, window, feat_dim]
            # y_batch: [B, horizon]
            logits = model(x_batch)  # [B, horizon, 3]
            # CrossEntropyLoss 要求输入 [B*horizon, C]，目标 [B*horizon]
            loss = criterion(logits.view(-1, 3), y_batch.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

if __name__ == "__main__":
    train()
