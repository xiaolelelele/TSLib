import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 1. 设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 数据集定义
class TimeSeriesMultiStepDataset(Dataset):
    def __init__(self, csv_path, window_size=128, horizon=96):
        df = pd.read_csv(csv_path, parse_dates=True)
        feats = df.drop(columns=['date', 'OT', 'delta'], errors='ignore')
        labels = df['delta'].values.astype('int64')
        self.X = feats.values.astype('float32')
        self.y = labels
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.horizon - self.window_size + 1

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.window_size]
        y = self.y[idx + self.window_size : idx + self.window_size + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)

# 3. 模型定义
class Seq2SeqClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, horizon=96, num_classes=3):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes * horizon)

    def forward(self, x):
        out, _ = self.lstm(x)             # [batch, window, hidden*2]
        h_last = out[:, -1, :]            # [batch, hidden*2]
        logits = self.classifier(h_last)  # [batch, num_classes*horizon]
        logits = logits.view(-1, self.horizon, 3)
        return logits                     # [batch, horizon, num_classes]

# 4. 训练循环（含准确度计算）
def train():
    csv_path = '代码测试/data_new.csv'
    window_size = 96
    horizon = 96
    batch_size = 512
    lr = 1e-3
    epochs = 200

    dataset = TimeSeriesMultiStepDataset(csv_path, window_size, horizon)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, pin_memory=True)

    feat_dim = dataset.X.shape[1]
    model = Seq2SeqClassifier(feat_dim, hidden_dim=512, num_layers=3,
                              horizon=horizon, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in loader:
            # 移动到 GPU/CPU
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(x_batch)            # [B, horizon, 3]
            B = logits.size(0)

            # 计算损失
            loss = criterion(logits.view(-1, 3), y_batch.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 计算准确度
            preds = logits.argmax(dim=2)       # [B, horizon]

            correct = (preds == y_batch).sum().item()
            total_correct += correct
            total_samples += B * horizon


        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4%}")

    # 训练完毕后的最终准确度
    def transform(arr):
        return np.where(arr == 0, 1,
                        np.where(arr == 1, 0, -1))
    steps = np.arange(1, horizon + 1)
    cum_preds = transform(preds.cpu().numpy()[0]).cumsum()
    cum_truth = transform(y_batch.cpu().numpy()[0]).cumsum()

    # 6. 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(steps, cum_truth, label='Cumulative True Labels', marker='o')
    plt.plot(steps, cum_preds, label='Cumulative Predicted Labels', marker='x')
    plt.xlabel('Future Step')
    plt.ylabel('Cumulative Sum of Labels')
    plt.title('Last Batch: Cumulative Sum of True vs Predicted Labels')
    plt.legend()
    plt.show()
    print(f"Final Accuracy over all epochs: {accuracy:.4%}")

if __name__ == "__main__":
    train()
