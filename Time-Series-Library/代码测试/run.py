import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import random
import os

# 1. 设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# 2. 数据集定义
class TimeSeriesMultiStepDataset(Dataset):
    def __init__(self, csv_path, window_size=128, horizon=96):
        df = pd.read_csv(csv_path, parse_dates=True)
        feats = df.drop(columns=['date', 'trend'], errors='ignore')
        labels = df['trend'].values.astype('int64')
        self.X = feats.values.astype('float32')
        self.y = labels
        self.window_size = window_size
        self.horizon = horizon
        self.total_length = len(self.X)

    def __len__(self):
        return self.total_length - self.horizon - self.window_size + 1

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.window_size]
        y = self.y[idx + self.window_size : idx + self.window_size + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)

# 3. 模型定义
class Seq2SeqClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=1, horizon=96, num_classes=3):
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

# 4. 训练、验证和测试函数
def train_validate_test():
    csv_path = '代码测试/weather_trend.csv'
    window_size = 96
    horizon = 96
    batch_size = 512
    lr = 1e-3
    epochs = 10
    val_ratio = 0.1
    test_ratio = 0.2

    # 创建完整数据集
    full_dataset = TimeSeriesMultiStepDataset(csv_path, window_size, horizon)
    
    # 按时间顺序分割数据集
    total_indices = len(full_dataset)
    train_end = int(total_indices * (1 - val_ratio - test_ratio))
    val_end = int(total_indices * (1 - test_ratio))
    
    train_indices = range(0, train_end)
    val_indices = range(train_end, val_end)
    test_indices = range(val_end-96+153, total_indices)
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                             drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False, pin_memory=True)

    # 初始化模型
    feat_dim = full_dataset.X.shape[1]
    model = Seq2SeqClassifier(feat_dim, hidden_dim=512, num_layers=3,
                              horizon=horizon, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 记录训练过程
    train_losses = []
    val_losses = []
    test_losses = []  # 测试损失将在训练结束后计算
    best_val_loss = float('inf')

    # for epoch in range(epochs):
    #     # ===== 训练阶段 =====
    #     model.train()
    #     total_train_loss = 0.0
    #     total_train_samples = 0

    #     for x_batch, y_batch in train_loader:
    #         x_batch = x_batch.to(device, non_blocking=True)
    #         y_batch = y_batch.to(device, non_blocking=True)

    #         logits = model(x_batch)
    #         loss = criterion(logits.view(-1, 3), y_batch.view(-1))
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         total_train_loss += loss.item() * x_batch.size(0)
    #         total_train_samples += x_batch.size(0)

    #     avg_train_loss = total_train_loss / total_train_samples
    #     train_losses.append(avg_train_loss)

    #     # ===== 验证阶段 =====
    #     model.eval()
    #     total_val_loss = 0.0
    #     total_val_samples = 0
        
    #     with torch.no_grad():
    #         for x_val, y_val in val_loader:
    #             x_val = x_val.to(device, non_blocking=True)
    #             y_val = y_val.to(device, non_blocking=True)
                
    #             val_logits = model(x_val)
    #             loss = criterion(val_logits.view(-1, 3), y_val.view(-1))
    #             total_val_loss += loss.item() * x_val.size(0)
    #             total_val_samples += x_val.size(0)
        
    #     avg_val_loss = total_val_loss / total_val_samples
    #     val_losses.append(avg_val_loss)
        
    #     # 保存最佳模型
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         torch.save(model.state_dict(), './best_model.pth')
    #         print(f"New best model saved with val loss: {best_val_loss:.4f}")

    #     print(f"Epoch {epoch+1}/{epochs} — Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # ===== 测试阶段 =====
    model.load_state_dict(torch.load('代码测试/best_model.pth'))
    model.eval()
    total_test_loss = 0.0
    total_test_samples = 0
    test_predictions = []
    test_targets = []
    test_trues = []
    test_preds = []
    
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device, non_blocking=True)
            y_test = y_test.to(device, non_blocking=True)
            
            test_logits = model(x_test)
            loss = criterion(test_logits.view(-1, 3), y_test.view(-1))
            total_test_loss += loss.item() * x_test.size(0)
            total_test_samples += x_test.size(0)
            
            # 保存第一个batch的一个例子用于可视化
            if len(test_predictions) == 0:
                test_predictions = test_logits.argmax(dim=2).cpu()
                test_targets = y_test.cpu()
            test_trues.append(y_test.cpu())
            test_preds.append(test_logits.argmax(dim=2).cpu())
    
    avg_test_loss = total_test_loss / total_test_samples
    test_losses = [avg_test_loss] * epochs  # 为绘图创建等长列表
    test_preds = torch.cat(test_preds, dim=0)
    test_trues = torch.cat(test_trues, dim=0)
    acc = (test_preds.flatten() == test_trues.flatten()).float().mean().item()
    print(test_preds.shape, test_trues.shape)
    print(f"Test Accuracy: {acc:.4%}")
    print('predictions:', test_preds.flatten()[:30], 'trues:', test_trues.flatten()[:30])
    print('\npredictions:', test_preds.flatten()[-30:], 'trues:', test_trues.flatten()[-30:])
    # 计算准确率
    test_acc = (test_predictions == test_targets).float().mean().item()
    print(f"\nsingle Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.4%}")
    
    # ===== 可视化损失曲线 =====
    plt.figure(figsize=(15, 5))
    
    # 1. 训练、验证和测试损失曲线
    plt.subplot(1, 2, 1)
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'g-', label='Validation Loss')
    plt.plot(epochs_range, test_losses, 'r--', label='Test Loss (constant)')
    plt.scatter(epochs, avg_test_loss, color='red', s=100, zorder=5, 
                label=f'Final Test Loss: {avg_test_loss:.4f}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. 训练和验证损失曲线（更详细视图）
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'g-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Detail')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./loss_curves.png')
    plt.show()
    
    # ===== 可视化预测结果 =====
    def transform(arr):
        return np.where(arr == 0, 1,
                        np.where(arr == 1, 0, -1))
    
    steps = np.arange(1, horizon + 1)
    cum_preds = transform(test_predictions.numpy()[0]).cumsum()
    cum_truth = transform(test_targets.numpy()[0]).cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(steps, cum_truth, label='Cumulative True Labels', marker='o')
    plt.plot(steps, cum_preds, label='Cumulative Predicted Labels', marker='x')
    plt.xlabel('Future Step')
    plt.ylabel('Cumulative Sum of Labels')
    plt.title('Test Set: Cumulative Sum of True vs Predicted Labels')
    plt.legend()
    plt.grid(True)
    plt.savefig('./cumulative_prediction.png')
    plt.show()

if __name__ == "__main__":
    train_validate_test()