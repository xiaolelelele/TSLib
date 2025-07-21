import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch.nn as nn

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            # 将优化器中的学习率更新
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def compute_trend_loss(y_pred, y_true,scale):
        """
        直接映射logit作为趋势预测器
        Compute the trend loss between y_pred and y_true.

        Args:
            y_pred (torch.Tensor): Predicted sequence, shape (batch_size, seq_len, num_features)
            y_true (torch.Tensor): Ground truth sequence, shape (batch_size, seq_len, num_features)

        Returns:
            torch.Tensor: Scalar loss value
        """
        delta_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        delta_true = y_true[:, 1:, :] - y_true[:, :-1, :]

        # === 缩放差分值 ===
        # mean = delta_pred.mean(dim=(0, 1), keepdim=True)  # shape (1, 1, D)
        std = delta_pred.std(dim=(0, 1), keepdim=True) + 1e-6  # 防止除以0
        delta_pred = delta_pred / std  # shape (B, T-1, D)

        trend_true = torch.where(delta_true < 0, 0,
                    torch.where(delta_true == 0, 1, 2)).long()
        trend_logits = torch.stack([
            -delta_pred,                          # class 0
            torch.abs(delta_pred)+ 1e-6,        # class 1
            delta_pred                            # class 2
        ], dim=-1)
        trend_logits = trend_logits.view(-1, 3)
        trend_true = trend_true.view(-1)

        loss = nn.CrossEntropyLoss()(trend_logits, trend_true)
        return loss
        
def calculate_trend_agreement(test_data, forecast_data,is_logits=False):
    """
    计算二维数据（b, h）或三维数据（b, h, d）的趋势一致率。

    参数：
    - test_data: ndarray，真实值，shape = (b, h, d)
    - forecast_data: ndarray，预测值，shape = (b, h, d)

    返回：
    - trend_agreement_ratio: float，趋势一致比例
    """
    if is_logits:
        # 如果是logits，直接计算趋势一致率
        forecast_trend = np.argmax(forecast_data, axis=-1)-1

    else:
        assert test_data.shape == forecast_data.shape, "test 和 forecast 的 shape 必须一致"
        # 如果不是numpy数组，则转换为numpy数组
        if not isinstance(test_data, np.ndarray):
            test_data = test_data.to_numpy()
            forecast_data = forecast_data.to_numpy()
        # 计算差分后的趋势：(b, h-1, d)
        
        forecast_trend = np.sign(np.diff(forecast_data, axis=1))
    test_trend = np.sign(np.diff(test_data, axis=1))
    # 趋势匹配：bool mask → (b, h-1, d)
    matches = (test_trend == forecast_trend)

    total_trends = matches.size  # = b × (h-1) × d
    matching_trends = np.sum(matches)

    # print("总趋势数:", total_trends)
    return matching_trends / total_trends

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
