from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Classify_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classify_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim


    def compute_loss(self,trend_probs, trend_true, mag_pred, mag_true,y_pred, y_true,
                    alpha=1.0, beta=1.0):
        """
        trend_probs: (B, H, 3) softmax 输出
        trend_true: (B, H) 真实分类标签 ∈ {0,1,2}
        mag_pred: (B, H) 幅度预测值
        mag_true: (B, H) 幅度真实值
        pred_seq: (B, H) 最终预测值（递推后）
        y_true: (B, H) 真实 y 序列
        """
        B, H = trend_probs.shape[:2]

        # 趋势分类交叉熵损失
        trend_loss = F.cross_entropy(
            trend_probs.view(B * H, 3),
            trend_true.view(B * H)
        )

        # 幅度回归损失（可选 MSE / MAE） 
        mag_loss = F.mse_loss(mag_pred, mag_true)

        # 最终预测序列损失
        pred_loss = F.mse_loss(y_pred, y_true)

        total_loss = pred_loss + alpha * mag_loss + beta * trend_loss
        return total_loss, trend_loss, mag_loss

    def _select_criterion(self):
        # 使用自定义多任务损失：
        criterion = self.compute_loss
        
        return criterion
    
    def inverse_y(self,data, y0,trend_probs, mag_pred):
        """

        """
        # 直接选择softmax最大值的类别标签作为系数
        idx = torch.argmax(trend_probs, dim=-1)        # (B, H)
        labels = torch.tensor([-1.0, 1.0, 1.0], device=trend_probs.device, dtype=mag_pred.dtype)
        coef = labels[idx]                             # (B, H)
    
        if data.scale:
            delta_y = data.inverse_delta_y_transform(mag_pred)
            y0 = data.inverse_y_transform(y0)

        else:
            delta_y = mag_pred
            y0 = y0
        delta = coef * delta_y                        # (B, H)    
        y_pred =  torch.cumsum(delta, dim=-1) + y0.unsqueeze(-1)
        if data.scale:
            y_pred = data.y_transform(y_pred)  # 对y_pred进行归一化
        
        return  y_pred
 

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (x, y0, y_true, trend_true, mag_true) in enumerate(vali_loader):
                x, y0 = x.float().to(self.device), y0.float().to(self.device)
                y_true = y_true.float().to(self.device)
                trend_true = trend_true.to(self.device)
                mag_true = mag_true.float().to(self.device)

                trend_probs, mag_pred = self.model(x, y0)
                y_pred = self.inverse_y(vali_data, y0,trend_probs, mag_pred)
                loss, _, _ = criterion(trend_probs, trend_true, mag_pred, mag_true, y_pred,y_true)
                total_loss.append(loss.item())

        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (x, y0, y_true, trend_true, mag_true) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                x, y0 = x.float().to(self.device), y0.float().to(self.device)
                y_true = y_true.float().to(self.device)
                trend_true = trend_true.to(self.device)
                mag_true = mag_true.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        trend_probs, mag_pred = self.model(x, y0)
                        loss, _, _ = criterion(trend_probs, trend_true, mag_pred, mag_true, y_true)
                        train_loss.append(loss.item())
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    trend_probs, mag_pred = self.model(x, y0)
                    y_pred = self.inverse_y(train_data, y0,trend_probs, mag_pred)
                    loss, _, _ = criterion(trend_probs, trend_true, mag_pred, mag_true, y_pred,y_true)
                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time:.2f}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
    
    # 需要调整
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        trend_probs = []
        trend_trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (x, y0, y_true, trend_true, mag_true) in enumerate(test_loader):   
                x, y0 = x.float().to(self.device), y0.float().to(self.device)
                y_true = y_true.float().to(self.device)
                # (B, H)
                trend_prob, mag_pred = self.model(x, y0)             

                batch_y = y_true
                

                f_dim = -1 if self.args.features == 'MS' else 0
                # batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                batch_y = batch_y.detach().cpu().numpy()
                trend_prob = trend_prob.detach().cpu().numpy()
                mag_pred = mag_pred.detach().cpu().numpy()
                y0 = y0.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    # outputs = test_data.inverse_y_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    delta_y = test_data.inverse_delta_y_transform(mag_pred)
                    batch_y = test_data.inverse_y_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    y0 = test_data.inverse_y_transform(y0)
                    print("delta_y",delta_y)
                idx = np.argmax(trend_prob, axis=-1)
                labels = np.array([-1.0, 0.0, 1.0], dtype=np.float32)  # -1:下降，0:平稳，1:上升
                coef = labels[idx]
                delta_y = coef*delta_y
                pred = np.cumsum(delta_y, axis=-1) + np.expand_dims(y0, -1)
                true = batch_y

                preds.append(pred)
                trues.append(true)
                trend_probs.append(trend_prob)
                trend_trues.append(trend_true.detach().cpu().numpy())
                if i % 20 == 0:
                    input = x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        input = input[:, :, -1].reshape(shape[0], -1)                    
                    gt = np.concatenate((input[0, :], true[0, :]), axis=0)
                    pd = np.concatenate((input[0, :], pred[0, :]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        trend_probs = np.concatenate(trend_probs, axis=0)
        # trend_true: (N, H) 真实标签，trend_probs: (N, H, 3) 概率
        trend_trues = np.concatenate(trend_trues, axis=0)  # (N, H)
        trend_preds = np.argmax(trend_probs, axis=-1)  # (N, H)

        # 计算准确率
        acc = (trend_preds == trend_trues).mean()

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        print("preds, trues", preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{},classify_acc:{}'.format(mse, mae, dtw, acc))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{},classify_acc:{}'.format(mse, mae, dtw, acc))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
