from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,calculate_trend_agreement
from utils.trend_loss import TrendLoss
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from models.ValueAndTrendModel import ValueAndTrendModel

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_Class(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Class, self).__init__(args)
        self.alpha =  args.trend_alpha  # 超参数alpha
        # self.TrendModel = ValueAndTrendModel().to(self.device)
        # self.trend_loss_fn = TrendLoss(tau=0.05, alpha=args.trend_alpha, warmup_epoch=0)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # # mlp映射，参数需要调整
        # model_optim = optim.Adam(
        #     list(self.model.parameters())+list(self.TrendModel.parameters()), lr=self.args.learning_rate
        #     )
        # 直接映射 
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
            )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    

    
    def compute_trend_loss(self,y_pred, y_true):
        """
        y_pred差分直接映射logit  构造真实趋势标签
        Compute the trend loss between y_pred and y_true.

        Args:
            y_pred (torch.Tensor): Predicted sequence, shape (batch_size, seq_len, num_features)
            y_true (torch.Tensor): Ground truth sequence, shape (batch_size, seq_len, num_features)

        Returns:
            torch.Tensor: Scalar loss value
        """
        delta_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        delta_true = y_true[:, 1:, :] - y_true[:, :-1, :]

        # === 标准化差分值 ===
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
    
    def compute_trend_loss2(self,y_pred, y_true):
        loss =self.TrendModel(y_pred, y_true)
        return loss
    
    def compute_trend_loss3(self,logits, y_true):
        # logits由nn给出 构造真实趋势标签
        delta_true = y_true[:, 1:, :] - y_true[:, :-1, :] # 计算真实值的差分 b,pred-1,d
        trend_true = torch.where(delta_true < 0, 0,
            torch.where(delta_true == 0, 1, 2)).long()
        crossentropy_loss = nn.CrossEntropyLoss()
        loss = crossentropy_loss(logits.reshape(-1, 3), trend_true.reshape(-1))

        return loss
    
    def compute_mse_delta_loss(self,y_pred, y_true, eps=1e-6,is_test= False):
        """
        计算预测值和真实值的差分的均方误差
        Args:
            y_pred (torch.Tensor): 预测值，形状为 (batch_size, seq_len, num_features)
            y_true (torch.Tensor): 真实值，形状为 (batch_size, seq_len, num_features)
        Returns:
            torch.Tensor: 差分均方误差损失
        """
        # # pred基于y_true计算差分本质与原mse相同
        delta_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        delta_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        # # 计算 delta_true 的均值和标准差（按 batch 和时间维度）
        # mean = delta_true.mean(dim=(0, 1), keepdim=True)
        # std = delta_true.std(dim=(0, 1), keepdim=True) + eps  # 避免除以0
        ### 有的序列过于平稳，使用归一化，会导致std过小，从而mse过大
        # # 用 delta_true 的统计量对两个差分序列做标准化
        # delta_pred_norm = (delta_pred - mean) / std
        # delta_true_norm = (delta_true - mean) / std
        mse_delta_loss = nn.MSELoss('mean')
        total_loss = mse_delta_loss(y_pred, y_true)
        loss = mse_delta_loss(delta_pred, delta_true)
        # loss = mse_delta_loss(delta_pred, delta_true)
        # # 多级反馈loss
        # for i in range(1,y_pred.shape[1]):
        #     delta_pred = y_pred[:, i:, :] - y_pred[:, :-i, :] # i ！= 0
        #     delta_true = y_true[:, i:, :] - y_true[:, :-i, :]
        #     loss = mse_delta_loss(delta_pred, delta_true)
        #     # 加上衰减系数
        #     # total_loss += 0.9 **i * loss 
        #     # total_loss += 1 / (i + 1)*loss
        #     total_loss += (1 - (i / 96))*loss
        return loss


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_mse_delta_loss = []
        total_trend_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # trend_loss = self.compute_trend_loss3(logits, batch_y)
                # trend_loss = self.trend_loss_fn(
                #         logits=logits,
                #         y_pred=outputs,
                #         y_true=batch_y,
                #         epoch=999  # 999表示不使用warmup
                #         ,means=means, stdev=stdev
                #     )
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                # 只使用mse来作验证
                loss = criterion(pred, true)
                mse_delta_loss = self.compute_mse_delta_loss(outputs, batch_y)
                trend_loss = self.compute_trend_loss(outputs, batch_y)
                total_loss.append(loss)
                total_mse_delta_loss.append(mse_delta_loss)
                total_trend_loss.append(trend_loss)
        total_loss = np.average([l.detach().cpu().item() for l in total_loss])
        total_mse_delta_loss = np.average([l.detach().cpu().item() for l in total_mse_delta_loss])
        total_trend_loss = np.average([l.detach().cpu().item() for l in total_trend_loss])
        self.model.train()
        return total_loss,total_mse_delta_loss,total_trend_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_loss_list = []
        vali_loss_list = []
        test_loss_list = []

        train_mse_delta_loss_list = []
        vali_mse_delta_loss_list = []
        test_mse_delta_loss_list = []
        train_trend_loss_list = []
        vali_trend_loss_list = []
        test_trend_loss_list = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_trend_loss = []
            train_mse_delta_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # loss = criterion(outputs, batch_y)+ self.alpha *self.compute_trend_loss2(outputs, batch_y)
                        # loss = criterion(outputs, batch_y)+ self.alpha *self.compute_trend_loss3(logits, batch_y)
                        mse_delta_loss = self.compute_mse_delta_loss(outputs, batch_y)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # 超参数alpha 
                    # loss = criterion(outputs, batch_y)+ self.alpha *self.compute_trend_loss2(outputs, batch_y)
                    # trend_loss = self.compute_trend_loss3(logits, batch_y)
                    # loss = criterion(outputs, batch_y)+ self.alpha *trend_loss
                    # print("loss:", loss.item())
                    # print("criterion(outputs, batch_y)",criterion(outputs, batch_y))
                    # print("self.compute_trend_loss(outputs, batch_y)",self.compute_trend_loss(outputs, batch_y))
                    # trend_loss = self.trend_loss_fn(
                    #     logits=logits,
                    #     y_pred=outputs,
                    #     y_true=batch_y,
                    #     epoch=epoch,
                    #     means=means, stdev= stdev,
                    # )
                    mse_delta_loss = self.compute_mse_delta_loss(outputs, batch_y)
                    mse_loss = criterion(outputs, batch_y)
                    trendloss = self.compute_trend_loss(outputs, batch_y)
                    # loss = 0.01*mse_loss  + 0.99 * mse_delta_loss  #  + 0.01*trendloss
                    loss_feq = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs().mean() 
                    # loss =0.01*mse_loss+ 0.9*mse_delta_loss + 0.09*loss_feq
                    loss = (1-self.alpha)*mse_loss  + self.alpha* mse_delta_loss
                    # loss = mse_loss
                    train_loss.append(mse_loss.item())
                    train_mse_delta_loss.append(mse_delta_loss.item())
                    train_trend_loss.append(trendloss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}| mse_delta_loss: {3:.7f}|trendloss:{4:.7f}|loss_feq:{5:.7f}".format(i + 1, epoch + 1, loss.item(),mse_delta_loss.item(),trendloss.item(),loss_feq.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_mse_delta_loss = np.average(train_mse_delta_loss)
            train_trend_loss = np.average(train_trend_loss)   
            vali_loss,vali_mse_delta_loss,vali_trend_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss,test_mse_delta_loss,test_trend_loss = self.vali(test_data, test_loader, criterion)

            train_loss_list.append(train_loss)
            vali_loss_list.append(vali_loss)
            test_loss_list.append(test_loss)

            train_mse_delta_loss_list.append(train_mse_delta_loss)
            vali_mse_delta_loss_list.append(vali_mse_delta_loss)
            test_mse_delta_loss_list.append(test_mse_delta_loss)

            # train_trend_loss_list.append(train_trend_loss)
            # vali_trend_loss_list.append(vali_trend_loss)
            # test_trend_loss_list.append(test_trend_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}\n" \
            " train_mse_delta_loss: {5:.7f} vali_mse_delta_loss: {6:.7f} test_mse_delta_loss: {7:.7f}\n" \
            "train_trend_loss: {8:.7f} vali_trend_loss: {9:.7f} test_trend_loss: {10:.7f}\n".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, 
                train_mse_delta_loss, vali_mse_delta_loss, test_mse_delta_loss,
                train_trend_loss, vali_trend_loss, test_trend_loss))
            # 先关掉早停机制
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # 实现了lr的调整
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        results_dir = './results/' + setting
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        ## 绘制loss曲线
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(train_loss_list, label='Train Loss')
        plt.plot(vali_loss_list, label='Validation Loss')
        plt.plot(test_loss_list, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training/Validation/Test Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('./results/'+setting+'/loss_curve.png')
        plt.close()

        plt.figure()
        plt.plot(train_mse_delta_loss_list, label='Train Trend Loss')
        plt.plot(vali_mse_delta_loss_list, label='Validation Trend Loss')
        plt.plot(test_mse_delta_loss_list, label='Test Trend Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Trend Classification Loss')
        plt.title('Trend Classification Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('./results/'+setting+'/trend_loss_curve.png')
        plt.close()
        # 保存模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        logits_list = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                # 测试 temp
                # trend_loss = self.trend_loss_fn(
                #         logits=logits,
                #         y_pred=outputs,
                #         y_true=batch_y,
                #         epoch=999  # 999表示不使用warmup
                #         ,means=means, stdev=stdev
                #     )
                # if i == 0:
                #     # temp 训练完成后 查看trend_loss 内部情况
                #     self.compute_trend_loss2(outputs, batch_y)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                
                # 若为M模式，是0：预测所有的特征
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # logits_list.append(logits.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # logits_list = np.concatenate(logits_list, axis=0)
        # print('test shape:', preds.shape, trues.shape,logits_list.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # logits_list = logits_list.reshape(logits_list.shape[0], logits_list.shape[1], -1,3)
        print('test shape:', preds.shape, trues.shape)

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
        # logits_statistic = calculate_trend_agreement(trues, logits_list, is_logits=True)
        trend_statistic = calculate_trend_agreement(trues, preds)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{},trend_statistic:{}'.format(mse, mae, dtw,trend_statistic))
        f = open("result_long_term_forecast_trend_TimesNet.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{},trend_statistic:{}'.format(mse, mae, dtw,trend_statistic))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,trend_statistic]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
