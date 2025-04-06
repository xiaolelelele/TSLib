import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from typing import Tuple, Dict, Any
import math
from datetime import datetime
import json
import os
from sklearn.utils import shuffle
from sklearn.metrics  import r2_score
import argparse
from model.models import GRUModel, TCN, Seq2Seq, TimesNet
from OPT.optimization import Train_OPT, HyperOptimizer

class DataProcessor:
    """数据处理类"""
    def __init__(self, file_path: str, timesteps: int, shuffle_data: bool = True):
        self.timesteps = timesteps
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.data = pd.read_csv(file_path, header=0)
        print(type(self.data))
        #############################
        self.data = self.data[['FCC2_FIQ235_PNT','HLU1_FIC3105A_PV','CDU1_FIQ-3023_PV',"HLU1_FI3118_PV","CDU1_FIQ-3021_PV","HLU1_TIC3129_PV",'DQSH_C5_SL']].values.astype("float32")
        self.is_fitted = False
        self.shuffle_data = shuffle_data
        
    def split_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """根据参数决定是否打乱并按8:2比例分割训练集和测试集"""
        # 首先处理序列数据
        x, y = [], []
        for i in range(self.timesteps, len(self.data) + 1):
            x.append(self.data[i - self.timesteps:i, :-1])
            y.append(self.data[i - 1, -1])
        
        # 将序列数据转换为numpy数组
        x = np.array(x)
        y = np.array(y).reshape(-1, 1)
        
        # 根据参数决定是否打乱数据
        if self.shuffle_data:
            x_processed, y_processed = shuffle(x, y, random_state=42)
        else:
            x_processed, y_processed = x, y
        
        # 组合特征和标签
        combined_data = np.concatenate([
            x_processed.reshape(x_processed.shape[0], -1), 
            y_processed
        ], axis=1)
        
        # 分割数据
        train_size = int(len(combined_data) * 0.8)
        return combined_data[:train_size], combined_data[train_size:]

    def process_train_data(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理训练集数据"""
        # 重塑数据以适应标准化
        features_count = self.data.shape[1] - 1
        x_flat = data[:, :-1].reshape(-1, features_count)
        y = data[:, -1].reshape(-1, 1)
        
        # 对训练集进行拟合和转换
        scaled_x = self.scaler_x.fit_transform(x_flat)
        scaled_y = self.scaler_y.fit_transform(y)
        self.is_fitted = True
        
        # 重塑回序列形式
        scaled_x = scaled_x.reshape(data.shape[0], self.timesteps, features_count)
        
        return (torch.tensor(scaled_x, dtype=torch.float32),
                torch.tensor(scaled_y, dtype=torch.float32))

    def process_test_data(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理测试集数据"""
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted before processing test data")
        
        # 重塑数据以适应标准化
        features_count = self.data.shape[1] - 1
        x_flat = data[:, :-1].reshape(-1, features_count)
        y = data[:, -1].reshape(-1, 1)
        
        # 对测试集只进行转换
        scaled_x = self.scaler_x.transform(x_flat)
        scaled_y = self.scaler_y.transform(y)
        
        # 重塑回序列形式
        scaled_x = scaled_x.reshape(data.shape[0], self.timesteps, features_count)
        
        return (torch.tensor(scaled_x, dtype=torch.float32),
                torch.tensor(scaled_y, dtype=torch.float32))

class Logger:
    """日志记录器"""
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{self.timestamp}.txt")
        self.results = []

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """记录训练参数"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=== Training Parameters ({self.timestamp}) ===\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")

    def log_metrics(self, metrics: Dict[str, Any], iteration: int = None) -> None:
        """记录评估指标"""
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'iteration': iteration,
            'metrics': metrics
        }
        self.results.append(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            if iteration is not None:
                f.write(f"\n=== Iteration {iteration} Results ===\n")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{metric_name}: {value:.4f}\n")
                else:
                    f.write(f"{metric_name}: {value}\n")

    def convert_to_serializable(self, obj):
        """将不可序列化的类型转换为可序列化的类型"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_serializable(item) for item in obj]
        return obj

    def save_final_results(self) -> None:
        """保存最终结果"""
        # 转换结果为可序列化的格式
        serializable_results = [
            {
                'timestamp': result['timestamp'],
                'iteration': result['iteration'],
                'metrics': self.convert_to_serializable(result['metrics'])
            }
            for result in self.results
        ]
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n=== Final Results ===\n")
            f.write(json.dumps(serializable_results, indent=2))

class DeltaMixLoss(nn.Module):
    """混合损失函数：DeltaMix Loss = α·MSE + (1-α)·Log_Cosh"""
    def __init__(self, delta: float = 0.5):
        super(DeltaMixLoss, self).__init__()
        self.delta = delta
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def log_cosh_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """计算Log-Cosh损失"""
        diff = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(diff)))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(y_pred, y_true)
        log_cosh_loss = self.log_cosh_loss(y_pred, y_true)
        mae_loss = self.mae(y_pred, y_true)
        return self.delta * mse_loss + (1 - self.delta) * mae_loss

class ModelTrainer:
    """模型训练器"""
    def __init__(self, device: torch.device, logger: Logger, delta: float = 0.5,criterion: nn.Module = None):
        self.device = device
        self.logger = logger
        self.criterion = criterion
        self.delta = delta

    def train(self, model: TimesNet, train_loader: DataLoader, 
             optimizer: torch.optim.Optimizer, epochs: int) -> None:
        """训练模型"""
        model.to(self.device)
        model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                if isinstance(model, Seq2Seq):
                    outputs = model(batch_x, batch_y)
                else:
                    outputs = model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                avg_loss = epoch_loss / len(train_loader)
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
                self.logger.log_metrics({'train_loss': avg_loss}, epoch + 1)

    def evaluate(self, model: GRUModel, test_x: torch.Tensor, 
                test_y: torch.Tensor, scaler_y: StandardScaler) -> float:
        """评估模型"""
        model.eval()
        test_x, test_y = test_x.to(self.device), test_y.to(self.device)

        with torch.no_grad():
            
            if isinstance(model, Seq2Seq):
                predictions = model(test_x,torch.zeros_like(test_y).to(self.device)).cpu().numpy()
            else:
                predictions = model(test_x).cpu().numpy()
            predictions = scaler_y.inverse_transform(predictions)
            test_y = scaler_y.inverse_transform(test_y.cpu().numpy())

            metrics = {
                'mae': np.mean(np.abs(test_y - predictions)),
                'mape': np.mean(np.abs((test_y - predictions) / test_y)) * 100,
                'mse': mean_squared_error(test_y, predictions),
                'rmse': math.sqrt(mean_squared_error(test_y, predictions)),
                "r2_score": r2_score(test_y, predictions)
            }

            for metric_name, value in metrics.items():
                print(f"Test Score {metric_name.upper()}: {value:.4f}")

            self.logger.log_metrics(metrics)
            return -metrics['mse']



if __name__ == "__main__":
    file_path = r"./C5SL.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='TimesNet')
    # loss
    parser.add_argument('--loss', type=str, default='MixLoss', help='loss function: MSE, MAE, Huber, LogCosh, MixLoss')
    # model
    parser.add_argument('--model', type=str, default='GRU', help='model name: GRU, seq2seq, TCN,Timesnet')
    # optimization
    parser.add_argument('--optimization', type=str, default='TPE', help='optimizer: TPE, BO, Grid')
    
    args = parser.parse_args()



    if args.model == 'GRU':
        model = GRUModel(input_dim=6, hidden_dim=32)
    elif args.model == 'seq2seq':
        model = Seq2Seq(input_size=6,hidden_dim=32,output_size=1)
    elif args.model == 'TCN':  
        model = TCN(input_size=6, output_size=1, num_channels=[32, 64], kernel_size=3, dropout=0.2)
    elif args.model == 'Timesnet':
        model = TimesNet(pred_len=1)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    data_processor = DataProcessor(file_path, timesteps=100, shuffle_data=True)
    # 数据准备
    train_data, test_data = data_processor.split_data()
    train_x, train_y = data_processor.process_train_data(train_data)
    test_x, test_y = data_processor.process_test_data(test_data)
    
    def train_test(delta: float, hidden:float,epoch: float, batch: float) -> float:
        """训练和测试模型"""
        # 记录训练参数
        params = {
            'delta': delta,
            'epochs': int(epoch),
            'batch_size': int(batch),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # loss函数
        if args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif args.loss == 'MAE':
            criterion = nn.L1Loss()
        elif args.loss == 'Huber':
            criterion = nn.SmoothL1Loss()
        elif args.loss == 'LogCosh':
            criterion = nn.SmoothL1Loss()
        elif args.loss == 'MixLoss':
            criterion = DeltaMixLoss(delta=delta)
        else:
            raise ValueError(f"Unsupported loss function: {args.loss}")
        
        # 创建数据加载器
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=int(batch), shuffle=True,num_workers=os.cpu_count()-1,pin_memory=True)
        logger = Logger()
        logger.log_parameters(params)
        trainer = ModelTrainer(device, logger,criterion=criterion)
        
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        # 训练和评估
        trainer.train(model, train_loader, optimizer, int(epoch))
        # 评估模型
        return trainer.evaluate(model, test_x, test_y, data_processor.scaler_y)

    if args.optimization == 'BO':
        # 贝叶斯优化训练
        optimizer = Train_OPT(init_points=5, n_iter=10, timesteps=100, shuffle_data=True,Logger=Logger, train_test=train_test)
        optimizer.BOA()
    elif args.optimization == 'TPE':
        # TPE优化训练
        hyper_optimizer = HyperOptimizer(n_trials=20,Logger=Logger,train_test=train_test)
        best_params = hyper_optimizer.optimize()
        print(f"Best parameters: {best_params}")
    elif args.optimization == 'Grid':
        # 网格优化训练（示例代码，实际实现需要根据具体需求进行调整）
        pass
    else:
        raise ValueError(f"Unsupported optimization method: {args.optimization}")
    

    
    # train_test(delta = 0.5,epoch=1,batch=128,hidden=32)