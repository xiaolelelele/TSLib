import numpy as np
import optuna
import os
from datetime import datetime

class HyperOptimizer:
    def __init__(self, args, exp_instance):
        self.args = args
        self.exp_instance = exp_instance
        self.params_to_optimize = args.optim_params.split(',')
        self.best_params = None
        self.best_score = float('inf')
        
        # 创建logs目录
        self.log_dir = os.path.join('/root/autodl-tmp/TSLib/logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 使用时间戳命名日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(self.log_dir, f'optim_logs_{timestamp}.txt')
        self.best_params_path = os.path.join(self.log_dir, f'best_params_{timestamp}.txt')
        
        self.patience = 5  # 早停耐心值
        self.patience_counter = 0

    def validate_params(self, params):
        """验证参数是否在有效范围内"""
        valid = True
        if 'train_epochs' in params:
            valid &= isinstance(params['train_epochs'], int)
            valid &= 10 <= params['train_epochs'] <= 100
        if 'alpha' in params:
            valid &= 0.0 <= params['alpha'] <= 1.0
        if 'batch_size' in params:
            valid &= params['batch_size'] in [4, 8, 16, 32, 64]
        return valid

    def log_trial(self, params, score):
        """记录每次试验的结果"""
        with open(self.log_path, 'a') as f:
            f.write(f"参数: {params}, 分数: {score}\n")

    def objective(self, trial):
        try:
            # 使用Optuna trial直接获取所有参数
            params = {
                'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64]),
                'train_epochs': trial.suggest_int('train_epochs', 10, 100),  # 直接获取整数
                'alpha': trial.suggest_float('alpha', 0.0, 1.0)
            }
            
            if not self.validate_params(params):
                print(f"参数验证失败: {params}")
                return float('inf')
            
            # 强制类型转换确保安全
            params['train_epochs'] = int(params['train_epochs'])
            
            # 更新模型参数
            for param_name, param_value in params.items():
                setattr(self.args, param_name, param_value)
                
            print(f"当前试验参数: {params}")
                
            # 训练和评估
            # setting与原始setting不同
            setting = f'optim_trial_{trial.number}'
            model = self.exp_instance.train(setting)
            
            # 验证性能
            train_loader = self.exp_instance._get_data(flag='train')[1]
            vali_loader = self.exp_instance._get_data(flag='val')[1]
            criterion = self.exp_instance._select_criterion(self.args.loss)
            score = self.exp_instance.vali(train_loader, vali_loader, criterion)
            
            # 更新最佳结果
            if score < self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            self.log_trial(params, score)
            
            # 早停检查
            if self.patience_counter >= self.patience:
                print("触发早停机制")
                trial.study.stop()
                
            return score
            
        except Exception as e:
            print(f"优化过程出错: {str(e)}")
            return float('inf')
    
    def optimize(self):
        try:
            study = optuna.create_study(direction='minimize')
            study.optimize(self.objective, n_trials=self.args.n_trials)

            if study.best_trial:
                self.best_params = study.best_params
                self.best_score = study.best_value
                # 确保train_epochs是整数类型
                if 'train_epochs' in self.best_params:
                    self.best_params['train_epochs'] = int(self.best_params['train_epochs'])
                
            return self.best_params, self.best_score

        except Exception as e:
            print(f"优化过程失败: {str(e)}")
            return {}, float('inf')

    def save_results(self):
        """保存最优结果"""
        with open(self.best_params_path, 'w') as f:
            f.write(f"最佳参数: {self.best_params}\n")
            f.write(f"最佳分数: {self.best_score}\n")