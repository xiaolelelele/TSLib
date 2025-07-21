import numpy as np
import optuna
import os
from datetime import datetime
from bayes_opt import BayesianOptimization

class HyperOptimizer:
    def __init__(self, n_trials, Logger=None,train_test=None):
        self.n_trials = n_trials
        self.train_test = train_test
        self.best_params = None
        self.best_score = float('inf')
        
        # 创建logs目录
        self.log_dir = os.path.join('/lxh/logs')
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
            
            score = self.train_test(delta = params['alpha'],hidden = 0,batch = params['batch_size'],epoch = params['train_epochs'])
            
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
            study.optimize(self.objective, n_trials=self.n_trials)

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


class Train_OPT:
    """贝叶斯优化训练类"""
    def __init__(self, init_points: int, n_iter: int, Logger=None,train_test=None):

        self.init_points = init_points
        self.n_iter = n_iter
        self.logger = Logger()
        self.train_test = train_test

    def BOA(self) -> None:
        """执行贝叶斯优化"""
        pbounds = {
            'delta': [0, 1],
            'hidden': (3, 32),
            'epoch': (300, 600),
            'batch': (4, 64),
        }

        optimizer = BayesianOptimization(
            f=self.train_test,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )
        optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
        )
        
        # 记录最优结果
        best_metrics = {
            'best_score': float(optimizer.max['target']),
            'best_params_delta': float(optimizer.max['params']['delta']),
            'best_params_hidden': float(optimizer.max['params']['hidden']),
            'best_params_epoch': float(optimizer.max['params']['epoch']),
            'best_params_batch': float(optimizer.max['params']['batch'])
        }
        self.logger.log_metrics(best_metrics)
        self.logger.save_final_results()
        print(optimizer.max)