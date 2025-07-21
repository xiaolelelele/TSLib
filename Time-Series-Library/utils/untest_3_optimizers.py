import numpy as np
from bayes_opt import BayesianOptimization
import optuna
# SMAC v2.x 新导入方式
from smac import Scenario
from smac.facade import BlackBoxFacade  # 替换 SMAC4HPO
from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace, UniformFloatHyperparameter
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import os
from datetime import datetime

class HyperOptimizer:
    def __init__(self, args, exp_instance):
        self.args = args
        self.exp_instance = exp_instance
        self.params_to_optimize = args.optim_params.split(',')
        self.best_params = None
        self.best_score = float('inf')
        
        # 修改参数空间定义
        self.param_space = {
            'train_epochs': [10.0, 100.0],  # 使用浮点数
            'batch_size': [4.0, 64.0],      # 使用浮点数
            'alpha': [0.0, 1.0]             # 使用浮点数
        }
        
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
        for param_name, value in params.items():
            if param_name not in self.param_space:
                raise ValueError(f"未知参数: {param_name}")
            if param_name == 'batch_size':
                if value < min(self.param_space[param_name]) or value > max(self.param_space[param_name]):
                    return False
            else:
                if value < self.param_space[param_name][0] or value > self.param_space[param_name][1]:
                    return False
        return True

    def log_trial(self, params, score):
        """记录每次试验的结果"""
        with open(self.log_path, 'a') as f:
            f.write(f"参数: {params}, 分数: {score}\n")

    def objective(self, **params):
        try:
            if not self.validate_params(params):
                print(f"参数验证失败: {params}")
                return float('inf')
            
            # 更新参数前先进行类型转换
            processed_params = {}
            for param_name, param_value in params.items():
                if param_name == 'batch_size':
                    # 确保batch_size为整数
                    power = round(np.log2(float(param_value)))
                    processed_params[param_name] = int(2 ** power)
                else:
                    processed_params[param_name] = float(param_value)
                    
            # 使用处理后的参数更新
            for param_name, param_value in processed_params.items():
                setattr(self.args, param_name, param_value)
            
            # 添加调试信息
            print(f"当前试验参数: {processed_params}")
                
            # 训练和评估
            setting = f'optim_trial_{np.random.randint(10000)}'
            model = self.exp_instance.train(setting)
            
            # 验证性能
            train_loader = self.exp_instance._get_data(flag='train')[1]
            vali_loader = self.exp_instance._get_data(flag='val')[1]
            criterion = self.exp_instance._select_criterion(self.args.loss)
            score = self.exp_instance.vali(train_loader, vali_loader, criterion)
            
            if score < self.best_score:
                self.best_score = score
                self.best_params = params
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            self.log_trial(params, score)
            
            # 早停检查
            if self.patience_counter >= self.patience:
                print("触发早停机制")
                return float('inf')
                
            return -score
        except Exception as e:
            print(f"优化过程出错: {str(e)}")
            return float('inf')
    
    def bayesian_optimize(self):
        try:
            # 设置较大的初始随机采样次数
            optimizer = BayesianOptimization(
                f=lambda **params: self.objective(**params),
                pbounds={name: self.param_space[name] for name in self.params_to_optimize},
                random_state=1,
                verbose=2  # 增加详细程度
            )
            
            # 添加初始随机探索
            optimizer.maximize(
                init_points=5,  # 初始随机点数
                n_iter=self.args.n_trials - 5  # 减去初始点数
            )
            
            if optimizer.max is not None:
                self.best_params = {k: float(v) for k, v in optimizer.max['params'].items()}
                self.best_score = float(-optimizer.max['target'])
            else:
                self.best_params = {}
                self.best_score = float('inf')
                
            print(f"优化完成 - 最佳参数: {self.best_params}, 最佳分数: {self.best_score}")
            return self.best_params, self.best_score
        except Exception as e:
            print(f"贝叶斯优化出错: {e}")
            return {}, float('inf')
    
    def tpe_optimize(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(
            **{name: trial.suggest_float(name, *self.param_space[name])
               if name != 'batch_size' else 
               trial.suggest_categorical(name, self.param_space[name])
               for name in self.params_to_optimize}
        ), n_trials=self.args.n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        return self.best_params, self.best_score
    
    def _get_configspace(self):
        """创建SMAC配置空间（v2.x兼容）"""
        cs = ConfigurationSpace()
        
        for param_name in self.params_to_optimize:
            if param_name == 'batch_size':
                param = CategoricalHyperparameter(
                    param_name, choices=self.param_space[param_name]
                )
            else:
                param = UniformFloatHyperparameter(
                    param_name,
                    lower=self.param_space[param_name][0],
                    upper=self.param_space[param_name][1]
                )
            cs.add_hyperparameter(param)
        return cs

    def optimize(self):
        try:
            if self.args.optim_method == 'bayesian':
                self.best_params, self.best_score = self.bayesian_optimize()
            elif self.args.optim_method == 'tpe':
                self.best_params, self.best_score = self.tpe_optimize()
            elif self.args.optim_method == 'smac':
                # =========== SMAC v2.3.0 修改部分 ===========
                cs = self._get_configspace()
                scenario = Scenario(
                    configspace=cs,
                    output_directory=self.log_dir,
                    deterministic=True,
                    n_trials=self.args.n_trials,  # 替代 runcount-limit
                    objectives="quality"         # 替代 run_obj
                )
                smac = BlackBoxFacade(
                    scenario=scenario,
                    target_function=self.objective,
                    overwrite=True,
                    logging_level=3
                )
                result = smac.optimize()
                self.best_params = result  # 需要适配SMAC的返回值
                self.best_score = None  # 需要根据实际情况计算
                # 移除 smac.save() 调用，因为 BlackBoxFacade 不需要手动保存
                # =========== 修改结束 ===========
            elif self.args.optim_method == 'bohb':
                # ... [BOHB部分保持不变] ...
                pass
            
            self.save_results()
            return self.best_params, self.best_score
        except Exception as e:
            print(f"优化过程失败: {str(e)}")
            return {}, float('inf')

    def save_results(self):
        """保存最优结果"""
        with open(self.best_params_path, 'w') as f:
            f.write(f"最佳参数: {self.best_params}\n")
            f.write(f"最佳分数: {self.best_score}\n")