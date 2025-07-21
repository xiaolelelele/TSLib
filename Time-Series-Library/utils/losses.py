# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class AlphaMixLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, clip_value: float = 50.0, eps: float = 1e-8):
        super(AlphaMixLoss, self).__init__()
        self.alpha = alpha
        self.clip_value = clip_value  # 梯度截断阈值
        self.eps = eps

    def safe_log_cosh(self, diff: t.Tensor) -> t.Tensor:
        """
        数值稳定的 log_cosh 实现：
        - 当 |diff| > clip_value 时，近似为 |diff| - log(2)
        - 否则精确计算 log(cosh(diff))
        """
        abs_diff = t.abs(diff)
        # 分段计算
        linear_region = abs_diff > self.clip_value
        safe_diff = t.where(
            linear_region,
            abs_diff - np.log(2.0),  # log(cosh(x)) ≈ |x| - log(2) 当x→±∞
            t.log(t.cosh(t.clamp(diff, -self.clip_value, self.clip_value)))  # 小值时精确计算
        )
        return safe_diff

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.Tensor:
        # 确保所有张量在相同设备
        forecast = forecast.to(target.device)
        mask = mask.to(target.device)
        
        # 应用 Mask
        valid_forecast = forecast * mask
        valid_target = target * mask
        diff = valid_forecast - valid_target
        
        # 计算加权 MSE
        mse_numerator = t.sum(diff ** 2)
        mse_denominator = t.sum(mask) + self.eps
        mse_loss = mse_numerator / mse_denominator
        
        # 计算稳定的 Log-Cosh
        log_cosh = self.safe_log_cosh(diff)
        log_cosh_loss = t.sum(log_cosh * mask) / mse_denominator  # 与 MSE 共享分母
        
        # 混合损失
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * log_cosh_loss
        
        # 数值安全检查
        if t.isnan(total_loss) or t.isinf(total_loss):
            raise RuntimeError(f"损失值异常: mse={mse_loss.item()}, log_cosh={log_cosh_loss.item()}")
        
        return total_loss