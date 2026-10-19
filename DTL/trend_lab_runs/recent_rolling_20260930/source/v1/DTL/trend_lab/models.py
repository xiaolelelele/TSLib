"""Neural predictors and direction adapters for the trend prototype."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _check_series(x: torch.Tensor, lookback: int, channels: int) -> None:
    if x.ndim != 3 or x.shape[1:] != (lookback, channels):
        raise ValueError(f"expected x shaped [N,{lookback},{channels}]")
    if not torch.is_floating_point(x):
        raise TypeError("x must be floating point")


def _check_base(x: torch.Tensor, base: torch.Tensor, horizon: int, channels: int) -> None:
    if base.shape != (x.shape[0], horizon, channels):
        raise ValueError(f"expected base shaped [N,{horizon},{channels}]")
    if base.device != x.device or base.dtype != x.dtype:
        raise ValueError("x and base must share device and dtype")


class _AnchoredBase(nn.Module):
    def __init__(self, lookback: int, horizon: int, channels: int) -> None:
        super().__init__()
        if min(lookback, horizon, channels) < 1:
            raise ValueError("lookback, horizon, and channels must be positive")
        self.lookback, self.horizon, self.channels = lookback, horizon, channels

    def _anchor(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        _check_series(x, self.lookback, self.channels)
        return x[:, -1:, :] + offsets.reshape(x.shape[0], self.horizon, self.channels)


class _LinearBase(_AnchoredBase):
    def __init__(self, lookback: int, horizon: int, channels: int) -> None:
        super().__init__(lookback, horizon, channels)
        self.proj = nn.Linear(lookback * channels, horizon * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._anchor(x, self.proj(x.reshape(x.shape[0], -1)))


class _MLPBase(_AnchoredBase):
    def __init__(self, lookback: int, horizon: int, channels: int, hidden: int) -> None:
        super().__init__(lookback, horizon, channels)
        self.net = nn.Sequential(nn.Linear(lookback * channels, hidden), nn.ReLU(),
                                 nn.Linear(hidden, horizon * channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._anchor(x, self.net(x.reshape(x.shape[0], -1)))


class _MovingAverage(nn.Module):
    """DLinear's endpoint-replicated moving average, matching Autoformer."""

    def __init__(self, window: int) -> None:
        super().__init__()
        if window < 1 or window % 2 != 1:
            raise ValueError("DLinear moving-average window must be a positive odd integer")
        self.kernel_size = window
        self.avg = nn.AvgPool1d(kernel_size=window, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        padded = torch.cat((front, x, end), dim=1)
        return self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)


class _SeriesDecomposition(nn.Module):
    """State-dict-compatible implementation of DLinear's series_decomp."""

    def __init__(self, window: int) -> None:
        super().__init__()
        self.moving_avg = _MovingAverage(window)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class _DLinearBase(nn.Module):
    """Shared-channel DLinear predictor; preserves the reference's absolute output."""

    def __init__(self, lookback: int, horizon: int, channels: int,
                 moving_average: int = 25) -> None:
        super().__init__()
        if min(lookback, horizon, channels) < 1:
            raise ValueError("lookback, horizon, and channels must be positive")
        self.lookback, self.horizon, self.channels = lookback, horizon, channels
        # Keep attribute spelling aligned with the local Time-Series-Library
        # implementation so reference state_dicts load directly.
        self.decompsition = _SeriesDecomposition(moving_average)
        self.Linear_Seasonal = nn.Linear(lookback, horizon)
        self.Linear_Trend = nn.Linear(lookback, horizon)
        with torch.no_grad():
            self.Linear_Seasonal.weight.fill_(1.0 / lookback)
            self.Linear_Trend.weight.fill_(1.0 / lookback)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_series(x, self.lookback, self.channels)
        seasonal, trend = self.decompsition(x)
        seasonal = self.Linear_Seasonal(seasonal.permute(0, 2, 1))
        trend = self.Linear_Trend(trend.permute(0, 2, 1))
        return (seasonal + trend).permute(0, 2, 1)


def build_base(name: str, lookback: int, horizon: int, channels: int, hidden: int = 32) -> nn.Module:
    if name == "linear":
        return _LinearBase(lookback, horizon, channels)
    if name == "mlp":
        if hidden < 1:
            raise ValueError("hidden must be positive")
        return _MLPBase(lookback, horizon, channels, hidden)
    if name == "dlinear":
        return _DLinearBase(lookback, horizon, channels, moving_average=25)
    raise ValueError("name must be 'linear', 'mlp', or 'dlinear'")


def _validate_scales(scales: Sequence[int], lookback: int) -> tuple[int, ...]:
    result = tuple(int(k) for k in scales)
    if not result or any(k < 1 or k > lookback for k in result):
        raise ValueError("scales must be a non-empty sequence within lookback")
    return result


class DirectionHead(nn.Module):
    def __init__(self, lookback: int, horizon: int, channels: int,
                 scales: Sequence[int], hidden: int = 32) -> None:
        super().__init__()
        if min(lookback, horizon, channels, hidden) < 1:
            raise ValueError("dimensions and hidden must be positive")
        self.lookback, self.horizon, self.channels = lookback, horizon, channels
        self.scales = _validate_scales(scales, lookback)
        self.net = nn.Sequential(nn.Linear(lookback * channels + horizon * channels, hidden),
                                 nn.ReLU(), nn.Linear(hidden, len(self.scales) * horizon * channels * 3))

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        _check_series(x, self.lookback, self.channels)
        _check_base(x, base, self.horizon, self.channels)
        joined = torch.cat((x.reshape(x.shape[0], -1), base.reshape(x.shape[0], -1)), dim=-1)
        return self.net(joined).reshape(x.shape[0], len(self.scales), self.horizon, self.channels, 3)


class ResidualAdapter(nn.Module):
    def __init__(self, lookback: int, horizon: int, channels: int, hidden: int = 32) -> None:
        super().__init__()
        if min(lookback, horizon, channels, hidden) < 1:
            raise ValueError("dimensions and hidden must be positive")
        self.lookback, self.horizon, self.channels = lookback, horizon, channels
        self.net = nn.Sequential(nn.Linear(lookback * channels + horizon * channels, hidden),
                                 nn.ReLU(), nn.Linear(hidden, horizon * channels))

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        _check_series(x, self.lookback, self.channels)
        _check_base(x, base, self.horizon, self.channels)
        joined = torch.cat((x.reshape(x.shape[0], -1), base.reshape(x.shape[0], -1)), dim=-1)
        return base + self.net(joined).reshape_as(base)


class EventAdapter(nn.Module):
    def __init__(self, lookback: int, horizon: int, channels: int,
                 hidden: int = 32, use_events: bool = True) -> None:
        super().__init__()
        if min(lookback, horizon, channels, hidden) < 1:
            raise ValueError("dimensions and hidden must be positive")
        self.lookback, self.horizon, self.channels = lookback, horizon, channels
        self.use_events = bool(use_events)
        self.candidates = nn.Sequential(nn.Linear(lookback * channels + horizon * channels, hidden),
                                        nn.ReLU(), nn.Linear(hidden, horizon * channels))
        self.hazard = (nn.Sequential(nn.Linear(lookback * channels + horizon * channels, hidden),
                                     nn.ReLU(), nn.Linear(hidden, horizon * channels))
                       if self.use_events else None)

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_series(x, self.lookback, self.channels)
        _check_base(x, base, self.horizon, self.channels)
        joined = torch.cat((x.reshape(x.shape[0], -1), base.reshape(x.shape[0], -1)), dim=-1)
        candidate = self.candidates(joined).reshape_as(base)
        raw_turn_logits = self.hazard(joined).reshape_as(base) if self.use_events else None
        # 首步用历史差分斜率初始化，避免把未来标签当作事件先验。
        old_slope = x[:, -1:, :] - x[:, -2:-1, :] if self.lookback > 1 else torch.zeros_like(x[:, -1:, :])
        slopes, hazards = [], []
        previous = old_slope
        for h in range(self.horizon):
            new_slope = candidate[:, h:h + 1, :]
            if self.use_events:
                hazard = torch.sigmoid(raw_turn_logits[:, h:h + 1, :])
                previous = (1 - hazard) * previous + hazard * new_slope
                hazards.append(hazard)
            else:
                previous = new_slope
            slopes.append(previous)
        path = x[:, -1:, :] + torch.cat(slopes, dim=1).cumsum(dim=1)
        turn_logits = raw_turn_logits if self.use_events else base.new_zeros(base.shape)
        return {"path": path, "turn_logits": turn_logits}
