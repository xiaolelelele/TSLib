"""Trend-label operators shared by training and evaluation."""

import torch


def _as_3d(value):
    if value.ndim != 3:
        raise ValueError("expected [N,T,C] tensors")
    return value


def multiscale_slopes(history, future, scales):
    _as_3d(history)
    _as_3d(future)
    if history.shape[0] != future.shape[0] or history.shape[2] != future.shape[2]:
        raise ValueError("history and future batch/channel dimensions must match")
    if not scales or any(not isinstance(k, int) or k < 1 or k > history.shape[1] for k in scales):
        raise ValueError("scales must be nonempty integers in [1, lookback]")
    joined = torch.cat((history, future), dim=1)
    origin = history.shape[1] - 1
    rows = []
    for k in scales:
        step = torch.arange(future.shape[1], device=history.device) + 1
        end = joined[:, origin + step]
        start = joined[:, origin + step - k]
        rows.append((end - start) / k)
    return torch.stack(rows, dim=1).permute(0, 1, 2, 3)


def direction_labels(slopes, epsilon):
    if slopes.ndim != 4:
        raise ValueError("slopes must have shape [N,S,H,C]")
    eps = torch.as_tensor(epsilon, device=slopes.device, dtype=slopes.dtype)
    if eps.shape != (slopes.shape[1], slopes.shape[3]):
        raise ValueError("epsilon must have shape [S,C]")
    eps = eps[None, :, None, :]
    return torch.where(slopes > eps, 2, torch.where(slopes < -eps, 0, 1)).long()


def multiscale_direction_labels(history, future, scales, epsilon):
    """Classify slopes after float64 subtraction, shared by new training/scoring.

    Casting already-computed float32 slopes cannot recover rounding at a hard
    boundary. Convert the endpoint values before subtraction instead. The
    original ``direction_labels`` helper remains available for legacy runs.
    """
    slopes = multiscale_slopes(history.to(torch.float64), future.to(torch.float64), scales)
    return direction_labels(slopes, epsilon)


def fit_thresholds(train_x, train_y, scales, quantile=0.2, *, precision="legacy"):
    """Fit train-only dead zones; legacy precision preserves old configurations."""
    _as_3d(train_x)
    _as_3d(train_y)
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0,1]")
    if precision not in {"legacy", "float64"}:
        raise ValueError("precision must be 'legacy' or 'float64'")
    if precision == "float64":
        train_x, train_y = train_x.to(torch.float64), train_y.to(torch.float64)
    slopes = multiscale_slopes(train_x, train_y, scales).abs()
    flattened = slopes.permute(1, 3, 0, 2).reshape(slopes.shape[1], slopes.shape[3], -1)
    return torch.quantile(flattened, quantile, dim=-1).clamp_min(1e-6)


def turn_targets(history, future, epsilon):
    """Mark direction reversals as [N,H,C] targets.

    Flat steps emit no event and preserve the most recent non-flat direction.
    The first future non-flat direction is compared with the most recent
    non-flat direction in history; later future directions use the same rule.
    """
    _as_3d(history)
    _as_3d(future)
    eps = torch.as_tensor(epsilon, device=history.device, dtype=history.dtype)
    if eps.shape != (history.shape[2],):
        raise ValueError("epsilon must have shape [C]")
    points = torch.cat((history[:, -1:, :], future), dim=1)
    slopes = points[:, 1:, :] - points[:, :-1, :]
    directions = torch.where(slopes > eps, 1, torch.where(slopes < -eps, -1, 0))
    history_slopes = history[:, 1:, :] - history[:, :-1, :]
    history_directions = torch.where(history_slopes > eps, 1, torch.where(history_slopes < -eps, -1, 0))
    last = torch.zeros_like(directions[:, 0])
    for h in range(history_directions.shape[1] - 1, -1, -1):
        last = torch.where(last.eq(0), history_directions[:, h], last)
    turns = []
    for h in range(directions.shape[1]):
        current = directions[:, h]
        turns.append((current.ne(0) & last.ne(0) & current.ne(last)).to(torch.float32))
        last = torch.where(current.ne(0), current, last)
    return torch.stack(turns, dim=1)
