"""Stable direction-supervision controls, separate from evaluation dead zones.

These are experimental control objectives, not a proposed novel loss or a
claim of Fisher consistency for general conditional label distributions.
"""

import math

import torch
import torch.nn.functional as F

from .operators import multiscale_direction_labels, multiscale_slopes


def fit_slope_scales(train_x, train_y, scales, floor=0.02):
    """Return train-only median absolute slopes, floored in standardized units.

    The result has shape [S,C] and float64 precision. ``floor`` is a loss-scale
    hyperparameter fixed without test data; epsilon never enters this fit.
    Endpoint subtraction occurs in float64, matching the new label helper.
    """
    if not math.isfinite(float(floor)) or floor <= 0:
        raise ValueError("slope scale floor must be finite and positive")
    with torch.no_grad():
        slopes = multiscale_slopes(train_x.to(torch.float64), train_y.to(torch.float64), scales).abs()
        if slopes.numel() == 0 or not torch.isfinite(slopes).all():
            raise ValueError("training slopes must be nonempty and finite")
        flattened = slopes.permute(1, 3, 0, 2).reshape(slopes.shape[1], slopes.shape[3], -1)
        return torch.quantile(flattened, 0.5, dim=-1).clamp_min(float(floor))


def direction_surrogate_logits(slopes, epsilon, slope_scale):
    """Boundary-aligned logits [-d-eps, 0, d-eps] / scale.

    Epsilon determines class boundaries; a separate positive scale determines
    sensitivity. Away from exact boundaries, the winning logit agrees with
    down/flat/up labels at +/- epsilon. At an exact boundary, decode with
    ``direction_labels`` (which assigns flat), not raw argmax tie-breaking.
    This pointwise alignment does not establish population decision consistency.
    """
    if slopes.ndim != 4 or not torch.is_floating_point(slopes):
        raise ValueError("slopes must be a floating tensor with shape [N,S,H,C]")
    expected = (slopes.shape[1], slopes.shape[3])
    eps = torch.as_tensor(epsilon, dtype=slopes.dtype, device=slopes.device)
    scale = torch.as_tensor(slope_scale, dtype=slopes.dtype, device=slopes.device)
    if eps.shape != expected or scale.shape != expected:
        raise ValueError("epsilon and slope_scale must have shape [S,C]")
    if not all(torch.isfinite(value).all() for value in (slopes, eps, scale)):
        raise ValueError("slopes, epsilon, and slope_scale must be finite")
    if torch.any(eps < 0) or torch.any(scale <= 0):
        raise ValueError("epsilon must be nonnegative and slope_scale positive")
    eps = eps[None, :, None, :]
    scale = scale[None, :, None, :]
    return torch.stack(((-slopes - eps) / scale, torch.zeros_like(slopes),
                        (slopes - eps) / scale), dim=-1)


def direction_surrogate_loss(slopes, labels, epsilon, slope_scale, reduction="mean"):
    """Cross-entropy for the bounded-sensitivity control logits.

    For a single unreduced entry, the slope-gradient magnitude is at most
    2 / slope_scale, independently of epsilon. Use the train-fitted scale floor
    to bound this sensitivity; a tiny evaluation dead zone does not amplify it.
    """
    logits = direction_surrogate_logits(slopes, epsilon, slope_scale)
    if labels.shape != slopes.shape:
        raise ValueError("labels must have the same [N,S,H,C] shape as slopes")
    if labels.dtype != torch.long:
        raise TypeError("labels must be torch.long")
    if labels.device != slopes.device:
        raise ValueError("labels and slopes must be on the same device")
    if torch.any(labels < 0) or torch.any(labels > 2):
        raise ValueError("labels must use down=0, flat=1, up=2")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")
    loss = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), reduction=reduction)
    return loss.reshape_as(slopes) if reduction == "none" else loss


def residual_direction_diagnostics(history, truth, path, scales, epsilon, slope_scale,
                                   weight=0.1):
    """Measure separate losses and output-gradient norms without model gradients.

    Inputs are detached; no gradients accumulate on the supplied path or model.
    Zero-denominator ratios are None rather than NaN/infinity. Gradients are
    measured with respect to the path, not model parameters, and are diagnostic
    scale comparisons rather than a guarantee of useful training dynamics.
    """
    if not math.isfinite(float(weight)) or weight < 0:
        raise ValueError("direction weight must be finite and nonnegative")
    with torch.enable_grad():
        x = history.detach()
        y = truth.detach()
        prediction = path.detach().clone().requires_grad_(True)
        if prediction.shape != y.shape:
            raise ValueError("truth and path must have matching shapes")
        labels = multiscale_direction_labels(x, y, scales, epsilon)
        slopes = multiscale_slopes(x.to(torch.float64), prediction.to(torch.float64), scales)
        mse = F.mse_loss(prediction, y)
        direction = direction_surrogate_loss(slopes, labels, epsilon, slope_scale)
        mse_gradient, = torch.autograd.grad(mse, prediction, retain_graph=True)
        direction_gradient, = torch.autograd.grad(direction, prediction)
    mse_value, direction_value = float(mse.detach()), float(direction.detach())
    mse_norm = float(mse_gradient.norm())
    direction_norm = float(direction_gradient.norm())
    weighted_loss = float(weight) * direction_value
    weighted_norm = float(weight) * direction_norm
    return {
        "mse": mse_value, "direction_loss": direction_value,
        "weighted_direction_loss": weighted_loss,
        "mse_gradient_norm": mse_norm, "direction_gradient_norm": direction_norm,
        "weighted_direction_gradient_norm": weighted_norm,
        "loss_ratio": weighted_loss / mse_value if mse_value > 0 else None,
        "gradient_norm_ratio": weighted_norm / mse_norm if mse_norm > 0 else None,
    }
