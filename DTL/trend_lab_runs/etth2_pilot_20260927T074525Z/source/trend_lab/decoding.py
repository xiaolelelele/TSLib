"""Convex soft projection of one path onto posterior-weighted trend constraints."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .operators import multiscale_slopes


def project_trajectory(history: torch.Tensor, base: torch.Tensor, probabilities: torch.Tensor,
                       scales: Sequence[int], epsilon: torch.Tensor, strength: float = 1.0,
                       steps: int = 40, max_rms: float = 0.5) -> torch.Tensor:
    if history.ndim != 3 or base.ndim != 3 or history.shape[0] != base.shape[0] or history.shape[2] != base.shape[2]:
        raise ValueError("history and base must have shapes [N,L,C] and [N,H,C]")
    n, lookback, channels = history.shape
    horizon = base.shape[1]
    ks = tuple(int(k) for k in scales)
    if not ks or any(k < 1 or k > lookback for k in ks):
        raise ValueError("scales must be a non-empty sequence within lookback")
    if probabilities.shape != (n, len(ks), horizon, channels, 3):
        raise ValueError("probabilities must have shape [N,S,H,C,3]")
    if epsilon.shape != (len(ks), channels):
        raise ValueError("epsilon must have shape [S,C]")
    if not (history.device == base.device == probabilities.device == epsilon.device):
        raise ValueError("all tensors must be on the same device")
    if not (history.dtype == base.dtype == probabilities.dtype == epsilon.dtype):
        raise ValueError("all tensors must have the same dtype")
    if not all(torch.isfinite(t).all() for t in (history, base, probabilities, epsilon)):
        raise ValueError("inputs must be finite")
    if torch.any(probabilities < 0) or torch.any(epsilon < 0):
        raise ValueError("probabilities and epsilon must be non-negative")
    sums = probabilities.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
        raise ValueError("each probability vector must sum to one")
    if strength < 0 or steps < 1 or max_rms < 0:
        raise ValueError("strength and max_rms must be non-negative; steps must be positive")
    if strength == 0:
        return base.detach()

    history = history.detach()
    base = base.detach()
    probabilities = probabilities.detach()
    epsilon = epsilon.detach()
    top = probabilities.topk(2, dim=-1)
    classes, weights = top.indices[..., 0], (top.values[..., 0] - top.values[..., 1])
    eps = epsilon.unsqueeze(0).unsqueeze(2)
    step_size = 1.0 / (1.0 + 4.0 * strength * sum(1.0 / (k * k) for k in ks) / len(ks))
    z = base.detach().clone()
    with torch.enable_grad():
        for _ in range(steps):
            z.requires_grad_(True)
            objective = 0.5 * (z - base).square().sum()
            slopes = multiscale_slopes(history, z, ks)
            for s in range(len(ks)):
                d = slopes[:, s]
                cls = classes[:, s]
                penalty = torch.where(
                    cls == 2, 0.5 * F.relu(1.5 * eps[:, s] - d).square(),
                    torch.where(cls == 0, 0.5 * F.relu(1.5 * eps[:, s] + d).square(),
                                0.5 * F.relu(d.abs() - eps[:, s]).square()))
                objective = objective + strength / len(ks) * (weights[:, s] * penalty).sum()
            grad, = torch.autograd.grad(objective, z)
            with torch.no_grad():
                z = z - step_size * grad
                delta = z - base
                rms = delta.square().mean(dim=(1, 2), keepdim=True).sqrt()
                factor = (max_rms / rms.clamp_min(torch.finfo(rms.dtype).tiny)).clamp(max=1.0)
                z = base + delta * factor
    return z.detach()
