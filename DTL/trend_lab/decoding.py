"""Convex soft projection of one path onto posterior-weighted trend constraints."""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Integral

import torch
import torch.nn.functional as F

from .operators import multiscale_slopes


def project_trajectory(history: torch.Tensor, base: torch.Tensor, probabilities: torch.Tensor,
                       scales: Sequence[int], epsilon: torch.Tensor, strength: float = 1.0,
                       steps: int = 40, max_rms: float = 0.5, *,
                       convergence_tol: float | None = None,
                       return_diagnostics: bool = False):
    """Minimize the existing top-class/gap soft objective in an RMS ball.

    The default remains exactly ``steps`` iterations for compatibility with old
    experiments. With ``convergence_tol``, ``steps`` is a maximum: each sample
    stops when its projected-gradient mapping RMS is below the absolute
    tolerance. Reaching the iteration budget does not establish convergence.
    Optional diagnostics are JSON-serializable and do not change the path.
    This objective uses the top two probabilities; it is not posterior expected
    direction accuracy. See ``score_posterior_candidates`` for the latter.
    """
    if history.ndim != 3 or base.ndim != 3 or history.shape[0] != base.shape[0] or history.shape[2] != base.shape[2]:
        raise ValueError("history and base must have shapes [N,L,C] and [N,H,C]")
    n, lookback, channels = history.shape
    horizon = base.shape[1]
    if any(isinstance(k, bool) or not isinstance(k, Integral) for k in scales):
        raise ValueError("scales must be integers")
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
    if (not math.isfinite(strength) or not math.isfinite(max_rms)
            or isinstance(steps, bool) or not isinstance(steps, Integral)
            or strength < 0 or steps < 1 or max_rms < 0):
        raise ValueError("strength and max_rms must be non-negative; steps must be positive")
    if convergence_tol is not None and (not math.isfinite(convergence_tol) or convergence_tol <= 0):
        raise ValueError("convergence_tol must be finite and positive, or None")

    history = history.detach()
    base = base.detach()
    probabilities = probabilities.detach()
    epsilon = epsilon.detach()
    top = probabilities.topk(2, dim=-1)
    classes, weights = top.indices[..., 0], (top.values[..., 0] - top.values[..., 1])
    eps = epsilon.unsqueeze(0).unsqueeze(2)
    step_size = 1.0 / (1.0 + 4.0 * strength * sum(1.0 / (k * k) for k in ks) / len(ks))
    z = base.detach().clone()

    def objective_and_gradient(path):
        # Keep the original scalar summation order for legacy float32 paths.
        with torch.enable_grad():
            path = path.detach().requires_grad_(True)
            square = 0.5 * (path - base).square()
            objective = square.sum()
            per_path = square.sum(dim=(1, 2))
            slopes = multiscale_slopes(history, path, ks)
            for s in range(len(ks)):
                d = slopes[:, s]
                cls = classes[:, s]
                penalty = torch.where(
                    cls == 2, 0.5 * F.relu(1.5 * eps[:, s] - d).square(),
                    torch.where(cls == 0, 0.5 * F.relu(1.5 * eps[:, s] + d).square(),
                                0.5 * F.relu(d.abs() - eps[:, s]).square()))
                weighted = weights[:, s] * penalty
                objective = objective + strength / len(ks) * weighted.sum()
                per_path = per_path + strength / len(ks) * weighted.sum(dim=(1, 2))
            grad, = torch.autograd.grad(objective, path)
        return per_path.detach(), grad.detach()

    def projected_step(path, gradient):
        next_path = path - step_size * gradient
        delta = next_path - base
        rms = delta.square().mean(dim=(1, 2), keepdim=True).sqrt()
        factor = (max_rms / rms.clamp_min(torch.finfo(rms.dtype).tiny)).clamp(max=1.0)
        return base + delta * factor

    updates = torch.zeros(n, dtype=torch.long, device=base.device)
    active = torch.ones(n, dtype=torch.bool, device=base.device)
    # These two cases have the singleton optimum or the unchanged base optimum.
    if strength != 0 and max_rms != 0:
        for _ in range(steps):
            _, grad = objective_and_gradient(z)
            next_z = projected_step(z, grad)
            if convergence_tol is not None:
                residual = (z - next_z).square().mean(dim=(1, 2)).sqrt() / step_size
                active = active & (residual > convergence_tol)
                if not bool(active.any()):
                    break
                z = torch.where(active[:, None, None], next_z, z)
                updates += active.to(updates.dtype)
            else:
                z = next_z
                updates += 1
    if not return_diagnostics:
        return z.detach()
    objective, grad = objective_and_gradient(z)
    residual = (z - projected_step(z, grad)).square().mean(dim=(1, 2)).sqrt() / step_size
    rms = (z - base).square().mean(dim=(1, 2)).sqrt()
    converged = residual <= convergence_tol if convergence_tol is not None else None
    diagnostics = {
        "mode": "fixed_steps" if convergence_tol is None else "projected_gradient_tolerance",
        "max_steps": int(steps),
        "iterations_per_path": updates.cpu().tolist(),
        "convergence_tol": convergence_tol,
        "converged_per_path": None if converged is None else converged.cpu().tolist(),
        "all_converged": None if converged is None else bool(converged.all()),
        "projected_gradient_rms": residual.cpu().tolist(),
        "objective_per_path": objective.cpu().tolist(),
        "perturbation_rms": rms.cpu().tolist(),
        "cap_binding_per_path": (rms >= max_rms - 10 * torch.finfo(rms.dtype).eps).cpu().tolist(),
        "step_size": step_size,
    }
    return z.detach(), diagnostics


def score_posterior_candidates(history: torch.Tensor, candidates: torch.Tensor,
                               probabilities: torch.Tensor, scales: Sequence[int],
                               epsilon: torch.Tensor) -> torch.Tensor:
    """Expected 3-class accuracy for a finite set of paths, shape [K,N].

    Candidates have shape [K,N,H,C]; probabilities have [N,S,H,C,3]. The
    score is mean p(class(path)), not p(top class), and needs only marginal
    probabilities even when future outcomes are dependent. No claim of optimality
    outside the supplied candidates or an MSE guarantee follows from this score.
    Epsilon zero and scale [1] give raw-sign classes (down, zero, up).
    """
    if candidates.ndim != 4 or candidates.shape[0] < 1 or history.ndim != 3:
        raise ValueError("candidates and history must have shapes [K,N,H,C] and [N,L,C]")
    _, n, horizon, channels = candidates.shape
    if history.shape[0] != n or history.shape[2] != channels:
        raise ValueError("candidate batch/channel dimensions must match history")
    if probabilities.shape != (n, len(scales), horizon, channels, 3):
        raise ValueError("probabilities must have shape [N,S,H,C,3]")
    if epsilon.shape != (len(scales), channels):
        raise ValueError("epsilon must have shape [S,C]")
    tensors = (history, candidates, probabilities, epsilon)
    if any(t.device != history.device or t.dtype != history.dtype for t in tensors):
        raise ValueError("all tensors must have the same device and dtype")
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("inputs must be finite")
    if torch.any(probabilities < 0) or torch.any(epsilon < 0):
        raise ValueError("probabilities and epsilon must be non-negative")
    if not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("each probability vector must sum to one")
    scores = []
    with torch.no_grad():
        # Match the evaluator's hard-label arithmetic even for float32 paths.
        history64 = history.to(dtype=torch.float64)
        eps = epsilon.to(dtype=torch.float64)[None, :, None, :]
        probabilities64 = probabilities.to(dtype=torch.float64)
        for path in candidates:
            slopes = multiscale_slopes(history64, path.to(dtype=torch.float64), scales)
            classes = torch.where(slopes > eps, 2, torch.where(slopes < -eps, 0, 1)).long()
            assigned = probabilities64.gather(-1, classes[..., None]).squeeze(-1)
            scores.append(assigned.mean(dim=(1, 2, 3)))
    return torch.stack(scores)


def select_posterior_candidates(history: torch.Tensor, base: torch.Tensor,
                                candidates: torch.Tensor, probabilities: torch.Tensor,
                                scales: Sequence[int], epsilon: torch.Tensor, *,
                                penalty: float = 0.0):
    """Select each sample's path by posterior accuracy minus closeness cost.

    The cost is ``penalty * mean((candidate - base)**2)``. It regularizes
    displacement; it is NOT an estimate of the change in true MSE. Include base
    as the first candidate for a base-preferring exact tie. The caller owns RMS
    feasibility, calibration, and validation-only penalty selection.
    Returns selected paths [N,H,C] and candidate indices [N].
    """
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and non-negative")
    if candidates.ndim != 4 or base.shape != candidates.shape[1:]:
        raise ValueError("base must match candidate shape [N,H,C]")
    if base.device != candidates.device or base.dtype != candidates.dtype or not torch.isfinite(base).all():
        raise ValueError("base must be finite and share candidate device/dtype")
    scores = score_posterior_candidates(history, candidates, probabilities, scales, epsilon)
    cost = (candidates.to(torch.float64) - base.to(torch.float64)[None]).square().mean(dim=(2, 3))
    indices = (scores - penalty * cost).argmax(dim=0)
    batch = torch.arange(base.shape[0], device=base.device)
    return candidates[indices, batch].detach(), indices
