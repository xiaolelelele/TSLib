"""Small, source-checked forecasting loss baselines (reviewed 2026-09-29).

These implement the losses, not the papers' full benchmark/training protocols.
All inputs use the lab's [batch, time, channel] convention; the last history
observation anchors the first forecast difference.

CosDir reference:
  https://arxiv.org/html/2608.01857v1 (Eq. 2)
  https://github.com/seunghan96/cosdir/blob/main/losses/tslib_losses.py
The default follows the official implementation's torch cosine_similarity
normalization, whose individual norm clamps differ near zero from the paper's
``norm_product + epsilon`` notation. This is fixed-weight CosDir, not CosDir-UW.
The official repository's DA metric excludes true zero changes; this module
does not replace the lab's metrics or silently adopt that exclusion.

TDT/TDAlign reference:
  https://arxiv.org/html/2406.04777v1 (Eqs. 4-7)
  https://arxiv.org/html/2406.04777v3 (Eqs. 2-6; TKDE 2025)
  https://github.com/XQ-edu/TDAlign/blob/main/exp/exp_main.py (_comput_loss)
The official implementation computes rho per sample and channel, averaging
over the horizon before weighting the two losses. It counts strictly opposite
signs (difference products < 0), so a zero versus nonzero change is not counted.
The optional ``paper`` variant uses unequal torch.sign values, matching the
equation's zero convention, while retaining per-sample/channel aggregation.
It must be named separately in experiment records. A global minibatch rho is
not equivalent to either implementation here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _differences(history: Tensor, path: Tensor, truth: Tensor) -> tuple[Tensor, Tensor]:
    if history.ndim != 3 or path.ndim != 3 or truth.ndim != 3:
        raise ValueError("history, path, and truth must have shape [batch, time, channel]")
    if any(size == 0 for value in (history, path, truth) for size in value.shape):
        raise ValueError("history, path, and truth must have nonempty dimensions")
    if path.shape != truth.shape or history.shape[0] != path.shape[0] or history.shape[2] != path.shape[2]:
        raise ValueError("path/truth shapes and history batch/channel dimensions must match")
    if not all(value.is_floating_point() for value in (history, path, truth)):
        raise ValueError("loss inputs must be floating-point tensors")
    if len({value.device for value in (history, path, truth)}) != 1:
        raise ValueError("loss inputs must be on the same device")
    if len({value.dtype for value in (history, path, truth)}) != 1:
        raise ValueError("loss inputs must have the same dtype")
    anchor = history[:, -1:, :]
    return (
        path - torch.cat((anchor, path[:, :-1, :]), dim=1),
        truth - torch.cat((anchor, truth[:, :-1, :]), dim=1),
    )


def cosdir_loss(
    history: Tensor,
    path: Tensor,
    truth: Tensor,
    weight: float = 0.5,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Official fixed-weight CosDir: MSE + weight * mean(1 - cosine).

    Cosine is taken along the complete horizon separately for each sample and
    channel, not per scalar increment and not across channels. The official
    zero-vector convention is preserved: two flat paths contribute a cosine
    penalty of one, even if their MSE is zero.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    predicted, observed = _differences(history, path, truth)
    directional = (1 - F.cosine_similarity(predicted, observed, dim=1, eps=eps)).mean()
    return F.mse_loss(path, truth) + weight * directional


def tdt_loss(
    history: Tensor,
    path: Tensor,
    truth: Tensor,
    *,
    variant: str = "official",
) -> Tensor:
    """TDAlign/TDT adaptive MSE loss; default reproduces the official loss.

    ``rho`` weights the value loss, and ``1-rho`` weights the difference loss.
    Do not invert those weights. ``variant='paper'`` changes only the sign
    mismatch convention, as documented at module level. Neither has learned
    weights or a tunable mixing coefficient.
    """
    if variant not in {"official", "paper"}:
        raise ValueError("variant must be 'official' or 'paper'")
    predicted, observed = _differences(history, path, truth)
    if variant == "official":
        mismatch = predicted * observed < 0
        rho = mismatch.sum(dim=1).float() / predicted.shape[1]
    else:
        mismatch = torch.sign(predicted) != torch.sign(observed)
        rho = mismatch.to(path.dtype).mean(dim=1)
    # Comparison supplies a fixed, nondifferentiable weight at each update,
    # as in the reference implementation (there is no straight-through path).
    value_error = (path - truth).square().mean(dim=1)
    difference_error = (predicted - observed).square().mean(dim=1)
    return (rho * value_error + (1 - rho) * difference_error).mean()
