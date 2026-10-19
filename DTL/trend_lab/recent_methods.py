"""Explicit research adaptations for the frozen recent-data protocol.

CaReTS equations: https://arxiv.org/html/2511.09789v1, Eqs 4/5/13/15/17.
These are paper-formula adaptations, not the expired anonymous implementation.
The raw projection and paired ridge gate are exploratory, with no risk theorem.
"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import build_base


class ReferenceITransformer(nn.Module):
    def __init__(self, lookback, horizon, channels, reference_root):
        super().__init__()
        root = Path(reference_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("trend_reference_itransformer", root / "models/iTransformer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = SimpleNamespace(task_name="long_term_forecast", seq_len=lookback, pred_len=horizon,
                                 enc_in=channels, d_model=64, d_ff=128, e_layers=2, n_heads=4,
                                 embed="timeF", freq="h", dropout=.1, factor=5, activation="gelu")
        self.reference = module.Model(config)

    def forward(self, x):
        return self.reference(x, None, None, None)


def backbone(name, lookback, horizon, channels, reference_root):
    if name == "itransformer":
        return ReferenceITransformer(lookback, horizon, channels, reference_root)
    return build_base(name, lookback, horizon, channels)


class CaReTS(nn.Module):
    """CaReTS3/4 with the paper's supported two-layer CNN temporal encoder."""
    def __init__(self, lookback, horizon, channels, variant):
        super().__init__()
        if variant not in (3, 4):
            raise ValueError("only paper-formula CaReTS3 and CaReTS4 are implemented")
        self.variant, self.horizon, self.channels = variant, horizon, channels
        self.encoder = nn.Sequential(nn.Conv1d(channels, 64, 3, padding=1), nn.ReLU(),
                                     nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                        nn.Linear(64, horizon * channels * 2))
        in_features = 64 if variant == 3 else 64 + horizon * channels * 2
        out_features = horizon * channels * (2 if variant == 3 else 1)
        self.regressor = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                       nn.Linear(64, out_features))
        self.log_variances = nn.Parameter(torch.zeros(3 if variant == 3 else 2))

    def components(self, x):
        h = self.encoder(x.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(h).reshape(len(x), self.horizon, self.channels, 2)
        probabilities = logits.softmax(-1)  # class 0=up, class 1=down
        if self.variant == 3:
            deviations = F.relu(self.regressor(h)).reshape_as(probabilities)
            signed = probabilities[..., 0] * deviations[..., 0] - probabilities[..., 1] * deviations[..., 1]
        else:
            joined = torch.cat((h, probabilities.flatten(1)), dim=1)
            signed = self.regressor(joined).reshape(len(x), self.horizon, self.channels)
            deviations = None
        return x[:, -1:, :] + signed, logits, deviations

    def forward(self, x):
        return self.components(x)[0]

    def task_losses(self, x, y):
        path, logits, deviation = self.components(x)
        delta = y - x[:, -1:, :]
        # Equality is unspecified by the binary paper; the protocol fixes up.
        labels = (delta.double() < 0).long()
        ce = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
        mse = F.mse_loss(path, y)
        if self.variant == 4:
            return torch.stack((ce, mse))
        selected = deviation.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        magnitude = (selected - delta.abs()).square().mean()
        return torch.stack((ce, magnitude, mse))

    def loss(self, x, y):
        variance = self.log_variances.clamp(-10, 10)
        return (.5 * torch.exp(-variance) * self.task_losses(x, y) + .5 * variance).sum() + .001 * variance.square().sum()


def raw_differences(x, path):
    return path - torch.cat((x[:, -1:, :], path[:, :-1, :]), dim=1)


def raw_labels(x, path):
    d = raw_differences(x.double(), path.double())
    return torch.where(d > 0, 2, torch.where(d < 0, 0, 1))


def raw_objective_gradient(x, base, path, probabilities, margin, strength):
    d = raw_differences(x, path)
    top = probabilities.topk(2, dim=-1)
    cls = top.indices[..., 0]
    weight = top.values[..., 0] - top.values[..., 1]
    residual = torch.where(cls == 0, F.relu(d + margin),
                           torch.where(cls == 2, -F.relu(margin - d), d))
    # D.T * weighted residual: each future value affects its own and next difference.
    weighted = weight * residual
    difference_grad = weighted.clone()
    difference_grad[:, :-1, :] -= weighted[:, 1:, :]
    gradient = path - base + strength * difference_grad
    objective = .5 * ((path - base).square() + strength * weight * residual.square()).sum((1, 2))
    return objective, gradient


def raw_project(x, base, probabilities, margin, strength, max_rms=.5, tol=1e-5, steps=600):
    if x.ndim != 3 or base.ndim != 3 or probabilities.shape != (*base.shape, 3):
        raise ValueError("invalid raw projection dimensions")
    if x.shape[0] != len(base) or x.shape[2] != base.shape[2] or margin.shape != (base.shape[2],):
        raise ValueError("invalid raw projection batch/channel/margin")
    if strength < 0 or max_rms < 0 or tol <= 0 or steps < 1:
        raise ValueError("invalid solver constants")
    if not all(torch.isfinite(t).all() for t in (x, base, probabilities, margin)):
        raise ValueError("projection inputs must be finite")
    if torch.any(probabilities < 0) or torch.any(margin < 0) or not torch.allclose(probabilities.sum(-1), torch.ones_like(base), atol=1e-5):
        raise ValueError("invalid probability or margin")
    step_size = 1 / (1 + 4 * strength)

    def step(z, grad):
        delta = z - step_size * grad - base
        rms = delta.square().mean((1, 2), keepdim=True).sqrt()
        return base + delta * (max_rms / rms.clamp_min(torch.finfo(base.dtype).tiny)).clamp(max=1)

    with torch.no_grad():
        z = base.clone()
        iterations = torch.zeros(len(z), dtype=torch.long, device=z.device)
        for _ in range(steps):
            _, grad = raw_objective_gradient(x, base, z, probabilities, margin, strength)
            next_z = step(z, grad)
            residual = (z - next_z).square().mean((1, 2)).sqrt() / step_size
            active = residual > tol
            if not active.any():
                break
            z = torch.where(active[:, None, None], next_z, z)
            iterations += active.long()
        objective, grad = raw_objective_gradient(x, base, z, probabilities, margin, strength)
        residual = (z - step(z, grad)).square().mean((1, 2)).sqrt() / step_size
        diagnostic = {"all_converged": bool((residual <= tol).all()),
                      "max_residual": float(residual.max()), "max_iterations": int(iterations.max()),
                      "iterations": iterations.cpu().tolist(), "objective": objective.cpu().tolist(),
                      "rms": (z - base).square().mean((1, 2)).sqrt().cpu().tolist()}
    return z, diagnostic


def symbols(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return np.sign(np.diff(np.concatenate((x[:, -1:, :], y), axis=1), axis=1))


def paired_targets(x, y, base, candidate):
    observed, predicted, original = symbols(x, y), symbols(x, candidate), symbols(x, base)
    gain = ((predicted == observed).astype(float) - (original == observed)).mean((1, 2))
    risk = ((np.asarray(candidate, dtype=float) - y)**2 - (np.asarray(base, dtype=float) - y)**2).mean((1, 2))
    return np.column_stack((gain, risk))


def gate_features(x, base, candidate, probabilities):
    x, b, z, p = [np.asarray(v, dtype=np.float64) for v in (x, base, candidate, probabilities)]
    if x.ndim != 3 or x.shape[1] < 2:
        raise ValueError("gate features require at least two historical observations")
    delta = z - b
    db = np.diff(np.concatenate((x[:, -1:, :], b), axis=1), axis=1)
    dx = np.diff(x, axis=1)
    bz = symbols(x, b).astype(int) + 1
    zz = symbols(x, z).astype(int) + 1
    pb = np.take_along_axis(p, bz[..., None], axis=-1)[..., 0]
    pz = np.take_along_axis(p, zz[..., None], axis=-1)[..., 0]
    avg = lambda a: np.mean(a, axis=(1, 2))
    last = x[:, -1:, :]
    recent = x[:, -24:, :].mean(1, keepdims=True)
    # The last three terms are observable bias/amplitude proxies in the MSE identity.
    columns = [avg(pz - pb), avg(-np.sum(p * np.log(p.clip(1e-12)), axis=-1)),
               avg(p.max(-1)), avg(delta**2), avg(np.abs(delta)), np.max(np.abs(delta), axis=(1, 2)),
               avg(np.abs(db)), avg(np.abs(dx)), avg(dx**2), avg(np.abs(dx[:, -24:, :])),
               avg(np.abs(b - last)), avg(x[:, -24:, :].var(1, keepdims=True)),
               avg((p.argmax(-1) == 1).astype(float)), avg((bz != zz).astype(float)),
               avg(delta * (b - last)), avg(delta * (b - recent)), avg(delta * b)]
    features = np.column_stack(columns)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite gate features")
    return features


def fit_gate(features, targets, ridge=10.):
    features, targets = np.asarray(features, float), np.asarray(targets, float)
    mean, std = features.mean(0), features.std(0).clip(1e-8)
    design = np.column_stack((np.ones(len(features)), (features - mean) / std))
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {"mean": mean.tolist(), "std": std.tolist(), "coef": coef.tolist(), "ridge": ridge}


def predict_gate(spec, features):
    design = np.column_stack((np.ones(len(features)), (features - np.asarray(spec["mean"])) / np.asarray(spec["std"])))
    return design @ np.asarray(spec["coef"])
