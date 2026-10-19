"""Leakage-aware runner for the trend direction prototype."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import load_data
from .decoding import project_trajectory
from .metrics import evaluate_predictions, paired_block_interval
from .models import DirectionHead, EventAdapter, ResidualAdapter, build_base
from .operators import direction_labels, fit_thresholds, multiscale_slopes, turn_targets


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor(a: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(a, dtype=torch.float32, device=device)


def _params(model: nn.Module) -> dict[str, int]:
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x) if np.isfinite(x) else None
    if isinstance(x, float): return x if math.isfinite(x) else None
    if isinstance(x, (str, int, bool)) or x is None: return x
    return str(x)


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))


def _temperature(logits: torch.Tensor, labels: torch.Tensor, grid: list[float]) -> float:
    with torch.no_grad():
        return min(grid, key=lambda t: float(_nll(logits / t, labels)))


def _macro_f1(metrics: dict[str, Any]) -> float:
    vals = []
    for row in metrics.get("scales", {}).values():
        v = row.get("path", row).get("macro_f1")
        if v is not None:
            vals.append(float(v))
    if not vals:
        # Metrics implementations may expose scale rows as a list.
        vals = [float(r["macro_f1"]) for r in metrics.get("scales", [])
                if isinstance(r, dict) and r.get("macro_f1") is not None]
    return float(np.mean(vals)) if vals else -float("inf")


def _evaluate(x: torch.Tensor, y: torch.Tensor, path: torch.Tensor,
              scales: list[int], epsilon: torch.Tensor,
              probs: torch.Tensor | None = None) -> dict[str, Any]:
    return evaluate_predictions(x, y, path, scales, epsilon, probabilities=probs)


def _batch_indices(n: int, batch_size: int, shuffle: bool) -> list[torch.Tensor]:
    order = torch.randperm(n) if shuffle else torch.arange(n)
    return list(order.split(batch_size))


def _train(model: nn.Module, loss_fn, x: torch.Tensor, y: torch.Tensor,
           monitor_x: torch.Tensor, monitor_y: torch.Tensor,
           epochs: int, batch: int, lr: float, patience: int) -> tuple[int, float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_epoch, stale = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        for ids in _batch_indices(len(x), batch, True):
            loss = loss_fn(x[ids], y[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            weighted, count = 0.0, 0
            for i in _batch_indices(len(monitor_x), batch, False):
                weighted += float(loss_fn(monitor_x[i], monitor_y[i])) * len(i)
                count += len(i)
            val = weighted / count
        if val < best - 1e-8:
            best, best_epoch, stale = val, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best_state)
    return best_epoch, best


def _projection_alpha(validation_metrics: dict[str, Any], base_accuracy: float,
                      base_macro_f1: float, base_mse: float) -> tuple[float, list[dict[str, Any]]]:
    grid = validation_metrics["alpha_grid"]
    feasible = [r for r in grid if r["accuracy"] > base_accuracy
                and r["macro_f1"] >= base_macro_f1 - 1e-12 and r["mse"] <= base_mse * 1.01]
    chosen = sorted(feasible, key=lambda r: (-r["accuracy"], r["mse"], r["alpha"]))[0] if feasible else {"alpha": 0.0}
    return float(chosen["alpha"]), grid


def _residual_width(lookback: int, horizon: int, channels: int, head_params: int,
                    requested: int) -> int:
    limit = max(2 * requested, 128)
    return min(range(1, limit + 1), key=lambda width: abs(
        _params(ResidualAdapter(lookback, horizon, channels, width))["total"] - head_params))


def _direction_probabilities(labels: torch.Tensor, scales: list[int], epsilon: torch.Tensor,
                             history: torch.Tensor | None = None,
                             batch_size: int | None = None) -> torch.Tensor:
    n, _, horizon, channels = labels.shape
    if history is None:
        n = int(batch_size or n)
        chosen = torch.empty((len(scales), channels), dtype=torch.long, device=labels.device)
        for si in range(len(scales)):
            for ci in range(channels):
                chosen[si, ci] = torch.bincount(labels[:, si, :, ci].reshape(-1), minlength=3).argmax()
    else:
        n = history.shape[0]
        chosen = torch.empty((n, len(scales), channels), dtype=torch.long, device=history.device)
        for si, scale in enumerate(scales):
            slope = (history[:, -1, :] - history[:, -1 - scale, :]) / scale
            chosen[:, si] = torch.where(slope > epsilon[si], 2, torch.where(slope < -epsilon[si], 0, 1))
    if history is None:
        chosen = chosen[None].expand(n, -1, -1)
    expanded = chosen[:, :, None, :].expand(n, len(scales), horizon, channels)
    return F.one_hot(expanded, num_classes=3).float()


def _paired_differences(history: torch.Tensor, truth: torch.Tensor, candidate: torch.Tensor,
                        base: torch.Tensor, scales: list[int], epsilon: torch.Tensor,
                        std: np.ndarray) -> dict[str, Any]:
    with torch.no_grad():
        true = direction_labels(multiscale_slopes(history, truth, scales), epsilon)
        pred = direction_labels(multiscale_slopes(history, candidate, scales), epsilon)
        ref = direction_labels(multiscale_slopes(history, base, scales), epsilon)
        directional = {}
        for index, scale in enumerate(scales):
            score = (pred[:, index] == true[:, index]).float().mean((1, 2))
            base_score = (ref[:, index] == true[:, index]).float().mean((1, 2))
            directional[str(scale)] = paired_block_interval((score - base_score).cpu().numpy())
        mse = (candidate - truth).square().mean((1, 2))
        base_mse = (base - truth).square().mean((1, 2))
        std_t = torch.as_tensor(std, dtype=truth.dtype, device=truth.device)[None, None, :]
        original_mse = ((candidate - truth) * std_t).square().mean((1, 2))
        original_base_mse = ((base - truth) * std_t).square().mean((1, 2))
    return {"direction_accuracy_difference_by_scale": directional,
            "standardized_mse_difference": paired_block_interval((mse - base_mse).cpu().numpy()),
            "original_mse_difference": paired_block_interval((original_mse - original_base_mse).cpu().numpy())}


def _aggregate(values: list[Any]) -> Any:
    present = [v for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if present:
        return {"mean": float(np.mean(present)), "std": float(np.std(present, ddof=1)) if len(present) > 1 else 0.0}
    if values and all(isinstance(v, dict) for v in values):
        keys = set.intersection(*(set(v) for v in values))
        return {k: _aggregate([v[k] for v in values]) for k in sorted(keys)}
    if values and all(isinstance(v, list) for v in values) and len({len(v) for v in values}) == 1:
        try:
            arrays = [np.asarray(v, dtype=float) for v in values]
            if all(np.isfinite(a).all() for a in arrays):
                return {"mean": np.mean(arrays, axis=0).tolist(),
                        "std": (np.std(arrays, axis=0, ddof=1) if len(arrays) > 1
                                else np.zeros_like(arrays[0])).tolist()}
        except (TypeError, ValueError):
            pass
    return None


def _build_methods(cfg, lookback, horizon, channels, scales, hidden):
    methods = cfg.get("methods", ["base", "difference", "direction_head", "projection",
                                   "projection_single", "residual", "residual_direction", "event", "event_nohazard",
                                   "last_value", "last_slope", "majority", "last_direction"])
    valid = {"base", "difference", "direction_head", "projection", "projection_single",
             "residual", "residual_direction", "event", "event_nohazard", "last_value", "last_slope", "majority", "last_direction"}
    unknown = set(methods) - valid
    if unknown: raise ValueError(f"unknown methods: {sorted(unknown)}")
    return methods


def run(config: dict[str, Any], output: str | Path, smoke: bool = False) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(config)
    cfg["smoke"] = bool(smoke)
    if smoke:
        cfg["training"] = {**cfg.get("training", {}), "epochs": 1, "patience": 1,
                            "batch_size": min(8, cfg.get("training", {}).get("batch_size", 32))}
        cfg["seeds"] = cfg.get("smoke_seeds", [cfg.get("seeds", [0])[0]])
        dc = cfg.setdefault("data", {})
        dc["smoke"] = True
        if dc.get("kind", "synthetic") == "synthetic":
            ratios = dc.get("split_ratios", [0.6, 0.15, 0.1, 0.15])
            min_length = math.ceil((int(dc["horizon"]) + 1) / min(ratios[1:]))
            dc["length"] = min(int(dc.get("length", max(180, min_length + 4))), max(180, min_length + 4))
            dc["train_stride"] = max(int(dc.get("train_stride", 1)), int(dc["horizon"]))
    (output / "config.json").write_text(json.dumps(_jsonable(cfg), indent=2) + "\n")

    data_cfg = cfg["data"]
    bundle = load_data(data_cfg)
    # The bundle stores arrays standardized using train-only statistics.
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    train, cal, val, test = (s for s in (bundle.train, bundle.calibration, bundle.validation, bundle.test))
    tx, ty = _tensor(train.x, device), _tensor(train.y, device)
    cx, cy = _tensor(cal.x, device), _tensor(cal.y, device)
    vx, vy = _tensor(val.x, device), _tensor(val.y, device)
    qx, qy = _tensor(test.x, device), _tensor(test.y, device)
    lookback, channels = tx.shape[1], tx.shape[2]
    horizon = ty.shape[1]
    raw_scales = cfg.get("scales", [1, 2, 4])
    if (not raw_scales or any(not isinstance(k, int) or isinstance(k, bool) for k in raw_scales)
            or raw_scales[0] != 1 or len(set(raw_scales)) != len(raw_scales)
            or any(k < 1 or k >= lookback for k in raw_scales)):
        raise ValueError("scales must be unique integers, start with 1, and be smaller than lookback")
    scales = list(raw_scales)
    if any(len(getattr(bundle, name).x) == 0 for name in ("train", "calibration", "validation", "test")):
        raise ValueError("smoke/data configuration produced an empty chronological split")
    print(f"[trend_lab] loaded {bundle.metadata['kind']} data on {device}: "
          f"train/calibration/validation/test windows="
          f"{len(train.x)}/{len(cal.x)}/{len(val.x)}/{len(test.x)}", flush=True)
    eps = fit_thresholds(torch.from_numpy(train.x), torch.from_numpy(train.y), scales,
                         quantile=float(cfg.get("epsilon_quantile", 0.2)))
    epsilon = torch.as_tensor(eps, dtype=torch.float32, device=device)
    eps_np = epsilon.cpu().numpy()
    tr_slopes = multiscale_slopes(tx, ty, scales)
    tr_labels = direction_labels(tr_slopes, epsilon)
    cal_labels = direction_labels(multiscale_slopes(cx, cy, scales), epsilon)
    cfg_training = cfg.get("training", {})
    epochs = int(cfg_training.get("epochs", 40))
    batch = int(cfg_training.get("batch_size", 64))
    lr = float(cfg_training.get("learning_rate", 1e-3))
    patience = int(cfg_training.get("patience", 8))
    hidden = int(cfg.get("hidden", 32))
    method_names = _build_methods(cfg, lookback, horizon, channels, scales, hidden)
    seeds = list(map(int, cfg.get("seeds", [0, 1, 2, 3, 4])))
    data_seed = int(data_cfg.get("seed", 0))

    result: dict[str, Any] = {"schema_version": 1, "smoke": smoke, "data": _jsonable(bundle.metadata),
        "normalization": {"mean": _jsonable(bundle.mean), "std": _jsonable(bundle.std)},
        "scales": scales, "epsilon": _jsonable(eps_np), "data_seed": data_seed, "seeds": seeds,
        "methods": {}, "runs": []}
    predictions: dict[str, np.ndarray] = {"origins": np.asarray(test.origins)}
    checkpoints: dict[str, Any] = {}

    for seed in seeds:
        print(f"[trend_lab] seed {seed}: fitting base", flush=True)
        predictions[f"seed{seed}_truth"] = qy.cpu().numpy()
        predictions[f"seed{seed}_history"] = qx.cpu().numpy()
        _seed(seed)
        base = build_base(cfg.get("backbone", "linear"), lookback, horizon, channels, hidden).to(device)
        base_initial_state = copy.deepcopy(base.state_dict())
        base_parameter_counts = _params(base)
        _sync(device)
        t0 = time.perf_counter()
        base_loss = lambda x, y: F.mse_loss(base(x), y)
        _seed(seed + 100003)
        ep, train_loss = _train(base, base_loss, tx, ty, cx, cy, epochs, batch, lr, patience)
        _sync(device)
        base_seconds = time.perf_counter() - t0
        print(f"[trend_lab] seed {seed}: base complete ({ep} epochs, {base_seconds:.2f}s); fitting direction head",
              flush=True)
        base.eval()
        with torch.no_grad():
            bcal, bval = base(cx), base(vx)
        row: dict[str, Any] = {"seed": seed, "methods": {}}
        checkpoint = {"model_config": {"backbone": cfg.get("backbone", "linear"), "lookback": lookback,
                        "horizon": horizon, "channels": channels, "hidden": hidden, "scales": scales},
                      "config": copy.deepcopy(cfg),
                      "normalization": {"mean": torch.as_tensor(bundle.mean), "std": torch.as_tensor(bundle.std)},
                      "epsilon": epsilon.detach().cpu(), "temperature": None,
                      "selected_alpha": {}, "models": {"base": copy.deepcopy(base.state_dict())}}
        checkpoints[str(seed)] = checkpoint

        # Calibrate direction probabilities on calibration only, then hold temperature fixed.
        head = DirectionHead(lookback, horizon, channels, scales, hidden).to(device)
        head_parameter_counts = _params(head)
        base.requires_grad_(False)
        head_loss = lambda x, y: _nll(head(x, base(x).detach()), direction_labels(
            multiscale_slopes(x, y, scales), epsilon))
        _sync(device)
        t0 = time.perf_counter()
        hep, hloss = _train(head, head_loss, tx, ty, cx, cy, epochs, batch, lr, patience)
        _sync(device)
        head_seconds = time.perf_counter() - t0
        print(f"[trend_lab] seed {seed}: direction head complete ({hep} epochs, {head_seconds:.2f}s); "
              "selecting validation alpha", flush=True)
        head.eval()
        with torch.no_grad():
            cal_logits = head(cx, bcal)
            temperature = _temperature(cal_logits, cal_labels, list(map(float, cfg.get("temperature_grid", [0.5, 1, 2, 4]))))
            val_logits = head(vx, bval) / temperature
            val_probs = torch.softmax(val_logits, dim=-1)

        val_base_metrics = _evaluate(vx, vy, bval, scales, epsilon)
        base_mse = float(val_base_metrics.get("mse", F.mse_loss(bval, vy).item()))
        alpha_grid = list(map(float, cfg.get("alpha_grid", [0, .1, .3, 1, 3, 10])))
        if 0.0 not in alpha_grid or len(set(alpha_grid)) != len(alpha_grid):
            raise ValueError("alpha_grid must contain 0 and have unique values")
        base_accuracy = float(np.mean([r.get("path", r)["accuracy"] for r in val_base_metrics["scales"].values()]))
        base_macro_f1 = _macro_f1(val_base_metrics)
        grid_rows, single_grid_rows = [], []
        for alpha_value in alpha_grid:
            for target_scales, probs, eps, table in (
                (scales, val_probs, epsilon, grid_rows),
                ([1], val_probs[:, :1], epsilon[:1], single_grid_rows),
            ):
                pp = project_trajectory(vx, bval, probs, target_scales, eps, strength=alpha_value,
                    steps=int(cfg.get("projection_steps", 40)), max_rms=float(cfg.get("max_rms", .5)))
                with torch.no_grad():
                    mm = _evaluate(vx, vy, pp, scales, epsilon)
                accuracy = float(np.mean([r.get("path", r)["accuracy"] for r in mm["scales"].values()]))
                table.append({"alpha": alpha_value, "accuracy": accuracy, "macro_f1": _macro_f1(mm),
                              "mse": float(mm.get("mse", F.mse_loss(pp, vy).item()))})
        alpha, _ = _projection_alpha({"alpha_grid": grid_rows}, base_accuracy, base_macro_f1, base_mse)
        alpha_single, _ = _projection_alpha({"alpha_grid": single_grid_rows}, base_accuracy, base_macro_f1, base_mse)
        _sync(device)
        base_inference_start = time.perf_counter()
        with torch.no_grad():
            btest = base(qx)
        _sync(device)
        base_inference_seconds = time.perf_counter() - base_inference_start
        _sync(device)
        head_inference_start = time.perf_counter()
        with torch.no_grad():
            test_probs = torch.softmax(head(qx, btest) / temperature, dim=-1)
        _sync(device)
        head_inference_seconds = time.perf_counter() - head_inference_start
        checkpoint["temperature"] = temperature
        checkpoint["selected_alpha"] = {"projection": alpha, "projection_single": alpha_single}
        checkpoint["projection"] = {"steps": int(cfg.get("projection_steps", 40)),
                                     "max_rms": float(cfg.get("max_rms", 0.5))}
        checkpoint["models"]["direction_head"] = copy.deepcopy(head.state_dict())
        checkpoint["parameters"] = {"base": base_parameter_counts, "direction_head": head_parameter_counts}

        # Independent adapters use only x and the frozen base path.
        adapters: dict[str, nn.Module] = {}
        rwidth = _residual_width(lookback, horizon, channels, head_parameter_counts["total"], hidden)
        _seed(seed + 21001)
        residual = ResidualAdapter(lookback, horizon, channels, rwidth).to(device)
        adapters["residual"] = residual
        if "residual_direction" in method_names:
            residual_direction = ResidualAdapter(lookback, horizon, channels, rwidth).to(device)
            residual_direction.load_state_dict(copy.deepcopy(residual.state_dict()))
            adapters["residual_direction"] = residual_direction
        if "event" in method_names:
            _seed(seed + 21002)
            adapters["event"] = EventAdapter(lookback, horizon, channels, hidden, use_events=True).to(device)
        if "event_nohazard" in method_names:
            _seed(seed + 21002)
            nohazard = EventAdapter(lookback, horizon, channels, hidden, use_events=False).to(device)
            if "event" in adapters:
                nohazard.candidates.load_state_dict(copy.deepcopy(adapters["event"].candidates.state_dict()))
            adapters["event_nohazard"] = nohazard

        for method in method_names:
            print(f"[trend_lab] seed {seed}: evaluating method {method}", flush=True)
            if method in {"base", "difference", "direction_head", "projection", "projection_single",
                          "residual", "residual_direction", "event", "event_nohazard"}:
                pass
            else:
                continue
            fit_seconds, inference_seconds = 0.0, 0.0
            params, train_epochs, cal_monitor_loss = {"total": 0, "trainable": 0}, 0, None
            temp, selected_alpha = None, None
            if method == "base":
                pred, probabilities = btest, None
                params, train_epochs, cal_monitor_loss = base_parameter_counts, ep, train_loss
                inference_seconds = base_inference_seconds
                fit_seconds = base_seconds
            elif method == "difference":
                diff = build_base(cfg.get("backbone", "linear"), lookback, horizon, channels, hidden).to(device)
                diff.load_state_dict(base_initial_state)
                def dloss(x, y):
                    path = diff(x)
                    pred_diffs = torch.diff(torch.cat((x[:, -1:, :], path), dim=1), dim=1)
                    target_diffs = torch.diff(torch.cat((x[:, -1:, :], y), dim=1), dim=1)
                    return F.mse_loss(path, y) + float(cfg_training.get("difference_weight", 1.0)) * F.mse_loss(pred_diffs, target_diffs)
                _seed(seed + 100003)
                fit_start = time.perf_counter()
                de, dl = _train(diff, dloss, tx, ty, cx, cy, epochs, batch, lr, patience)
                _sync(device)
                fit_seconds = time.perf_counter() - fit_start
                _sync(device)
                inference_start = time.perf_counter()
                with torch.no_grad(): pred = diff(qx)
                _sync(device)
                inference_seconds = time.perf_counter() - inference_start
                params, train_epochs, cal_monitor_loss = _params(diff), de, dl
                checkpoints[str(seed)]["models"]["difference"] = copy.deepcopy(diff.state_dict())
                probabilities = None
            elif method == "direction_head":
                pred, probabilities = btest, test_probs
                params, train_epochs, cal_monitor_loss, temp = head_parameter_counts, hep, hloss, temperature
                fit_seconds = head_seconds
                inference_seconds = base_inference_seconds + head_inference_seconds
            elif method in {"projection", "projection_single"}:
                active_scales = scales if method == "projection" else [1]
                active_indices = [scales.index(k) for k in active_scales]
                p = test_probs[:, active_indices]
                selected_alpha = alpha if method == "projection" else alpha_single
                _sync(device)
                inference_start = time.perf_counter()
                pred = project_trajectory(qx, btest, p, active_scales, epsilon[active_indices], strength=selected_alpha,
                                          steps=int(cfg.get("projection_steps", 40)), max_rms=float(cfg.get("max_rms", .5)))
                _sync(device)
                inference_seconds = time.perf_counter() - inference_start
                probabilities = test_probs
                params, train_epochs, cal_monitor_loss, temp = head_parameter_counts, hep, hloss, temperature
                inference_seconds += head_inference_seconds
                fit_seconds = head_seconds
            else:
                model = adapters[method]
                def aloss(x, y):
                    with torch.no_grad(): b = base(x)
                    out = model(x, b)
                    path = out["path"] if isinstance(out, dict) else out
                    loss = F.mse_loss(path, y)
                    if method == "residual_direction":
                        scaled = multiscale_slopes(x, path, scales) / epsilon[None, :, None, :]
                        logits = torch.stack((-scaled, 1 - scaled.abs(), scaled), dim=-1)
                        labels = direction_labels(multiscale_slopes(x, y, scales), epsilon)
                        loss = loss + float(cfg_training.get("direction_weight", 0.1)) * _nll(logits, labels)
                    if method.startswith("event") and isinstance(out, dict) and "turn_logits" in out:
                        if method == "event":
                            target = turn_targets(x, y, epsilon[0])
                            loss = loss + float(cfg.get("event_loss_weight", 0.2)) * F.binary_cross_entropy_with_logits(out["turn_logits"], target)
                    return loss
                _sync(device)
                shuffle_seed = seed + (21001 if method.startswith("residual") else 21002)
                _seed(shuffle_seed)
                fit_start = time.perf_counter()
                ae, al = _train(model, aloss, tx, ty, cx, cy, epochs, batch, lr, patience)
                _sync(device)
                fit_seconds = time.perf_counter() - fit_start
                _sync(device)
                inference_start = time.perf_counter()
                with torch.no_grad():
                    out = model(qx, btest)
                    pred = out["path"] if isinstance(out, dict) else out
                _sync(device)
                inference_seconds = time.perf_counter() - inference_start
                params, train_epochs, cal_monitor_loss = _params(model), ae, al
                checkpoints[str(seed)]["models"][method] = copy.deepcopy(model.state_dict())
                if method.startswith("residual"):
                    checkpoints[str(seed)]["model_config"][method + "_hidden"] = rwidth
                elif method.startswith("event"):
                    checkpoints[str(seed)]["model_config"][method + "_hidden"] = hidden
                probabilities = None
            pred = pred.detach()
            evaluation_start = time.perf_counter()
            metrics = _evaluate(qx, qy, pred, scales, epsilon, probs=probabilities)
            base_test_mse = float(F.mse_loss(btest, qy).item())
            method_test_mse = float(metrics.get("mse", F.mse_loss(pred, qy).item()))
            mean = _tensor(bundle.mean, device)[None, None, :]
            std = _tensor(bundle.std, device)[None, None, :]
            original_metrics = _evaluate(qx * std + mean, qy * std + mean, pred * std + mean,
                                         scales, epsilon * std[:, 0, :])
            evaluation_seconds = time.perf_counter() - evaluation_start
            paired = _paired_differences(qx, qy, pred, btest, scales, epsilon, bundle.std)
            row["methods"][method] = {"metrics": _jsonable(metrics), "parameters": params,
                "fit_seconds": fit_seconds, "inference_seconds": inference_seconds,
                "shared_base_inference_seconds": base_inference_seconds,
                "evaluation_seconds": evaluation_seconds, "epochs": train_epochs,
                "calibration_monitor_loss": cal_monitor_loss, "temperature": temp,
                "selected_alpha": selected_alpha, "test_mse_ratio_to_base": method_test_mse / base_test_mse if base_test_mse > 0 else None,
                "original_units_metrics": _jsonable(original_metrics), "paired_vs_base": _jsonable(paired)}
            predictions[f"seed{seed}_{method}"] = pred.cpu().numpy()
            if method == "base":
                predictions[f"seed{seed}_truth"] = qy.cpu().numpy()
                predictions[f"seed{seed}_history"] = qx.cpu().numpy()
            if probabilities is not None:
                predictions[f"seed{seed}_{method}_probs"] = probabilities.cpu().numpy()
            print(f"[trend_lab] seed {seed}: method {method} complete "
                  f"(fit {fit_seconds:.2f}s, inference {inference_seconds:.2f}s)", flush=True)
        # Numerical and directional non-learned controls.
        with torch.no_grad():
            last = qx[:, -1:, :].expand(-1, horizon, -1)
            slope = qx[:, -1:, :] - qx[:, -2:-1, :]
            last_slope = qx[:, -1:, :] + torch.arange(1, horizon + 1, device=device)[None, :, None] * slope
            majority_probs = _direction_probabilities(tr_labels, scales, epsilon, batch_size=len(qx))
            last_probs = _direction_probabilities(tr_labels, scales, epsilon, history=qx)
            for method, pred in (("last_value", last), ("last_slope", last_slope)):
                if method in method_names:
                    inference_start = time.perf_counter()
                    met = _evaluate(qx, qy, pred, scales, epsilon)
                    original = _evaluate(qx * std + mean, qy * std + mean, pred * std + mean,
                                         scales, epsilon * std[:, 0, :])
                    paired = _paired_differences(qx, qy, pred, btest, scales, epsilon, bundle.std)
                    row["methods"][method] = {"metrics": _jsonable(met), "parameters": {"total": 0, "trainable": 0}, "fit_seconds": 0.0, "inference_seconds": time.perf_counter() - inference_start,
                        "original_units_metrics": _jsonable(original), "paired_vs_base": _jsonable(paired),
                        "test_mse_ratio_to_base": float(met.get("mse", F.mse_loss(pred, qy).item())) / float(F.mse_loss(btest, qy).item()) if float(F.mse_loss(btest, qy).item()) > 0 else None}
                    predictions[f"seed{seed}_{method}"] = pred.cpu().numpy()
            for method, probs in (("majority", majority_probs), ("last_direction", last_probs)):
                if method in method_names:
                    inference_start = time.perf_counter()
                    met = _evaluate(qx, qy, btest, scales, epsilon, probs=probs)
                    _sync(device)
                    inference_seconds = time.perf_counter() - inference_start
                    row["methods"][method] = {"direction_metrics": _jsonable({
                        "overall": met.get("probability"),
                        "by_scale": met.get("probability", {}).get("by_scale", {})}),
                        "parameters": {"total": 0, "trainable": 0}, "fit_seconds": 0.0, "inference_seconds": inference_seconds}
                    predictions[f"seed{seed}_{method}_probs"] = probs.cpu().numpy()
        row["validation"] = {"base": _jsonable(val_base_metrics), "alpha_grid": grid_rows,
                             "projection_single_alpha_grid": single_grid_rows,
                             "selected_alpha": {"projection": alpha, "projection_single": alpha_single},
                             "temperature": temperature,
                             "selection_sets": {"temperature": "calibration NLL", "temperature_grid": cfg.get("temperature_grid", [0.5, 1, 2, 4]),
                                                "projection_strength": "validation mean path accuracy, strictly above base; macro-F1 >= base - 1e-12; MSE <= 1.01 * base",
                                                "alpha_grid": alpha_grid}}
        row["training"] = {"base": {"parameters": base_parameter_counts, "fit_seconds": base_seconds, "epochs": ep,
                                     "calibration_monitor_loss": train_loss},
                            "direction_head": {"parameters": head_parameter_counts, "fit_seconds": head_seconds, "epochs": hep,
                                               "calibration_monitor_loss": hloss},
                            "residual_hidden": rwidth, "residual_parameters": _params(residual)}
        checkpoint["model_config"]["residual_hidden"] = rwidth
        row["test_origins"] = np.asarray(test.origins).tolist()
        result["runs"].append(_jsonable(row))
        print(f"[trend_lab] seed {seed}: complete", flush=True)

    for method in method_names:
        vals = [r["methods"][method].get("metrics", r["methods"][method].get("direction_metrics"))
                for r in result["runs"] if method in r["methods"]]
        if not vals: continue
        result["methods"][method] = {"per_seed": vals, "mean_std": _aggregate(vals)}
    (output / "results.json").write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    np.savez_compressed(output / "predictions.npz", **predictions)
    torch.save(checkpoints, output / "checkpoint.pt")
    reloaded = torch.load(output / "checkpoint.pt", map_location=device, weights_only=True)
    for seed in seeds:
        x_reload = torch.as_tensor(predictions[f"seed{seed}_history"], dtype=torch.float32, device=device)
        for method in reloaded[str(seed)]["models"]:
            if f"seed{seed}_{method}" not in predictions:
                continue
            if method == "base" or method == "direction_head" or method in {"difference", "residual", "residual_direction", "event", "event_nohazard", "projection", "projection_single"}:
                actual = predict_from_checkpoint(reloaded[str(seed)], x_reload, method)
                if isinstance(actual, dict):
                    actual = actual["path"]
                expected = torch.as_tensor(predictions[f"seed{seed}_{method}"], device=device)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for method in ("projection", "projection_single"):
            if f"seed{seed}_{method}" not in predictions:
                continue
            actual = predict_from_checkpoint(reloaded[str(seed)], x_reload, method)
            expected = torch.as_tensor(predictions[f"seed{seed}_{method}"], device=device)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    result["checkpoint_reload_verified"] = True
    (output / "results.json").write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    return result


def load_checkpoint(path: str | Path, seed: int = 0, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load one seed record from a runner checkpoint."""
    saved = torch.load(path, map_location=map_location, weights_only=True)
    return saved[str(seed)]


def predict_from_checkpoint(checkpoint: dict[str, Any], history: np.ndarray | torch.Tensor,
                            method: str = "base") -> torch.Tensor | dict[str, torch.Tensor]:
    """Predict from already standardized history using one saved method."""
    spec = checkpoint["model_config"]
    device = history.device if isinstance(history, torch.Tensor) else torch.device("cpu")
    x = torch.as_tensor(history, dtype=torch.float32, device=device)
    model_args = (spec["lookback"], spec["horizon"], spec["channels"])
    base = build_base(spec["backbone"], *model_args, spec["hidden"]).to(device)
    base.load_state_dict(checkpoint["models"]["base"])
    base.eval()
    with torch.no_grad():
        base_path = base(x)
    if method == "base":
        return base_path
    if method in {"projection", "projection_single", "direction_head"}:
        head = DirectionHead(*model_args, spec["scales"], spec["hidden"]).to(device)
        head.load_state_dict(checkpoint["models"]["direction_head"])
        head.eval()
        with torch.no_grad():
            probs = torch.softmax(head(x, base_path) / float(checkpoint["temperature"]), dim=-1)
        if method == "direction_head":
            return {"path": base_path, "probabilities": probs}
        ks = spec["scales"] if method == "projection" else [1]
        ids = [spec["scales"].index(k) for k in ks]
        return project_trajectory(x, base_path, probs[:, ids], ks,
            checkpoint["epsilon"].to(device)[ids],
            strength=float(checkpoint["selected_alpha"][method]),
            steps=int(checkpoint["projection"]["steps"]),
            max_rms=float(checkpoint["projection"]["max_rms"]))
    if method not in checkpoint["models"]:
        raise ValueError(f"method {method!r} is not present in this checkpoint")
    if method == "difference":
        model = build_base(spec["backbone"], *model_args, spec["hidden"]).to(device)
    elif method.startswith("residual"):
        model = ResidualAdapter(*model_args, spec[method + "_hidden"]).to(device)
    elif method.startswith("event"):
        model = EventAdapter(*model_args, spec[method + "_hidden"], use_events=(method == "event")).to(device)
    else:
        raise ValueError(f"unsupported checkpoint method: {method}")
    model.load_state_dict(checkpoint["models"][method])
    model.eval()
    with torch.no_grad():
        result = model(x, base_path) if method.startswith(("residual", "event")) else model(x)
    return result["path"] if isinstance(result, dict) else result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    output = args.output or Path(cfg.get("output", "trend_lab_output"))
    data_cfg = cfg.get("data", {})
    scenarios = data_cfg.pop("scenarios", None)
    backbones = cfg.pop("backbones", None)
    scenarios = scenarios or [data_cfg.get("scenario")]
    backbones = backbones or [cfg.get("backbone", "linear")]
    multi = len(scenarios) * len(backbones) > 1
    for scenario in scenarios:
        for backbone in backbones:
            current = copy.deepcopy(cfg)
            current["data"] = copy.deepcopy(data_cfg)
            if scenario is not None:
                current["data"]["scenario"] = scenario
            current["backbone"] = backbone
            target = output / f"{scenario or 'data'}__{backbone}" if multi else output
            run(current, target, args.smoke)
            print(f"Saved run artifacts to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
