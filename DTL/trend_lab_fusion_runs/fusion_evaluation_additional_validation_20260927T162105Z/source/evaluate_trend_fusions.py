"""Validation-first evaluation of the pre-registered trend fusion set.

No models are trained here. Each completed cell's checkpoint is reloaded,
projection strengths and fusion weights are selected on validation, and only
then are the frozen paths scored on that cell's test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "DTL"))

from trend_lab.data import load_data
from trend_lab.decoding import project_trajectory
from trend_lab.models import build_base, DirectionHead, ResidualAdapter
from trend_lab.operators import direction_labels, fit_thresholds, multiscale_slopes
from trend_lab.run import predict_from_checkpoint


ALPHA_GRID = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
SCALES = [1, 2, 4, 8]
PROJECTION_STEPS = 40
MAX_RMS = 0.5
SEEDS = [0, 1, 2]
ATOM_NAMES = ["learned_multi", "learned_single", "prior", "flat"]
STANDALONE_ORDER = [
    "base", "difference", "residual", "residual_direction",
    "learned_single", "learned_multi", "prior", "flat",
]
PAIR_SPECS = [
    ("base_plus_difference", "base", "difference"),
    ("base_plus_residual", "base", "residual"),
    ("base_plus_residual_direction", "base", "residual_direction"),
    ("learned_selected_plus_flat", "learned_selected", "flat"),
    ("learned_selected_plus_prior", "learned_selected", "prior"),
    ("learned_selected_plus_difference", "learned_selected", "difference"),
]
FUSION_ORDER = [pair[0] for pair in PAIR_SPECS] + ["threeway_mean"]
STRICT_MSE_TOLERANCE = 1e-8
BOOTSTRAP_BLOCK_SIZE = 8
BOOTSTRAP_REPEATS = 2000


def _bootstrap_indices(length: int) -> np.ndarray:
    block = min(BOOTSTRAP_BLOCK_SIZE, int(length))
    if block < 1:
        return np.empty((BOOTSTRAP_REPEATS, 0), dtype=np.int64)
    blocks_needed = int(math.ceil(length / block))
    rng = np.random.default_rng(0)
    starts = rng.integers(0, length - block + 1, size=(BOOTSTRAP_REPEATS, blocks_needed))
    return (starts[:, :, None] + np.arange(block, dtype=np.int64)[None, None, :]).reshape(
        BOOTSTRAP_REPEATS, -1
    )[:, :length]


def _interval_from_bootstrap(values: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not flat.size or not np.isfinite(flat).all():
        return {"mean": None, "ci95": [None, None]}
    means = flat[indices].mean(axis=1)
    return {"mean": float(flat.mean()),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    if not result:
        raise ValueError(f"cannot make a safe identifier from {value!r}")
    return result


def _number_token(value: float) -> str:
    text = f"{float(value):.8g}".replace("-", "m").replace(".", "p")
    return text


def _npz_key(group: str, seed: int, split: str, candidate_id: str, field: str = "path") -> str:
    return f"{group}__seed{seed}__{split}__{candidate_id}__{field}"


def _tensor_to_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _resolve_data_path(cell_dir: Path, value: str) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        (WORKSPACE / "DTL" / raw), (WORKSPACE / raw), (cell_dir / raw),
        (cell_dir.parent / raw),
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"could not resolve configured data file {value!r}; tried {candidates}")


def _cell_identity(cell_dir: Path, config: dict[str, Any]) -> tuple[str, str, str]:
    data_cfg = config["data"]
    dataset = str(data_cfg.get("dataset", data_cfg.get("name", "")))
    if not dataset:
        for part in reversed(cell_dir.parts):
            if part.lower() in {"etth1", "etth2", "weather", "exchange", "exchange_rate"}:
                dataset = part
                break
    if not dataset:
        dataset = Path(str(data_cfg.get("csv_path", "dataset"))).stem
    backbone = str(config["backbone"])
    dataset = _safe(dataset)
    backbone = _safe(backbone)
    group_key = _safe(f"{dataset}_{backbone}")
    return dataset, backbone, group_key


def _find_hash_manifests(cell_dir: Path) -> list[Path]:
    manifests = []
    for ancestor in [cell_dir, *cell_dir.parents]:
        for name in ("data_sha256.txt", "data_hashes.txt", "data_hashes.json"):
            item = ancestor / name
            if item.is_file() and item not in manifests:
                manifests.append(item)
    return manifests


def _read_data_hashes(paths: list[Path]) -> list[dict[str, str]]:
    output = []
    for path in paths:
        if path.suffix == ".json":
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    output.append({str(k): str(v) for k, v in raw.items()})
            except (json.JSONDecodeError, OSError):
                continue
        else:
            row = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                bits = line.strip().split(maxsplit=1)
                if len(bits) == 2:
                    row[bits[1]] = bits[0]
            output.append(row)
    return output


def _metric_labels(history: torch.Tensor, future: torch.Tensor,
                   scales: list[int], epsilon: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(_tensor_to_np(history), dtype=np.float64)
    y = np.asarray(_tensor_to_np(future), dtype=np.float64)
    eps = np.asarray(_tensor_to_np(epsilon), dtype=np.float64)
    joined = np.concatenate((x, y), axis=1)
    origin = x.shape[1] - 1
    steps = np.arange(1, y.shape[1] + 1)
    slopes = np.stack([
        (joined[:, origin + steps, :] - joined[:, origin + steps - scale, :]) / scale
        for scale in scales
    ], axis=1)
    labels = np.where(slopes > eps[None, :, None, :], 2,
                      np.where(slopes < -eps[None, :, None, :], 0, 1)).astype(np.int64)
    truth_delta = np.diff(np.concatenate((x[:, -1:, :], y), axis=1), axis=1)
    return labels, np.sign(truth_delta)


def _pred_labels(history: torch.Tensor, prediction: torch.Tensor,
                 scales: list[int], epsilon: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(_tensor_to_np(history), dtype=np.float64)
    z = np.asarray(_tensor_to_np(prediction), dtype=np.float64)
    eps = np.asarray(_tensor_to_np(epsilon), dtype=np.float64)
    joined = np.concatenate((x, z), axis=1)
    origin = x.shape[1] - 1
    steps = np.arange(1, z.shape[1] + 1)
    slopes = np.stack([
        (joined[:, origin + steps, :] - joined[:, origin + steps - scale, :]) / scale
        for scale in scales
    ], axis=1)
    labels = np.where(slopes > eps[None, :, None, :], 2,
                      np.where(slopes < -eps[None, :, None, :], 0, 1)).astype(np.int64)
    pred_delta = np.diff(np.concatenate((x[:, -1:, :], z), axis=1), axis=1)
    return labels, np.sign(pred_delta)


def _macro_f1_and_accuracy(true: np.ndarray, pred: np.ndarray) -> tuple[float, float, list[list[int]]]:
    true_flat, pred_flat = true.reshape(-1), pred.reshape(-1)
    matrix = np.bincount(true_flat * 3 + pred_flat, minlength=9).reshape(3, 3)
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros(3), where=matrix.sum(axis=0) != 0)
    recall = np.divide(tp, matrix.sum(axis=1), out=np.zeros(3), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(3), where=precision + recall != 0)
    accuracy = float(np.mean(true_flat == pred_flat))
    return accuracy, float(f1.mean()), matrix.astype(int).tolist()


def _score_path(history: torch.Tensor, truth: torch.Tensor, prediction: torch.Tensor,
                true_labels: np.ndarray, true_raw_sign: np.ndarray,
                scales: list[int], epsilon: torch.Tensor, std: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    pred_labels, pred_raw_sign = _pred_labels(history, prediction, scales, epsilon)
    y, z = _tensor_to_np(truth), _tensor_to_np(prediction)
    std_arr = np.asarray(std, dtype=np.float64).reshape(1, 1, -1)
    err = z.astype(np.float64) - y.astype(np.float64)
    accuracy_by_scale: dict[str, float] = {}
    macro_f1_by_scale: dict[str, float] = {}
    confusion_by_scale: dict[str, list[list[int]]] = {}
    per_window_accuracy: dict[str, np.ndarray] = {}
    for index, scale in enumerate(scales):
        accuracy, macro_f1, matrix = _macro_f1_and_accuracy(true_labels[:, index], pred_labels[:, index])
        key = str(scale)
        accuracy_by_scale[key] = accuracy
        macro_f1_by_scale[key] = macro_f1
        confusion_by_scale[key] = matrix
        per_window_accuracy[key] = np.mean(true_labels[:, index] == pred_labels[:, index], axis=(1, 2))
    raw_correct = true_raw_sign == pred_raw_sign
    window = {
        "raw_sign_accuracy": np.mean(raw_correct, axis=(1, 2)),
        "mean_3class_accuracy": np.mean(np.stack(list(per_window_accuracy.values()), axis=1), axis=1),
        "standardized_mse": np.mean(err ** 2, axis=(1, 2)),
        "original_units_mse": np.mean((err * std_arr) ** 2, axis=(1, 2)),
    }
    window.update({f"accuracy_scale_{scale}": per_window_accuracy[str(scale)] for scale in scales})
    metrics = {
        "raw_sign_accuracy": float(np.mean(raw_correct)),
        "standardized_mse": float(np.mean(err ** 2)),
        "original_units_mse": float(np.mean((err * std_arr) ** 2)),
        "accuracy_by_scale": accuracy_by_scale,
        "mean_3class_accuracy": float(np.mean(list(accuracy_by_scale.values()))),
        "macro_f1_by_scale": macro_f1_by_scale,
        "mean_macro_f1": float(np.mean(list(macro_f1_by_scale.values()))),
        "confusion_matrix_by_scale": confusion_by_scale,
    }
    return metrics, window, raw_correct


def _selection_eligible(metrics: dict[str, Any], base: dict[str, Any]) -> bool:
    return bool(
        metrics["raw_sign_accuracy"] > base["raw_sign_accuracy"]
        and metrics["standardized_mse"] <= base["standardized_mse"] + STRICT_MSE_TOLERANCE
    )


def _choose_candidate(rows: list[dict[str, Any]], base_metrics: dict[str, Any],
                      tie_order: list[str], id_field: str = "candidate_id") -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    eligible = [row for row in rows if _selection_eligible(row["metrics"], base_metrics)]
    rank = {name: index for index, name in enumerate(tie_order)}
    eligible.sort(key=lambda row: (
        -row["metrics"]["raw_sign_accuracy"], row["metrics"]["standardized_mse"],
        rank.get(row.get(id_field, ""), len(rank)), row.get("tie_value", 0.0),
    ))
    return (eligible[0] if eligible else None), eligible


def _path_differs(left: torch.Tensor, right: torch.Tensor) -> bool:
    return not np.array_equal(_tensor_to_np(left), _tensor_to_np(right))


def _convex_mix(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
    if weight == 0.0 or not _path_differs(a, b):
        return a
    if weight == 1.0:
        return b
    return (1.0 - weight) * a + weight * b


def _fixed_mean(paths: list[torch.Tensor]) -> torch.Tensor:
    if all(not _path_differs(paths[0], path) for path in paths[1:]):
        return paths[0]
    return sum(paths) / float(len(paths))


def _has_actual_learned_component(alpha: float, learned_path: torch.Tensor,
                                  base_path: torch.Tensor, coefficient: float) -> bool:
    return bool(coefficient > 0.0 and alpha > 0.0 and _path_differs(learned_path, base_path))


def _complementarity(history: torch.Tensor, truth: torch.Tensor,
                     a: torch.Tensor, b: torch.Tensor,
                     a_name: str, b_name: str) -> dict[str, Any]:
    x, y = _tensor_to_np(history), _tensor_to_np(truth)
    pa, pb = _tensor_to_np(a), _tensor_to_np(b)
    true_sign = np.sign(np.diff(np.concatenate((x[:, -1:, :], y), axis=1), axis=1))
    sign_a = np.sign(np.diff(np.concatenate((x[:, -1:, :], pa), axis=1), axis=1))
    sign_b = np.sign(np.diff(np.concatenate((x[:, -1:, :], pb), axis=1), axis=1))
    correct_a, correct_b = sign_a == true_sign, sign_b == true_sign
    residual_a, residual_b = (pa - y).reshape(-1), (pb - y).reshape(-1)
    correlation = None
    if np.std(residual_a) > 0 and np.std(residual_b) > 0:
        correlation = float(np.corrcoef(residual_a, residual_b)[0, 1])
    n = int(correct_a.size)
    return {
        "member_a": a_name,
        "member_b": b_name,
        "direction_point_counts": {
            "only_a_correct": int(np.count_nonzero(correct_a & ~correct_b)),
            "only_b_correct": int(np.count_nonzero(~correct_a & correct_b)),
            "both_wrong": int(np.count_nonzero(~correct_a & ~correct_b)),
            "both_correct": int(np.count_nonzero(correct_a & correct_b)),
            "total": n,
        },
        "direction_point_fractions": {
            "only_a_correct": float(np.mean(correct_a & ~correct_b)),
            "only_b_correct": float(np.mean(~correct_a & correct_b)),
            "both_wrong": float(np.mean(~correct_a & ~correct_b)),
            "both_correct": float(np.mean(correct_a & correct_b)),
        },
        "validation_residual_correlation": correlation,
        "members_identical_path": not _path_differs(a, b),
    }


def _write_candidate_path(npz_data: dict[str, np.ndarray], group: str, seed: int,
                          split: str, candidate_id: str, path: torch.Tensor,
                          probabilities: torch.Tensor | None = None) -> str:
    path_key = _npz_key(group, seed, split, candidate_id, "path")
    npz_data[path_key] = _tensor_to_np(path).astype(np.float32, copy=False)
    if probabilities is not None:
        npz_data[_npz_key(group, seed, split, candidate_id, "probabilities")] = (
            _tensor_to_np(probabilities).astype(np.float32, copy=False)
        )
    return path_key


def _load_seed_arrays(cell_dir: Path, config: dict[str, Any], checkpoint: dict[str, Any],
                      saved_predictions: dict[str, np.ndarray], seed: int,
                      device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    data_cfg = dict(config["data"])
    data_path = _resolve_data_path(cell_dir, str(data_cfg["csv_path"]))
    data_cfg["csv_path"] = str(data_path)
    data_cfg["smoke"] = False
    bundle = load_data(data_cfg)
    scales = list(config["scales"])
    if scales != SCALES:
        raise ValueError(f"expected fixed scales {SCALES}, got {scales} in {cell_dir}")
    lookback, horizon, channels = bundle.train.x.shape[1], bundle.train.y.shape[1], bundle.train.x.shape[2]
    if checkpoint["model_config"]["lookback"] != lookback or checkpoint["model_config"]["horizon"] != horizon or checkpoint["model_config"]["channels"] != channels:
        raise AssertionError("checkpoint shape does not match reconstructed data")
    arrays = {
        "train_x": torch.as_tensor(bundle.train.x, dtype=torch.float32, device=device),
        "train_y": torch.as_tensor(bundle.train.y, dtype=torch.float32, device=device),
        "cal_x": torch.as_tensor(bundle.calibration.x, dtype=torch.float32, device=device),
        "cal_y": torch.as_tensor(bundle.calibration.y, dtype=torch.float32, device=device),
        "val_x": torch.as_tensor(bundle.validation.x, dtype=torch.float32, device=device),
        "val_y": torch.as_tensor(bundle.validation.y, dtype=torch.float32, device=device),
        "test_x": torch.as_tensor(bundle.test.x, dtype=torch.float32, device=device),
        "test_y": torch.as_tensor(bundle.test.y, dtype=torch.float32, device=device),
        "mean": torch.as_tensor(bundle.mean, dtype=torch.float32, device=device),
        "std": torch.as_tensor(bundle.std, dtype=torch.float32, device=device),
    }
    eps = fit_thresholds(arrays["train_x"], arrays["train_y"], scales,
                         quantile=float(config["epsilon_quantile"]))
    arrays["epsilon"] = torch.as_tensor(eps, dtype=torch.float32, device=device)
    if not torch.allclose(arrays["epsilon"].cpu(), checkpoint["epsilon"].cpu(), rtol=0, atol=1e-7):
        raise AssertionError(f"recomputed train epsilon differs for seed {seed}")
    for stat in ("mean", "std"):
        saved_stat = torch.as_tensor(checkpoint["normalization"][stat], dtype=torch.float32)
        if not torch.allclose(arrays[stat].cpu(), saved_stat, rtol=0, atol=1e-6):
            raise AssertionError(f"recomputed train normalization {stat} differs for seed {seed}")
    for field, recreated in (("history", bundle.test.x), ("truth", bundle.test.y),
                             ("origins", bundle.test.origins)):
        saved_key = "origins" if field == "origins" else f"seed{seed}_{field}"
        if saved_key not in saved_predictions or not np.array_equal(recreated, saved_predictions[saved_key]):
            raise AssertionError(f"recreated test {field} differs from saved NPZ for seed {seed}")
    metadata = {
        "data_path": data_path,
        "data_sha256": _sha256(data_path),
        "data_metadata": bundle.metadata,
        "normalization": {"mean": bundle.mean.tolist(), "std": bundle.std.tolist()},
        "epsilon": _tensor_to_np(arrays["epsilon"]).tolist(),
        "scales": scales,
        "split_origins": {
            "validation": bundle.validation.origins.tolist(),
            "test": bundle.test.origins.tolist(),
        },
        "lookback": lookback,
        "horizon": horizon,
        "channels": channels,
    }
    return arrays, metadata


def _seed_model_paths(checkpoint: dict[str, Any], arrays: dict[str, torch.Tensor],
                      device: torch.device, seed: int, saved_predictions: dict[str, np.ndarray]
                      ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    paths: dict[str, torch.Tensor] = {}
    probs: dict[str, torch.Tensor] = {}
    for split in ("validation", "test"):
        x = arrays["val_x"] if split == "validation" else arrays["test_x"]
        base = predict_from_checkpoint(checkpoint, x, "base")
        head = predict_from_checkpoint(checkpoint, x, "direction_head")
        paths[split + "__base"] = base
        probs[split + "__learned_multi"] = head["probabilities"]
        for name in ("difference", "residual", "residual_direction"):
            paths[split + "__" + name] = predict_from_checkpoint(checkpoint, x, name)
        if split == "test":
            expected_keys = {
                "base": f"seed{seed}_base",
                "difference": f"seed{seed}_difference",
                "residual": f"seed{seed}_residual",
                "residual_direction": f"seed{seed}_residual_direction",
                "direction_head": f"seed{seed}_direction_head",
            }
            for name, key in expected_keys.items():
                actual = base if name == "direction_head" else paths[split + "__" + name]
                torch.testing.assert_close(
                    actual, torch.as_tensor(saved_predictions[key], dtype=torch.float32, device=device),
                    rtol=1e-5, atol=1e-6,
                )
            torch.testing.assert_close(
                probs[split + "__learned_multi"],
                torch.as_tensor(saved_predictions[f"seed{seed}_direction_head_probs"], dtype=torch.float32, device=device),
                rtol=1e-5, atol=1e-6,
            )
    return paths, probs


def _prior_probabilities(train_x: torch.Tensor, train_y: torch.Tensor,
                         scales: list[int], epsilon: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_labels = direction_labels(multiscale_slopes(train_x, train_y, scales), epsilon)
    counts = torch.nn.functional.one_hot(train_labels, num_classes=3).sum(dim=0)
    probabilities = counts.float() / counts.sum(dim=-1, keepdim=True).clamp_min(1)
    flat = torch.zeros_like(probabilities)
    flat[..., 1] = 1.0
    return probabilities.unsqueeze(0), flat.unsqueeze(0), counts


def _chosen_alpha(grid: list[dict[str, Any]], base_metrics: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    chosen, eligible = _choose_candidate(grid, base_metrics, [r["candidate_id"] for r in grid])
    if chosen is None:
        zero = next(row for row in grid if row["alpha"] == 0.0)
        return zero, False
    return chosen, True


def _record_candidate(row_id: str, metric: dict[str, Any], prediction_key: str,
                      **extra: Any) -> dict[str, Any]:
    return {"candidate_id": row_id, "prediction_key": prediction_key,
            "metrics": metric, **extra}


def _mean_std_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("raw_sign_accuracy", "standardized_mse", "original_units_mse",
            "mean_3class_accuracy", "mean_macro_f1")
    output = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        output[key] = {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "per_seed": values,
        }
    for metric_name in ("accuracy_by_scale", "macro_f1_by_scale"):
        scales = rows[0][metric_name]
        output[metric_name] = {
            scale: {
                "mean": float(np.mean([row[metric_name][scale] for row in rows])),
                "sample_std": float(np.std([row[metric_name][scale] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            }
            for scale in scales
        }
    return output


def _per_seed_cell(cell_dir: Path, group_key: str, dataset: str, backbone: str,
                   config: dict[str, Any], seed_checkpoint: dict[str, Any], seed: int,
                   saved_predictions: dict[str, np.ndarray], device: torch.device,
                   npz_data: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays, data_info = _load_seed_arrays(cell_dir, config, seed_checkpoint, saved_predictions, seed, device)
    scales = list(config["scales"])
    epsilon = arrays["epsilon"]
    std = _tensor_to_np(arrays["std"])
    model_paths, learned_probs = _seed_model_paths(
        seed_checkpoint, arrays, device, seed, saved_predictions
    )
    prior_prob, flat_prob, prior_counts = _prior_probabilities(
        arrays["train_x"], arrays["train_y"], scales, epsilon
    )
    probabilities = {
        "learned_multi": learned_probs["validation__learned_multi"],
        "learned_single": learned_probs["validation__learned_multi"][:, :1],
        "prior": prior_prob,
        "flat": flat_prob,
    }
    test_probabilities = {
        "learned_multi": learned_probs["test__learned_multi"],
        "learned_single": learned_probs["test__learned_multi"][:, :1],
        "prior": prior_prob,
        "flat": flat_prob,
    }
    val_x, val_y = arrays["val_x"], arrays["val_y"]
    test_x, test_y = arrays["test_x"], arrays["test_y"]
    val_true_labels, val_true_sign = _metric_labels(val_x, val_y, scales, epsilon)

    # Shared split arrays and protocol metadata needed for independent recomputation.
    for split, x, y, origins in (
        ("validation", val_x, val_y, np.asarray(data_info["split_origins"]["validation"], dtype=np.int64)),
        ("test", test_x, test_y, np.asarray(data_info["split_origins"]["test"], dtype=np.int64)),
    ):
        npz_data[_npz_key(group_key, seed, split, "split", "history")] = _tensor_to_np(x).astype(np.float32, copy=False)
        npz_data[_npz_key(group_key, seed, split, "split", "truth")] = _tensor_to_np(y).astype(np.float32, copy=False)
        npz_data[_npz_key(group_key, seed, split, "split", "origins")] = origins
    npz_data[f"{group_key}__train_prior_class_counts"] = _tensor_to_np(prior_counts).astype(np.int64, copy=False)
    npz_data[f"{group_key}__prior_probabilities"] = _tensor_to_np(prior_prob[0]).astype(np.float32, copy=False)
    npz_data[f"{group_key}__flat_probabilities"] = _tensor_to_np(flat_prob[0]).astype(np.float32, copy=False)
    npz_data[f"{group_key}__scales"] = np.asarray(scales, dtype=np.int64)
    npz_data[f"{group_key}__epsilon"] = _tensor_to_np(epsilon).astype(np.float32, copy=False)
    npz_data[f"{group_key}__normalization_mean"] = _tensor_to_np(arrays["mean"]).astype(np.float32, copy=False)
    npz_data[f"{group_key}__normalization_std"] = _tensor_to_np(arrays["std"]).astype(np.float32, copy=False)

    val_prediction_paths: dict[str, torch.Tensor] = {
        method: model_paths["validation__" + method]
        for method in ("base", "difference", "residual", "residual_direction")
    }
    test_prediction_paths: dict[str, torch.Tensor] = {
        method: model_paths["test__" + method]
        for method in ("base", "difference", "residual", "residual_direction")
    }
    val_metrics_cache: dict[str, dict[str, Any]] = {}
    val_windows_cache: dict[str, dict[str, np.ndarray]] = {}
    val_raw_correct_cache: dict[str, np.ndarray] = {}
    for method, path in val_prediction_paths.items():
        _write_candidate_path(npz_data, group_key, seed, "validation", method, path)
    validation_probability_keys = {}
    for atom, probability in probabilities.items():
        probability_key = _npz_key(group_key, seed, "validation", f"probabilities__{atom}", "probabilities")
        npz_data[probability_key] = _tensor_to_np(probability).astype(np.float32, copy=False)
        validation_probability_keys[atom] = probability_key
    base_val_metrics, base_window, base_raw_correct = _score_path(
        val_x, val_y, val_prediction_paths["base"], val_true_labels, val_true_sign, scales, epsilon, std
    )
    val_metrics_cache["base"], val_windows_cache["base"], val_raw_correct_cache["base"] = (
        base_val_metrics, base_window, base_raw_correct
    )

    validation_grids: dict[str, Any] = {"atom_alpha_grids": {}, "standalone_candidates": [],
                                        "pairwise_weight_grids": {}, "fixed_threeway_mean": None,
                                        "fusion_selected_candidates": []}
    atom_selection: dict[str, Any] = {}
    atom_selected_paths: dict[str, torch.Tensor] = {}
    atom_selected_rows: dict[str, dict[str, Any]] = {}
    for atom in ATOM_NAMES:
        atom_scales = scales if atom != "learned_single" else [1]
        atom_eps = epsilon if atom != "learned_single" else epsilon[:1]
        probs = probabilities[atom]
        grid_rows: list[dict[str, Any]] = []
        grid_paths: dict[str, torch.Tensor] = {}
        grid_metrics: dict[str, dict[str, Any]] = {}
        for alpha in ALPHA_GRID:
            candidate_id = f"atom__{atom}__alpha_{_number_token(alpha)}"
            projection = val_prediction_paths["base"] if alpha == 0.0 else project_trajectory(
                val_x, val_prediction_paths["base"], probs.expand(len(val_x), -1, -1, -1, -1),
                atom_scales, atom_eps, strength=alpha,
                steps=PROJECTION_STEPS, max_rms=MAX_RMS,
            )
            metric, window, raw_correct = _score_path(
                val_x, val_y, projection, val_true_labels, val_true_sign, scales, epsilon, std
            )
            val_prediction_paths[candidate_id] = projection
            grid_paths[candidate_id] = projection
            grid_metrics[candidate_id] = metric
            val_metrics_cache[candidate_id] = metric
            val_windows_cache[candidate_id] = window
            val_raw_correct_cache[candidate_id] = raw_correct
            key = _write_candidate_path(npz_data, group_key, seed, "validation", candidate_id, projection)
            row = _record_candidate(
                candidate_id, metric, key, alpha=alpha,
                probability_key=validation_probability_keys[atom],
                eligible=_selection_eligible(metric, base_val_metrics),
                is_no_op=not _path_differs(projection, val_prediction_paths["base"]),
                scale_set=atom_scales,
            )
            grid_rows.append(row)
        selected, has_feasible = _chosen_alpha(grid_rows, base_val_metrics)
        selected_id = selected["candidate_id"]
        atom_selection[atom] = {
            "selected_candidate_id": selected_id,
            "prediction_key": selected["prediction_key"],
            "selected_alpha": float(selected["alpha"]),
            "eligible": bool(has_feasible),
            "fallback_to_alpha_zero": not has_feasible,
            "is_no_op": bool(selected["is_no_op"]),
            "selection_metric": "raw_sign_accuracy",
            "mse_constraint": f"standardized_mse <= base + {STRICT_MSE_TOLERANCE:g}",
            "tie_break": "raw-sign accuracy descending, standardized MSE ascending, alpha ascending",
        }
        atom_selected_paths[atom] = grid_paths[selected_id]
        atom_selected_rows[atom] = selected
        validation_grids["atom_alpha_grids"][atom] = grid_rows

    # Materialize selected atoms under stable method IDs for downstream selections.
    selected_atom_ids: dict[str, str] = {}
    for atom in ATOM_NAMES:
        row = atom_selected_rows[atom]
        selected_atom_ids[atom] = row["candidate_id"]
        val_prediction_paths[atom] = atom_selected_paths[atom]
        val_metrics_cache[atom] = row["metrics"]
        val_windows_cache[atom] = val_windows_cache[row["candidate_id"]]
        val_raw_correct_cache[atom] = val_raw_correct_cache[row["candidate_id"]]
        atom_alias_key = _write_candidate_path(
            npz_data, group_key, seed, "validation", atom, atom_selected_paths[atom]
        )
        atom_selection[atom]["source_prediction_key"] = atom_selection[atom]["prediction_key"]
        atom_selection[atom]["prediction_key"] = atom_alias_key

    # Select the two learned multi-scale/single-scale alternatives on validation.
    learned_rows = []
    for atom in ("learned_single", "learned_multi"):
        alpha_row = atom_selected_rows[atom]
        learned_rows.append({
            "candidate_id": atom,
            "source_candidate_id": selected_atom_ids[atom],
            "prediction_key": alpha_row["prediction_key"],
            "metrics": atom_selected_rows[atom]["metrics"],
            "alpha": atom_selected_rows[atom]["alpha"],
            "eligible": bool(atom_selection[atom]["eligible"]),
            "fallback_to_alpha_zero": bool(atom_selection[atom]["fallback_to_alpha_zero"]),
            "tie_value": 0 if atom == "learned_single" else 1,
        })
    learned_rows.sort(key=lambda row: (
        -row["metrics"]["raw_sign_accuracy"], row["metrics"]["standardized_mse"], row["tie_value"]
    ))
    learned_choice = learned_rows[0]
    learned_selected_source = learned_choice["source_candidate_id"]
    val_prediction_paths["learned_selected"] = val_prediction_paths[learned_selected_source]
    val_metrics_cache["learned_selected"] = learned_choice["metrics"]
    val_windows_cache["learned_selected"] = val_windows_cache[learned_selected_source]
    val_raw_correct_cache["learned_selected"] = val_raw_correct_cache[learned_selected_source]
    learned_selected_key = _write_candidate_path(
        npz_data, group_key, seed, "validation", "learned_selected",
        val_prediction_paths["learned_selected"],
    )

    # Numeric methods and selected atoms form the frozen standalone set.
    standalone_rows: list[dict[str, Any]] = []
    for name in STANDALONE_ORDER:
        source_candidate_id = selected_atom_ids[name] if name in ATOM_NAMES else name
        path = val_prediction_paths[source_candidate_id]
        metric = val_metrics_cache[source_candidate_id] if source_candidate_id in val_metrics_cache else None
        if metric is None:
            metric, window, raw_correct = _score_path(
                val_x, val_y, path, val_true_labels, val_true_sign, scales, epsilon, std
            )
            val_metrics_cache[source_candidate_id] = metric
            val_windows_cache[source_candidate_id] = window
            val_raw_correct_cache[source_candidate_id] = raw_correct
        key = _npz_key(group_key, seed, "validation", source_candidate_id, "path")
        standalone_rows.append(_record_candidate(
            name, metric, key,
            source_candidate_id=source_candidate_id,
            eligible=_selection_eligible(metric, base_val_metrics),
            is_no_op=not _path_differs(path, val_prediction_paths["base"]),
        ))
    validation_grids["standalone_candidates"] = standalone_rows
    standalone_choice, standalone_eligible = _choose_candidate(
        standalone_rows, base_val_metrics, STANDALONE_ORDER
    )
    standalone_fallback = standalone_choice is None
    if standalone_fallback:
        standalone_choice = next(row for row in standalone_rows if row["candidate_id"] == "base")
    standalone_source = standalone_choice["source_candidate_id"]
    val_prediction_paths["standalone_selected"] = val_prediction_paths[standalone_source]
    val_metrics_cache["standalone_selected"] = standalone_choice["metrics"]
    val_windows_cache["standalone_selected"] = val_windows_cache[standalone_source]
    val_raw_correct_cache["standalone_selected"] = val_raw_correct_cache[standalone_source]
    standalone_selected_key = _write_candidate_path(
        npz_data, group_key, seed, "validation", "standalone_selected",
        val_prediction_paths["standalone_selected"],
    )

    # Pairwise fusion weights are chosen only after each member is frozen.
    selected_pair_rows: dict[str, dict[str, Any]] = {}
    pair_test_specs: dict[str, dict[str, Any]] = {}
    pair_complementarity: dict[str, Any] = {}
    for pair_name, member_a, member_b in PAIR_SPECS:
        a = val_prediction_paths[member_a]
        b = val_prediction_paths[member_b]
        a_id = selected_atom_ids[member_a] if member_a in selected_atom_ids else member_a
        b_id = selected_atom_ids[member_b] if member_b in selected_atom_ids else member_b
        if member_a == "learned_selected":
            a_id = learned_selected_source
        rows: list[dict[str, Any]] = []
        for weight in WEIGHT_GRID:
            candidate_id = f"pair__{pair_name}__w_{_number_token(weight)}"
            mixed = _convex_mix(a, b, weight)
            metric, window, raw_correct = _score_path(
                val_x, val_y, mixed, val_true_labels, val_true_sign, scales, epsilon, std
            )
            val_prediction_paths[candidate_id] = mixed
            val_metrics_cache[candidate_id] = metric
            val_windows_cache[candidate_id] = window
            val_raw_correct_cache[candidate_id] = raw_correct
            key = _write_candidate_path(npz_data, group_key, seed, "validation", candidate_id, mixed)
            rows.append(_record_candidate(
                candidate_id, metric, key,
                pair_name=pair_name, member_a=member_a, member_a_candidate_id=a_id,
                member_b=member_b, member_b_candidate_id=b_id, weight=float(weight),
                eligible=_selection_eligible(metric, base_val_metrics),
                is_interior=bool(0.0 < weight < 1.0),
                members_different_path=_path_differs(a, b),
                learned_coefficient=float((1.0 - weight) if member_a == "learned_selected" else
                                          (weight if member_b == "learned_selected" else 0.0)),
                includes_actual_learned_component=_has_actual_learned_component(
                    float(atom_selection[learned_choice["candidate_id"]]["selected_alpha"]),
                    val_prediction_paths["learned_selected"], val_prediction_paths["base"],
                    float((1.0 - weight) if member_a == "learned_selected" else
                          (weight if member_b == "learned_selected" else 0.0)),
                ),
                is_nontrivial_blend=bool(0.0 < weight < 1.0 and _path_differs(a, b)),
                tie_value=float(weight),
            ))
        chosen, eligible_rows = _choose_candidate(rows, base_val_metrics, [r["candidate_id"] for r in rows])
        fallback = chosen is None
        if fallback:
            selected_pair_rows[pair_name] = {
                "candidate_id": "base", "prediction_key": _npz_key(group_key, seed, "validation", "base", "path"),
                "metrics": base_val_metrics, "member_a": member_a, "member_a_candidate_id": a_id,
                "member_b": member_b, "member_b_candidate_id": b_id, "weight": None,
                "eligible": False, "fallback_to_base": True, "is_interior": False,
                "members_different_path": _path_differs(a, b), "is_nontrivial_blend": False,
                "includes_actual_learned_component": False,
            }
        else:
            selected_pair_rows[pair_name] = {**chosen, "fallback_to_base": False}
        pair_test_specs[pair_name] = {
            "member_a": member_a, "member_a_candidate_id": a_id,
            "member_b": member_b, "member_b_candidate_id": b_id,
            "selected_weight": selected_pair_rows[pair_name]["weight"],
            "selected_candidate_id": selected_pair_rows[pair_name]["candidate_id"],
            "fallback_to_base": fallback,
        }
        validation_grids["pairwise_weight_grids"][pair_name] = rows
        pair_complementarity[pair_name] = _complementarity(
            val_x, val_y, a, b, member_a, member_b
        )

    # Fixed equal-weight three-way mean; no weight search.
    triple_members = ("learned_selected", "flat", "difference")
    triple_path = _fixed_mean([val_prediction_paths[name] for name in triple_members])
    triple_id = "fixedmean__learned_selected_flat_difference"
    triple_metric, triple_window, triple_raw_correct = _score_path(
        val_x, val_y, triple_path, val_true_labels, val_true_sign, scales, epsilon, std
    )
    triple_unique_member_paths = len({
        _tensor_to_np(val_prediction_paths[name]).tobytes() for name in triple_members
    })
    val_prediction_paths[triple_id] = triple_path
    val_metrics_cache[triple_id] = triple_metric
    val_windows_cache[triple_id] = triple_window
    val_raw_correct_cache[triple_id] = triple_raw_correct
    triple_key = _write_candidate_path(npz_data, group_key, seed, "validation", triple_id, triple_path)
    triple_row = _record_candidate(
        triple_id, triple_metric, triple_key,
        members=list(triple_members), weights=[1 / 3, 1 / 3, 1 / 3],
        eligible=_selection_eligible(triple_metric, base_val_metrics),
        members_pairwise_different=all(
            _path_differs(val_prediction_paths[a], val_prediction_paths[b])
            for index, a in enumerate(triple_members) for b in triple_members[index + 1:]
        ),
        unique_member_path_count=triple_unique_member_paths,
        at_least_two_distinct_members=triple_unique_member_paths >= 2,
        includes_actual_learned_component=_has_actual_learned_component(
            float(atom_selection[learned_choice["candidate_id"]]["selected_alpha"]),
            val_prediction_paths["learned_selected"], val_prediction_paths["base"], 1.0 / 3.0,
        ),
        is_nontrivial_blend=triple_unique_member_paths >= 2,
    )
    triple_fallback = not bool(triple_row["eligible"])
    triple_row["deployed_candidate_id"] = "base" if triple_fallback else triple_id
    triple_row["fallback_to_base"] = triple_fallback
    validation_grids["fixed_threeway_mean"] = triple_row

    # Select among the six validation-selected pair outputs and the fixed mean.
    fusion_rows = []
    fusion_selected_paths: dict[str, torch.Tensor] = {}
    for pair_name in FUSION_ORDER:
        if pair_name == "threeway_mean":
            row = triple_row
            path = val_prediction_paths["base"] if triple_fallback else triple_path
            row_id = "base" if triple_fallback else triple_id
        else:
            row = selected_pair_rows[pair_name]
            path = (val_prediction_paths[row["candidate_id"]]
                    if row["candidate_id"] != "base" else val_prediction_paths["base"])
            row_id = row["candidate_id"]
        deployed_metrics = (
            base_val_metrics
            if pair_name == "threeway_mean" and triple_fallback
            else row["metrics"]
        )
        deployed_eligible = bool(
            pair_name != "threeway_mean" or not triple_fallback
        ) and _selection_eligible(deployed_metrics, base_val_metrics)
        fusion_selected_paths[pair_name] = path
        fusion_rows.append({
            "candidate_id": pair_name,
            "source_candidate_id": row_id,
            "prediction_key": _npz_key(group_key, seed, "validation", row_id, "path"),
            "metrics": deployed_metrics,
            "eligible": deployed_eligible,
            "fallback_to_base": bool(triple_fallback if pair_name == "threeway_mean" else row.get("fallback_to_base", False)),
            "is_nontrivial_blend": bool(deployed_eligible and row.get("is_nontrivial_blend", False)),
            "includes_actual_learned_component": bool(
                deployed_eligible and row.get("includes_actual_learned_component", False)
            ),
            "pair_name": pair_name if pair_name != "threeway_mean" else None,
        })
    validation_grids["fusion_selected_candidates"] = fusion_rows
    fusion_choice, fusion_eligible = _choose_candidate(
        fusion_rows, base_val_metrics, FUSION_ORDER
    )
    fusion_fallback = fusion_choice is None
    if fusion_fallback:
        fusion_choice = {
            "candidate_id": "base", "source_candidate_id": "base", "metrics": base_val_metrics,
            "eligible": False, "fallback_to_base": True, "is_nontrivial_blend": False,
            "includes_actual_learned_component": False,
        }
    fusion_choice_source = fusion_choice["source_candidate_id"]
    if fusion_choice_source == "base":
        fusion_path = val_prediction_paths["base"]
    elif fusion_choice_source == triple_id:
        fusion_path = triple_path
    else:
        fusion_path = val_prediction_paths[fusion_choice_source]
    fusion_selected_key = _write_candidate_path(
        npz_data, group_key, seed, "validation", "fusion_selected", fusion_path
    )

    # Selected test paths are first produced after all validation choices are frozen.
    selected_alphas = {name: float(atom_selection[name]["selected_alpha"]) for name in ATOM_NAMES}
    test_atom_paths: dict[str, torch.Tensor] = {}
    for atom in ATOM_NAMES:
        atom_scales = scales if atom != "learned_single" else [1]
        atom_eps = epsilon if atom != "learned_single" else epsilon[:1]
        prob = test_probabilities[atom]
        path = test_prediction_paths["base"] if selected_alphas[atom] == 0.0 else project_trajectory(
            test_x, test_prediction_paths["base"], prob.expand(len(test_x), -1, -1, -1, -1),
            atom_scales, atom_eps, strength=selected_alphas[atom],
            steps=PROJECTION_STEPS, max_rms=MAX_RMS,
        )
        test_atom_paths[atom] = path
        test_prediction_paths[atom] = path
    learned_selected_test = test_atom_paths[learned_choice["candidate_id"]]
    test_prediction_paths["learned_selected"] = learned_selected_test
    standalone_selected_test = (
        test_atom_paths[standalone_choice["candidate_id"]]
        if standalone_choice["candidate_id"] in ATOM_NAMES
        else test_prediction_paths[standalone_source]
    )
    test_prediction_paths["standalone_selected"] = standalone_selected_test

    # Finalize all test candidates at their validation-selected weights.
    selected_test_methods: dict[str, torch.Tensor] = {
        name: test_prediction_paths[name]
        for name in ("base", "difference", "residual", "residual_direction",
                     "learned_single", "learned_multi", "prior", "flat",
                     "learned_selected", "standalone_selected")
    }
    selected_test_candidate_ids: dict[str, str] = {
        **{name: atom_selection[name]["selected_candidate_id"] for name in ATOM_NAMES},
        "learned_selected": learned_selected_source,
        "standalone_selected": standalone_source,
        **{name: name for name in ("base", "difference", "residual", "residual_direction")},
    }
    test_pair_paths: dict[str, torch.Tensor] = {}
    for pair_name, member_a, member_b in PAIR_SPECS:
        selected = selected_pair_rows[pair_name]
        if selected["candidate_id"] == "base":
            path = test_prediction_paths["base"]
            candidate_id = "base"
        else:
            weight = float(selected["weight"])
            path = _convex_mix(test_prediction_paths[member_a], test_prediction_paths[member_b], weight)
            candidate_id = selected["candidate_id"]
        test_pair_paths[pair_name] = path
        selected_test_methods[pair_name] = path
        selected_test_candidate_ids[pair_name] = candidate_id
    raw_test_triple_path = _fixed_mean([
        test_prediction_paths["learned_selected"], test_prediction_paths["flat"],
        test_prediction_paths["difference"],
    ])
    test_triple_path = test_prediction_paths["base"] if triple_fallback else raw_test_triple_path
    selected_test_methods["threeway_mean"] = test_triple_path
    selected_test_candidate_ids["threeway_mean"] = "base" if triple_fallback else triple_id
    fusion_selected_test = (
        test_prediction_paths["base"] if fusion_choice_source == "base"
        else raw_test_triple_path if fusion_choice_source == triple_id
        else test_pair_paths[next(name for name, row in selected_pair_rows.items()
                                  if row["candidate_id"] == fusion_choice_source)]
    )
    selected_test_methods["fusion_selected"] = fusion_selected_test
    selected_test_candidate_ids["fusion_selected"] = fusion_choice_source

    # Re-evaluate only frozen test outputs; no test metric enters any selection.
    test_true_labels, test_true_sign = _metric_labels(test_x, test_y, scales, epsilon)
    test_metrics: dict[str, Any] = {}
    test_windows: dict[str, dict[str, np.ndarray]] = {}
    for method, path in selected_test_methods.items():
        metric, window, _ = _score_path(
            test_x, test_y, path, test_true_labels, test_true_sign, scales, epsilon, std
        )
        test_metrics[method] = {
            "candidate_id": selected_test_candidate_ids[method],
            "prediction_key": _write_candidate_path(
                npz_data, group_key, seed, "test", selected_test_candidate_ids[method], path,
                test_probabilities.get(method),
            ),
            "metrics": metric,
        }
        # Aliases have explicit paths for audit and are never silently overwritten.
        if method in ("learned_selected", "standalone_selected", "fusion_selected"):
            alias_key = _write_candidate_path(npz_data, group_key, seed, "test", method, path)
            test_metrics[method]["alias_prediction_key"] = alias_key
        test_windows[method] = window

    test_base_metrics = test_metrics["base"]["metrics"]
    for method in selected_test_methods:
        method_metrics = test_metrics[method]["metrics"]
        method_metrics["strict_joint_pass"] = _selection_eligible(method_metrics, test_base_metrics)
        method_metrics["strict_user_joint_pass"] = method_metrics["strict_joint_pass"]
        method_metrics["relative_to_base"] = _metric_delta(test_windows[method], test_windows["base"])
        method_metrics["relative_to_validation_selected_standalone"] = _metric_delta(
            test_windows[method], test_windows["standalone_selected"]
        )

    # Each pair's chosen blend and fixed mean are also evaluated against their members.
    val_pair_complementarity = pair_complementarity
    pair_test_comparisons = {}
    for pair_name, member_a, member_b in PAIR_SPECS:
        pair_path = test_pair_paths[pair_name]
        pair_test_comparisons[pair_name] = {
            "member_a": member_a,
            "member_b": member_b,
            "relative_to_member_a": _metric_delta(test_windows[pair_name], test_windows[member_a]),
            "relative_to_member_b": _metric_delta(test_windows[pair_name], test_windows[member_b]),
            "is_interior_nonidentical_blend": bool(
                selected_pair_rows[pair_name].get("is_interior", False)
                and selected_pair_rows[pair_name].get("members_different_path", False)
            ),
            "includes_actual_learned_component": bool(
                selected_pair_rows[pair_name].get("includes_actual_learned_component", False)
            ),
            "is_true_learned_fusion": bool(
                selected_pair_rows[pair_name].get("is_interior", False)
                and selected_pair_rows[pair_name].get("members_different_path", False)
                and selected_pair_rows[pair_name].get("includes_actual_learned_component", False)
            ),
            "fallback_to_base": bool(selected_pair_rows[pair_name].get("fallback_to_base", False)),
        }
    pair_test_comparisons["threeway_mean"] = {
        "members": list(triple_members),
        "relative_to_each_member": {
            member: _metric_delta(test_windows["threeway_mean"], test_windows[member])
            for member in triple_members
        },
        "members_pairwise_different": bool(triple_row["members_pairwise_different"]),
        "unique_member_path_count": triple_unique_member_paths,
        "at_least_two_distinct_members": triple_unique_member_paths >= 2,
        "includes_actual_learned_component": bool(triple_row["includes_actual_learned_component"]),
        "is_true_learned_fusion": bool(
            not triple_fallback
            and triple_unique_member_paths >= 2
            and triple_row["includes_actual_learned_component"]
        ),
        "fallback_to_base": triple_fallback,
    }

    for method, path in selected_test_methods.items():
        key = test_metrics[method]["prediction_key"]
        # Alias entries are present for all selected test outcomes under the exact
        # selected candidate ID; this keeps JSON prediction_key and NPZ schema aligned.
        if key not in npz_data:
            raise AssertionError(f"missing NPZ path for {method}: {key}")

    val_complementarity = {}
    for pair_name, member_a, member_b in PAIR_SPECS:
        val_complementarity[pair_name] = _complementarity(
            val_x, val_y, val_prediction_paths[member_a], val_prediction_paths[member_b], member_a, member_b
        )

    validation_selection = {
        "reference": "base",
        "gate": f"raw_sign_accuracy > base AND standardized_mse <= base + {STRICT_MSE_TOLERANCE:g}",
        "candidate_rank": "raw-sign accuracy descending, standardized MSE ascending, then the explicit fixed protocol order",
        "atom_selection": atom_selection,
        "learned_selected": {
            "selected_member": learned_choice["candidate_id"],
            "source_candidate_id": learned_selected_source,
            "selected_alpha": selected_alphas[learned_choice["candidate_id"]],
            "prediction_key": learned_selected_key,
            "validation_metrics": learned_choice["metrics"],
            "members_compared": ["learned_single", "learned_multi"],
            "tie_break": "highest raw-sign accuracy, then lowest standardized MSE, then learned_single",
            "is_no_op_vs_base": not _path_differs(
                val_prediction_paths["learned_selected"], val_prediction_paths["base"]
            ),
            "includes_actual_learned_component": _has_actual_learned_component(
                selected_alphas[learned_choice["candidate_id"]],
                val_prediction_paths["learned_selected"], val_prediction_paths["base"], 1.0,
            ),
        },
        "standalone_selected": {
            "selected_member": standalone_choice["candidate_id"],
            "source_candidate_id": standalone_source,
            "prediction_key": standalone_selected_key,
            "eligible_candidate_ids": [row["candidate_id"] for row in standalone_eligible],
            "fallback_to_base": standalone_fallback,
            "includes_actual_learned_component": bool(
                standalone_choice["candidate_id"] in ("learned_single", "learned_multi")
                and _has_actual_learned_component(
                    float(atom_selection[standalone_choice["candidate_id"]]["selected_alpha"]),
                    val_prediction_paths[standalone_source], val_prediction_paths["base"], 1.0,
                )
            ),
            "is_no_op_vs_base": not _path_differs(
                val_prediction_paths["standalone_selected"], val_prediction_paths["base"]
            ),
            "tie_break_order": STANDALONE_ORDER,
        },
        "pairwise_fusions": {
            pair: {
                "selected_candidate_id": row["candidate_id"],
                "selected_weight": row["weight"],
                "member_a": row["member_a"],
                "member_a_candidate_id": row["member_a_candidate_id"],
                "member_b": row["member_b"],
                "member_b_candidate_id": row["member_b_candidate_id"],
                "eligible_candidate_ids": [candidate["candidate_id"] for candidate in
                                             validation_grids["pairwise_weight_grids"][pair]
                                             if candidate["eligible"]],
                "fallback_to_base": bool(row["fallback_to_base"]),
                "is_endpoint": row["weight"] in (0.0, 1.0) if row["weight"] is not None else False,
                "members_different_path": bool(row["members_different_path"]),
                "includes_actual_learned_component": bool(row["includes_actual_learned_component"]),
                "is_true_learned_fusion": bool(
                    row.get("is_interior", False) and row["members_different_path"]
                    and row["includes_actual_learned_component"]
                ),
                "validation_metrics": row["metrics"],
                "tie_break": "raw-sign accuracy descending, standardized MSE ascending, w ascending",
            }
            for pair, row in selected_pair_rows.items()
        },
        "fixed_threeway_mean": {
            "candidate_id": triple_id,
            "deployed_candidate_id": "base" if triple_fallback else triple_id,
            "members": list(triple_members),
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "prediction_key": triple_key,
            "eligible": triple_row["eligible"],
            "fallback_to_base": triple_fallback,
            "members_pairwise_different": triple_row["members_pairwise_different"],
            "unique_member_path_count": triple_unique_member_paths,
            "at_least_two_distinct_members": triple_unique_member_paths >= 2,
            "includes_actual_learned_component": triple_row["includes_actual_learned_component"],
            "is_nontrivial_blend": triple_row["is_nontrivial_blend"],
            "is_true_learned_fusion": bool(
                not triple_fallback and triple_row["is_nontrivial_blend"]
                and triple_row["includes_actual_learned_component"]
            ),
            "validation_metrics": triple_metric,
        },
        "fusion_selected": {
            "selected_scheme": fusion_choice["candidate_id"],
            "source_candidate_id": fusion_choice_source,
            "prediction_key": fusion_selected_key,
            "eligible_candidate_ids": [row["candidate_id"] for row in fusion_eligible],
            "fallback_to_base": fusion_fallback,
            "is_nontrivial_blend": bool(fusion_choice.get("is_nontrivial_blend", False)),
            "includes_actual_learned_component": bool(fusion_choice.get("includes_actual_learned_component", False)),
            "tie_break_order": FUSION_ORDER,
        },
    }
    validation_grids["learned_selected_candidates"] = learned_rows

    # Per-window paired deltas are retained as audit/bootstrap inputs.
    test_window_refs: dict[str, dict[str, np.ndarray]] = {}
    for method, window in test_windows.items():
        test_window_refs[method] = window
    return {
        "seed": seed,
        "validation_selection": validation_selection,
        "validation_grids": validation_grids,
        "validation_metrics": {
            "base": base_val_metrics,
            "selected_atoms": {name: atom_selected_rows[name]["metrics"] for name in ATOM_NAMES},
            "learned_selected": learned_choice["metrics"],
            "standalone_selected": standalone_choice["metrics"],
        },
        "validation_complementarity": val_complementarity,
        "test_metrics": test_metrics,
        "test_comparisons_to_fusion_members": pair_test_comparisons,
        "test_selected_identity": {
            "selected_test_candidate_ids": selected_test_candidate_ids,
            "fusion_selected_candidate_id": fusion_choice_source,
            "fusion_selected_scheme": fusion_choice["candidate_id"],
            "fusion_selected_is_nontrivial": bool(fusion_choice.get("is_nontrivial_blend", False)),
            "fusion_selected_includes_actual_learned_component": bool(
                fusion_choice.get("includes_actual_learned_component", False)
            ),
            "threeway_mean_fallback_to_base": triple_fallback,
        },
        "reconstruction": {
            "checkpoint_weights_only_cpu": True,
            "test_model_paths_match_saved_npz_rtol_1e_5_atol_1e_6": True,
            "test_windows_origins_truth_match_saved_npz": True,
            "cell_dir": str(cell_dir),
            "data": data_info,
            "train_prior_label_source": "train split windows only; labels from direction_labels(multiscale_slopes(train_x, train_y), epsilon)",
        },
    }, test_window_refs


def _metric_delta(candidate_window: dict[str, np.ndarray], reference_window: dict[str, np.ndarray]) -> dict[str, Any]:
    output = {}
    for key in candidate_window:
        diff = np.asarray(candidate_window[key], dtype=np.float64) - np.asarray(reference_window[key], dtype=np.float64)
        output[key + "_mean_delta"] = float(diff.mean())
        if key in ("standardized_mse", "original_units_mse"):
            ref_mean = float(np.asarray(reference_window[key], dtype=np.float64).mean())
            cand_mean = float(np.asarray(candidate_window[key], dtype=np.float64).mean())
            output[key + "_percent_change"] = (
                float(100.0 * (cand_mean / ref_mean - 1.0)) if ref_mean != 0.0 else None
            )
    return output


def _bootstrap_group(group_key: str, seed_results: list[dict[str, Any]],
                     seed_windows: list[dict[str, dict[str, np.ndarray]]],
                     output_npz: dict[str, np.ndarray]) -> dict[str, Any]:
    comparisons: dict[str, dict[str, Any]] = {}
    method_names = list(seed_windows[0])
    comparison_specs: list[tuple[str, str, str]] = []
    for method in method_names:
        if method == "base":
            continue
        comparison_specs.append((f"{method}__vs__base", method, "base"))
        comparison_specs.append((f"{method}__vs__standalone_selected", method, "standalone_selected"))
    for pair_name, member_a, member_b in PAIR_SPECS:
        comparison_specs.append((f"{pair_name}__vs__{member_a}", pair_name, member_a))
        comparison_specs.append((f"{pair_name}__vs__{member_b}", pair_name, member_b))
    comparison_specs.extend([
        (f"threeway_mean__vs__{member}", "threeway_mean", member)
        for member in ("learned_selected", "flat", "difference")
    ])
    seen = set()
    comparison_specs = [row for row in comparison_specs if not (row[0] in seen or seen.add(row[0]))]
    metrics = list(seed_windows[0]["base"])
    bootstrap_indices = _bootstrap_indices(len(seed_windows[0]["base"][metrics[0]]))
    for comparison_id, candidate, reference in comparison_specs:
        if any(candidate not in windows or reference not in windows for windows in seed_windows):
            continue
        metric_intervals = {}
        for metric in metrics:
            per_seed = [
                np.asarray(windows[candidate][metric], dtype=np.float64)
                - np.asarray(windows[reference][metric], dtype=np.float64)
                for windows in seed_windows
            ]
            if len({len(row) for row in per_seed}) != 1:
                continue
            seed_mean_delta = np.stack(per_seed, axis=0).mean(axis=0)
            array_key = f"{group_key}__{comparison_id}__seedmean_window_delta__{metric}"
            output_npz[array_key] = seed_mean_delta.astype(np.float64)
            for run, per_seed_delta in zip(seed_results, per_seed):
                output_npz[
                    f"{group_key}__seed{run['seed']}__{comparison_id}__window_delta__{metric}"
                ] = per_seed_delta.astype(np.float64)
            metric_intervals[metric] = _interval_from_bootstrap(seed_mean_delta, bootstrap_indices)
        comparisons[comparison_id] = {
            "candidate": candidate,
            "reference": reference,
            "seed_average_first": True,
            "block_size_windows": BOOTSTRAP_BLOCK_SIZE,
            "repeats": BOOTSTRAP_REPEATS,
            "metrics": metric_intervals,
        }

    # Compare the validation-selected fusion to the member in each role chosen by
    # that seed's own validation selector. The reference can differ across seeds.
    selected_member_roles: dict[int, list[tuple[int, str]]] = {}
    pair_members = {name: [a, b] for name, a, b in PAIR_SPECS}
    for run_index, run in enumerate(seed_results):
        scheme = run["validation_selection"]["fusion_selected"]["selected_scheme"]
        members = pair_members.get(scheme, list(("learned_selected", "flat", "difference"))
                                   if scheme == "threeway_mean" else [])
        for role_index, member in enumerate(members):
            selected_member_roles.setdefault(role_index, []).append((run_index, member))
    for role_index, references in sorted(selected_member_roles.items()):
        comparison_id = f"fusion_selected__vs__selected_member_role_{role_index + 1}"
        included_seeds = []
        seed_refs: dict[str, str] = {}
        per_seed_by_metric: dict[str, list[np.ndarray]] = {metric: [] for metric in metrics}
        for run_index, reference in references:
            run = seed_results[run_index]
            windows = seed_windows[run_index]
            included_seeds.append(int(run["seed"]))
            seed_refs[str(run["seed"])] = reference
            for metric in metrics:
                per_seed_by_metric[metric].append(
                    np.asarray(windows["fusion_selected"][metric], dtype=np.float64)
                    - np.asarray(windows[reference][metric], dtype=np.float64)
                )
        metric_intervals = {}
        for metric, per_seed in per_seed_by_metric.items():
            if not per_seed or len({len(row) for row in per_seed}) != 1:
                continue
            seed_mean_delta = np.stack(per_seed, axis=0).mean(axis=0)
            output_npz[
                f"{group_key}__{comparison_id}__seedmean_window_delta__{metric}"
            ] = seed_mean_delta.astype(np.float64)
            for seed, per_seed_delta in zip(included_seeds, per_seed):
                output_npz[
                    f"{group_key}__seed{seed}__{comparison_id}__window_delta__{metric}"
                ] = per_seed_delta.astype(np.float64)
            metric_intervals[metric] = _interval_from_bootstrap(seed_mean_delta, bootstrap_indices)
        comparisons[comparison_id] = {
            "candidate": "fusion_selected",
            "reference": f"selected_member_role_{role_index + 1}_each_seed",
            "seed_references": seed_refs,
            "included_seeds": included_seeds,
            "seed_average_first": True,
            "block_size_windows": BOOTSTRAP_BLOCK_SIZE,
            "repeats": BOOTSTRAP_REPEATS,
            "metrics": metric_intervals,
        }
    return comparisons


def _validated_cell_metadata(cell_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    required = [cell_dir / name for name in
                ("config.json", "checkpoint.pt", "predictions.npz", "results.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cell is missing required artifacts: {missing}")
    config = json.loads(required[0].read_text(encoding="utf-8"))
    source_results = json.loads(required[3].read_text(encoding="utf-8"))
    if config.get("smoke", False) or source_results.get("smoke", False):
        raise ValueError(f"smoke/incomplete cells are not accepted: {cell_dir}")
    if source_results.get("checkpoint_reload_verified") is not True:
        raise ValueError(f"source cell lacks checkpoint_reload_verified=true: {cell_dir}")
    if source_results.get("data", {}).get("smoke_truncated", False):
        raise ValueError(f"source data were smoke-truncated: {cell_dir}")
    return config, source_results


def _process_cell(cell_value: str, cohort: str, output_root: Path,
                  device: torch.device, code_hash: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    cell_dir = Path(cell_value).expanduser().resolve()
    config, source_results = _validated_cell_metadata(cell_dir)
    dataset, backbone, group_key = _cell_identity(cell_dir, config)
    if config["backbone"].lower() != backbone.lower():
        raise AssertionError("backbone identity mismatch")
    if list(config.get("seeds", SEEDS)) != SEEDS:
        raise ValueError(f"expected seeds {SEEDS} in {cell_dir}")
    if list(config.get("scales", [])) != SCALES:
        raise ValueError(f"expected scales {SCALES} in {cell_dir}")
    if list(map(float, config.get("alpha_grid", []))) != ALPHA_GRID:
        raise ValueError(f"expected alpha grid {ALPHA_GRID} in {cell_dir}")
    if int(config.get("projection_steps", -1)) != PROJECTION_STEPS or float(config.get("max_rms", -1)) != MAX_RMS:
        raise ValueError(f"projection settings differ from frozen protocol in {cell_dir}")

    checkpoint = torch.load(cell_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    if sorted(int(key) for key in checkpoint) != SEEDS:
        raise ValueError(f"checkpoint seeds differ from fixed protocol in {cell_dir}")
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    for seed in SEEDS:
        seed_config = checkpoint[str(seed)].get("config")
        if not isinstance(seed_config, dict) or json.dumps(
            seed_config, sort_keys=True, separators=(",", ":")
        ) != config_json:
            raise ValueError(f"checkpoint/config mismatch for seed {seed} in {cell_dir}")
    saved_npz_file = np.load(cell_dir / "predictions.npz", allow_pickle=False)
    saved_predictions = {key: saved_npz_file[key] for key in saved_npz_file.files}
    saved_npz_file.close()
    required_prediction_keys = {
        f"seed{seed}_{name}"
        for seed in SEEDS
        for name in ("history", "truth", "base", "difference", "residual",
                     "residual_direction", "direction_head", "direction_head_probs")
    } | {"origins"}
    missing_predictions = sorted(required_prediction_keys - set(saved_predictions))
    if missing_predictions:
        raise ValueError(f"source prediction archive is incomplete: {missing_predictions}")
    cell_output = output_root / "cell_artifacts" / group_key
    if cell_output.exists():
        raise FileExistsError(f"refusing to overwrite cell output: {cell_output}")
    cell_output.mkdir(parents=True, exist_ok=False)
    cell_npz: dict[str, np.ndarray] = {}
    seed_results = []
    seed_windows = []
    data_info_first = None
    t0 = time.perf_counter()
    for seed in SEEDS:
        print(f"[{group_key}] seed {seed}: rebuilding validation atoms and frozen fusion set", flush=True)
        record, windows = _per_seed_cell(
            cell_dir, group_key, dataset, backbone, config, checkpoint[str(seed)], seed,
            saved_predictions, device, cell_npz,
        )
        seed_results.append(record)
        seed_windows.append(windows)
        if data_info_first is None:
            data_info_first = record["reconstruction"]["data"]
        elif record["reconstruction"]["data"]["data_sha256"] != data_info_first["data_sha256"]:
            raise AssertionError(f"data source changed across seeds in {cell_dir}")
        # Persist each seed atomically enough to retain work if a later seed fails.
        seed_partial = {"group_key": group_key, "seed_result": record}
        _write_json(cell_output / f"seed{seed}_partial.json", seed_partial)
    bootstrap = _bootstrap_group(group_key, seed_results, seed_windows, cell_npz)
    method_names = list(seed_results[0]["test_metrics"])
    test_summary = {
        method: _mean_std_metric_rows([run["test_metrics"][method]["metrics"] for run in seed_results])
        for method in method_names
    }
    group_result = {
        "dataset": dataset,
        "backbone": backbone,
        "group_key": group_key,
        "cohort": cohort,
        "cell_dir": str(cell_dir),
        "device": str(device),
        "source": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": code_hash,
            "trend_lab_module_sha256": {
                name: _sha256((WORKSPACE / "DTL/trend_lab" / name))
                for name in ("data.py", "decoding.py", "metrics.py", "models.py", "operators.py", "run.py")
            },
            "config_sha256": _sha256(cell_dir / "config.json"),
            "checkpoint_sha256": _sha256(cell_dir / "checkpoint.pt"),
            "source_prediction_npz_sha256": _sha256(cell_dir / "predictions.npz"),
            "checkpoint_reload_verified_in_source": True,
            "data_sha256": data_info_first["data_sha256"],
            "data_hash_manifests": [str(path) for path in _find_hash_manifests(cell_dir)],
        },
        "protocol": {
            "alpha_grid": ALPHA_GRID,
            "weight_grid": WEIGHT_GRID,
            "scales": SCALES,
            "projection_steps": PROJECTION_STEPS,
            "max_rms": MAX_RMS,
            "strict_validation_gate": f"raw_sign_accuracy > base AND standardized_mse <= base + {STRICT_MSE_TOLERANCE:g}",
            "weighted_path_formula": "(1-w)*a + w*b",
            "test_selection": "all alpha, w, and method choices frozen from validation before test metrics are computed",
            "fresh_test_segment_not_previously_scored": cohort == "additional_validation",
            "formal_independent_benchmark": False,
            "prior_definition": "frequency counts from train labels at each scale/horizon/channel; same probabilities for all val/test windows",
            "flat_definition": "one-hot flat class index 1 at every scale/horizon/channel/window",
            "bootstrap": {
                "block_size_windows": BOOTSTRAP_BLOCK_SIZE,
                "repeats": BOOTSTRAP_REPEATS,
                "order": "average per-window deltas across seeds, then resample contiguous time blocks",
            },
        },
        "group_data": data_info_first,
        "seeds": {str(run["seed"]): run for run in seed_results},
        "test_summary_mean_sample_std": test_summary,
        "time_block_bootstrap_after_seed_mean": bootstrap,
        "runtime_seconds": time.perf_counter() - t0,
    }
    _write_json(cell_output / "results.json", group_result)
    np.savez_compressed(cell_output / "predictions.npz", **cell_npz)
    bootstrap_arrays = {key: value for key, value in cell_npz.items() if "window_delta" in key}
    np.savez_compressed(cell_output / "bootstrap_inputs.npz", **bootstrap_arrays)
    _write_json(cell_output / "cell_manifest.json", {
        "cohort": cohort, "group_key": group_key, "cell_dir": str(cell_dir),
        "script_sha256": code_hash, "data_sha256": data_info_first["data_sha256"],
        "results": "results.json", "predictions": "predictions.npz",
        "bootstrap_inputs": "bootstrap_inputs.npz",
    })
    return group_key, group_result, cell_npz


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells", nargs="+", help="one or more completed cell directories")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cohort", required=True, choices=("exploratory", "additional_validation"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    device = torch.device(args.device)
    output_root = args.output_dir.expanduser().resolve()
    preflight_groups = set()
    resolved_cells = []
    for cell in args.cells:
        cell_dir = Path(cell).expanduser().resolve()
        config, _ = _validated_cell_metadata(cell_dir)
        dataset, backbone, group_key = _cell_identity(cell_dir, config)
        if group_key in preflight_groups:
            raise ValueError(f"multiple inputs map to duplicate dataset/backbone group {group_key}")
        preflight_groups.add(group_key)
        resolved_cells.append((cell_dir, dataset, backbone, group_key))
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    code_hash = _sha256(Path(__file__).resolve())
    script_snapshot = output_root / "evaluate_trend_fusions.py"
    shutil.copy2(Path(__file__).resolve(), script_snapshot)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    groups: dict[str, Any] = {}
    seen_groups: set[str] = set()
    all_cell_npz: dict[str, np.ndarray] = {}
    manifest = {
        "schema_version": 1,
        "run_id_utc": timestamp,
        "cohort": args.cohort,
        "device": args.device,
        "torch_num_threads": 2,
        "script": str(Path(__file__).resolve()),
        "script_snapshot": str(script_snapshot),
        "script_sha256": code_hash,
        "requested_cells": [str(cell[0]) for cell in resolved_cells],
        "requested_groups": [cell[3] for cell in resolved_cells],
        "completed_groups": [],
        "in_progress_note": "per-cell results/predictions are written under cell_artifacts as each cell finishes",
    }
    _write_json(output_root / "output_manifest.json", manifest)
    for cell_dir, dataset, backbone, expected_group in resolved_cells:
        group_key, result, cell_npz = _process_cell(str(cell_dir), args.cohort, output_root, device, code_hash)
        if group_key != expected_group:
            raise AssertionError(f"preflight identity changed for {cell_dir}")
        if group_key in seen_groups:
            raise ValueError(f"multiple cells map to duplicate dataset/backbone group {group_key}")
        seen_groups.add(group_key)
        groups[group_key] = result
        all_cell_npz.update(cell_npz)
        manifest["completed_groups"].append(group_key)
        _write_json(output_root / "results.json", {
            "schema_version": 1, "run_id_utc": timestamp, "cohort": args.cohort,
            "device": args.device, "script_sha256": code_hash, "groups": groups,
        })
        _write_json(output_root / "output_manifest.json", manifest)
        print(f"[{group_key}] persisted cell output; aggregate groups so far={len(groups)}", flush=True)
    np.savez_compressed(output_root / "predictions.npz", **all_cell_npz)
    all_bootstrap = {key: value for key, value in all_cell_npz.items() if "window_delta" in key}
    np.savez_compressed(output_root / "bootstrap_inputs.npz", **all_bootstrap)
    final_manifest = {**manifest, "complete": True, "predictions": "predictions.npz",
                      "bootstrap_inputs": "bootstrap_inputs.npz", "results": "results.json"}
    _write_json(output_root / "output_manifest.json", final_manifest)
    print(f"[complete] output={output_root}", flush=True)
    print(f"[complete] groups={','.join(groups)} script_sha256={code_hash}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
