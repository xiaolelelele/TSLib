#!/usr/bin/env python3
"""Independent NumPy audit for fixed three-way forecast averages.

This auditor reads only JSON and NPZ files (NPZ loading always disables pickle).
It never loads checkpoints and does not import the evaluator.  The numerical
core independently recomputes float64 labels/metrics, float32 fixed averages,
validation gates/selections, deployment fallbacks, paired window deltas, and
seed-mean time-block bootstrap intervals.

The artifact adapter follows the fixed-threeway schema described in
``fixed_threeway_verification_protocol_2026-09-28.md``.  It intentionally
states the scope of the train-prior check: saved training counts are compared
with saved probabilities, but counts are not rebuilt from source CSV labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ATOL = 3e-6
PATH_ATOL = 3e-6
STRICT_MSE_TOLERANCE = 1e-8
ALPHA_GRID = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
ATOMS = ("learned_single", "learned_multi", "prior", "flat")
STANDALONE_ORDER = [
    "base", "difference", "residual", "residual_direction",
    "learned_single", "learned_multi", "prior", "flat",
]
COMBOS = {
    "learned_threeway": ("learned_selected", "flat", "difference"),
    "nohead_threeway": ("base", "flat", "difference"),
    "base_difference_mean": ("base", "difference"),
    "flat_difference_mean": ("flat", "difference"),
}


def fail(message: str) -> None:
    raise AssertionError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def close(actual: Any, expected: Any, label: str, atol: float = ATOL) -> None:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if a.shape != e.shape or not np.allclose(a, e, atol=atol, rtol=0, equal_nan=False):
        fail(f"{label}: actual={a!r}, expected={e!r}, atol={atol}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_array(archive: Any, key: str) -> np.ndarray:
    require(key in archive, f"missing NPZ key: {key}")
    arr = np.asarray(archive[key])
    require(np.issubdtype(arr.dtype, np.number), f"non-numeric NPZ array: {key}")
    require(np.isfinite(arr).all(), f"non-finite NPZ array: {key}")
    return arr


def _classification(true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    t, p = np.asarray(true, dtype=np.int64).reshape(-1), np.asarray(pred, dtype=np.int64).reshape(-1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (t, p), 1)
    support = matrix.sum(axis=1)
    tp = np.diag(matrix)
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros(3, dtype=np.float64), where=matrix.sum(axis=0) != 0)
    recall = np.divide(tp, support, out=np.zeros(3, dtype=np.float64), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3, dtype=np.float64), where=precision + recall != 0)
    present = support > 0
    return {
        "accuracy": float(np.mean(t == p)) if t.size else None,
        # Match trend_lab.metrics._classification: absent true classes score 0 in macro-F1.
        "macro_f1": float(f1.mean()) if t.size else None,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else None,
        "confusion_matrix": matrix.tolist(),
    }


def labels(history: np.ndarray, future: np.ndarray, scales: Iterable[int], epsilon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return [N,S,H,C] 3-class slope labels and [N,H,C] raw sign labels."""
    x = np.asarray(history, dtype=np.float64)
    y = np.asarray(future, dtype=np.float64)
    scale_values = [int(k) for k in scales]
    eps = np.asarray(epsilon, dtype=np.float64)
    require(x.ndim == y.ndim == 3, "history/future must be [N,T,C]")
    require(x.shape[0] == y.shape[0] and x.shape[2] == y.shape[2], "history/future shapes are incompatible")
    require(eps.shape == (len(scale_values), x.shape[2]), f"epsilon has unexpected shape {eps.shape}")
    joined = np.concatenate((x, y), axis=1)
    origin = x.shape[1] - 1
    steps = np.arange(1, y.shape[1] + 1)
    slope_values = [
        (joined[:, origin + steps, :] - joined[:, origin + steps - scale, :]) / float(scale)
        for scale in scale_values
    ]
    slopes = np.stack(slope_values, axis=1)
    eps_full = eps[None, :, None, :]
    classes = np.where(slopes > eps_full, 2, np.where(slopes < -eps_full, 0, 1)).astype(np.int64)
    history_last = x[:, -1:, :]
    raw_sign = np.sign(np.diff(np.concatenate((history_last, y), axis=1), axis=1))
    return classes, raw_sign


def score(history: np.ndarray, truth: np.ndarray, path: np.ndarray, scales: list[int],
          epsilon: np.ndarray, normalization_std: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Recompute metrics in float64 and per-window values for paired intervals."""
    x = np.asarray(history, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    z = np.asarray(path, dtype=np.float64)
    require(y.shape == z.shape and x.ndim == y.ndim == z.ndim == 3, "incompatible score input shapes")
    true_class, true_sign = labels(x, y, scales, epsilon)
    pred_class, pred_sign = labels(x, z, scales, epsilon)
    err = z - y
    std = np.asarray(normalization_std, dtype=np.float64).reshape(1, 1, -1)
    raw_window = np.mean(true_sign == pred_sign, axis=(1, 2))
    mse_window = np.mean(err ** 2, axis=(1, 2))
    orig_mse_window = np.mean((err * std) ** 2, axis=(1, 2))
    accuracy_by_scale: dict[str, float] = {}
    macro_f1_by_scale: dict[str, float] = {}
    scale_window: dict[str, np.ndarray] = {}
    for i, scale in enumerate(scales):
        k = str(int(scale))
        stats = _classification(true_class[:, i], pred_class[:, i])
        accuracy_by_scale[k] = float(stats["accuracy"])
        macro_f1_by_scale[k] = float(stats["macro_f1"])
        scale_window[k] = np.mean(true_class[:, i] == pred_class[:, i], axis=(1, 2))
    mean3_window = np.mean(np.stack(list(scale_window.values()), axis=1), axis=1)
    metric = {
        "raw_sign_accuracy": float(np.mean(true_sign == pred_sign)),
        "standardized_mse": float(np.mean(err ** 2)),
        "original_units_mse": float(np.mean((err * std) ** 2)),
        "accuracy_by_scale": accuracy_by_scale,
        "mean_3class_accuracy": float(np.mean(list(accuracy_by_scale.values()))),
        "macro_f1_by_scale": macro_f1_by_scale,
        "mean_macro_f1": float(np.mean(list(macro_f1_by_scale.values()))),
    }
    windows = {
        "raw_sign_accuracy": raw_window,
        "standardized_mse": mse_window,
        "original_units_mse": orig_mse_window,
        "mean_3class_accuracy": mean3_window,
        **{f"accuracy_scale_{k}": v for k, v in scale_window.items()},
    }
    return metric, windows


def _metric_value(metrics: dict[str, Any], name: str) -> Any:
    aliases = {
        "standardized_mse": ("standardized_mse", "mse"),
        "original_units_mse": ("original_units_mse", "original_mse"),
        "mean_3class_accuracy": ("mean_3class_accuracy", "mean_accuracy_3class"),
        "mean_macro_f1": ("mean_macro_f1",),
    }
    for key in aliases.get(name, (name,)):
        if key in metrics:
            return metrics[key]
    return None


def check_reported_metrics(reported: dict[str, Any], measured: dict[str, Any], label: str) -> None:
    """Accept evaluator naming aliases but require all protocol metrics."""
    require(isinstance(reported, dict), f"{label}: metrics must be an object")
    for key in ("raw_sign_accuracy", "standardized_mse", "original_units_mse", "mean_3class_accuracy", "mean_macro_f1"):
        value = _metric_value(reported, key)
        require(value is not None, f"{label}: missing {key}")
        close(value, measured[key], f"{label}.{key}")
    accs = reported.get("accuracy_by_scale")
    macro = reported.get("macro_f1_by_scale")
    scales_obj = reported.get("scales")
    if accs is None and isinstance(scales_obj, dict):
        accs = {k: v.get("accuracy") for k, v in scales_obj.items() if isinstance(v, dict)}
    if macro is None and isinstance(scales_obj, dict):
        macro = {k: v.get("macro_f1") for k, v in scales_obj.items() if isinstance(v, dict)}
    require(isinstance(accs, dict) and isinstance(macro, dict), f"{label}: missing per-scale accuracy/macro-F1")
    for scale, value in measured["accuracy_by_scale"].items():
        close(accs.get(scale), value, f"{label}.accuracy_by_scale.{scale}")
    for scale, value in measured["macro_f1_by_scale"].items():
        close(macro.get(scale), value, f"{label}.macro_f1_by_scale.{scale}")


def eligible(metric: dict[str, Any], base: dict[str, Any]) -> bool:
    return bool(
        metric["raw_sign_accuracy"] > base["raw_sign_accuracy"]
        and metric["standardized_mse"] <= base["standardized_mse"] + STRICT_MSE_TOLERANCE
    )


def selected_alpha(rows: list[dict[str, Any]], base_metric: dict[str, Any]) -> dict[str, Any] | None:
    qualified = [row for row in rows if eligible(row["_measured"], base_metric)]
    qualified.sort(key=lambda row: (
        -row["_measured"]["raw_sign_accuracy"],
        row["_measured"]["standardized_mse"],
        float(row["alpha"]),
    ))
    return qualified[0] if qualified else None


def selected_standalone(metrics: dict[str, dict[str, Any]], base_metric: dict[str, Any]) -> str:
    qualified = [name for name in STANDALONE_ORDER if eligible(metrics[name], base_metric)]
    if not qualified:
        return "base"
    return min(qualified, key=lambda name: (
        -metrics[name]["raw_sign_accuracy"], metrics[name]["standardized_mse"],
        STANDALONE_ORDER.index(name),
    ))


def fixed_mean(paths: list[np.ndarray], arithmetic: str) -> np.ndarray:
    """Protocol float32 equal mean, returning an existing path for all-equal members.

    ``sum_divide`` replays the historical CPU evaluator.  CUDA outputs used
    ``sum_reciprocal_multiply``; the two can differ by one float32 ULP and that
    difference can change an exact raw-sign label, so callers must choose from
    recorded source metadata rather than accepting a loose path tolerance.
    """
    require(bool(paths), "fixed_mean needs at least one path")
    cast = [np.asarray(value, dtype=np.float32) for value in paths]
    require(all(value.shape == cast[0].shape for value in cast), "fixed_mean member shapes differ")
    if all(np.array_equal(cast[0], value) for value in cast[1:]):
        return cast[0]
    summed = cast[0]
    for member in cast[1:]:
        summed = np.asarray(summed + member, dtype=np.float32)
    if arithmetic == "sum_divide":
        return np.asarray(summed / np.float32(len(cast)), dtype=np.float32)
    if arithmetic == "sum_reciprocal_multiply":
        reciprocal = np.float32(1.0 / len(cast))
        return np.asarray(summed * reciprocal, dtype=np.float32)
    fail(f"unsupported fixed-mean arithmetic {arithmetic!r}")


def assert_path(actual: np.ndarray, expected: np.ndarray, label: str, exact: bool = False) -> None:
    if exact:
        require(np.array_equal(actual, expected), f"{label}: paths are not exact aliases")
    else:
        close(actual, expected, label, atol=PATH_ATOL)


def unique_path_count(paths: list[np.ndarray]) -> int:
    unique = []
    for path in paths:
        if not any(np.array_equal(path, previous) for previous in unique):
            unique.append(path)
    return len(unique)


def bootstrap_ci(window_deltas: np.ndarray, block_size: int = 8, repeats: int = 2000,
                 seed: int = 0) -> dict[str, Any]:
    """Seed-mean, non-circular moving-block bootstrap (frozen seed-0 rule)."""
    values = np.asarray(window_deltas, dtype=np.float64).reshape(-1)
    require(values.size > 0 and np.isfinite(values).all(), "bootstrap requires finite window deltas")
    block = min(int(block_size), values.size)
    blocks_needed = int(math.ceil(values.size / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, values.size - block + 1, size=(int(repeats), blocks_needed))
    indices = (starts[:, :, None] + np.arange(block, dtype=np.int64)[None, None, :]).reshape(int(repeats), -1)
    indices = indices[:, :values.size]
    boot_means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(boot_means, [0.025, 0.975])],
        "block_size_windows": int(block_size),
        "repeats": int(repeats),
        "seed": int(seed),
    }


def _path_row(row: dict[str, Any], archive: Any, seed: int, split: str,
              expected_id: str | None = None, label: str = "path",
              expected_candidate_id: str | None = None) -> tuple[np.ndarray, str]:
    require(isinstance(row, dict), f"{label}: expected a row object")
    key = row.get("path_key", row.get("prediction_key"))
    require(isinstance(key, str), f"{label}: missing path_key")
    if expected_candidate_id is not None:
        require(row.get("candidate_id") == expected_candidate_id,
                f"{label}: source candidate ID mismatch {row.get('candidate_id')!r} != {expected_candidate_id!r}")
    if expected_id is not None:
        wanted = f"seed{seed}__{split}__{expected_id}__path"
        require(key == wanted, f"{label}: path key mismatch {key!r} != {wanted!r}")
    path = get_array(archive, key)
    return path, key


def _reported_metrics(row: dict[str, Any], history: np.ndarray, truth: np.ndarray,
                      path: np.ndarray, scales: list[int], epsilon: np.ndarray,
                      norm_std: np.ndarray, label: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    require(isinstance(row.get("metrics"), dict), f"{label}: missing metrics object")
    measured, windows = score(history, truth, path, scales, epsilon, norm_std)
    check_reported_metrics(row["metrics"], measured, label)
    return measured, windows


def _field_id(row: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, str):
            return value
    return None


def _records(value: Any, label: str) -> list[tuple[str | None, dict[str, Any]]]:
    if isinstance(value, list):
        require(all(isinstance(row, dict) for row in value), f"{label}: list rows must be objects")
        return [(_field_id(row, "method", "member", "name", "candidate_id"), row) for row in value]
    if isinstance(value, dict):
        if "candidate_id" in value or "path_key" in value or "prediction_key" in value:
            return [(_field_id(value, "method", "member", "name", "candidate_id"), value)]
        records = []
        for name, row in value.items():
            require(isinstance(row, dict), f"{label}.{name}: row must be an object")
            records.append((str(name), row))
        return records
    fail(f"{label}: expected object or list")


def _method_from_record(key: str | None, row: dict[str, Any]) -> str:
    for value in (key, row.get("method"), row.get("member"), row.get("name"), row.get("source_method")):
        if isinstance(value, str):
            normalized = value[len("standalone_"):] if value.startswith("standalone_") else value
            if normalized in STANDALONE_ORDER or normalized in ("learned_selected", "standalone_selected"):
                return normalized
    candidate = _field_id(row, "source_candidate_id")
    if candidate in STANDALONE_ORDER:
        return candidate
    fail(f"cannot identify standalone method in row {row}")


def _mean_operation(group: dict[str, Any]) -> str:
    provenance = group.get("group", {}).get("source_provenance", group.get("source_provenance", {}))
    group_meta = group.get("group", {})
    operation = (provenance.get("combination_arithmetic")
                 or group_meta.get("combination_arithmetic")
                 or group.get("combination_arithmetic"))
    if operation in ("sum_divide", "sum_reciprocal_multiply"):
        return operation
    device = (provenance.get("source_device") or provenance.get("device")
              or group_meta.get("source_device") or group_meta.get("device")
              or group.get("device"))
    if isinstance(device, str):
        return "sum_reciprocal_multiply" if "cuda" in device.lower() else "sum_divide"
    fail("missing source device/combination_arithmetic; float32 fixed mean cannot be replayed exactly")


def _array_for(archive: Any, seed: int, suffixes: Iterable[str]) -> np.ndarray:
    for suffix in suffixes:
        key = f"seed{seed}__metadata__{suffix}"
        if key in archive:
            return get_array(archive, key)
    fail(f"missing metadata array for seed {seed}; tried {list(suffixes)}")


def _validate_prior(archive: Any, seed: int, split: str, scales: list[int], channels: int) -> dict[str, Any]:
    counts = _array_for(archive, seed, ("prior_class_counts", "train_prior_class_counts")).astype(np.float64)
    require(counts.shape[-1] == 3 and counts.sum(axis=-1).min() > 0, f"seed {seed}: invalid/empty train prior counts")
    checks = 0
    for method in ("prior", "flat"):
        key = f"seed{seed}__{split}__probabilities__{method}__probabilities"
        require(key in archive, f"missing {method} probability array for {split}, seed {seed}")
        probabilities = get_array(archive, key).astype(np.float64)
        require(probabilities.shape[-1] == 3, f"{key}: expected three classes")
        probability_sums = probabilities.sum(axis=-1)
        close(probability_sums, np.ones_like(probability_sums), f"{key}: probability sums", atol=1e-5)
        if method == "prior":
            expected = counts / counts.sum(axis=-1, keepdims=True)
            try:
                expected = np.broadcast_to(expected, probabilities.shape)
            except ValueError:
                fail(f"{key}: prior count shape {counts.shape} cannot broadcast to probability shape {probabilities.shape}")
            close(probabilities, expected, f"{key}: saved train counts to prior probability check", atol=1e-6)
        else:
            expected = np.zeros_like(probabilities)
            expected[..., 1] = 1.0
            close(probabilities, expected, f"{key}: flat probability identity", atol=1e-6)
        checks += 1
    return {"train_counts_to_probabilities_checked": checks > 0, "scope": "saved train counts to probabilities only; counts not rebuilt from CSV labels"}


def _audit_json_path_keys(value: Any, archive: Any, context: str) -> int:
    count = 0
    if isinstance(value, dict):
        for key_name, child in value.items():
            if (key_name in ("path_key", "prediction_key") or key_name.endswith("_path_key")) and isinstance(child, str):
                require(child in archive, f"{context}: JSON {key_name} absent from NPZ: {child}")
                count += 1
            count += _audit_json_path_keys(child, archive, context)
    elif isinstance(value, list):
        for child in value:
            count += _audit_json_path_keys(child, archive, context)
    return count


def _check_selection_ref(row: dict[str, Any], chosen_id: str, label: str) -> None:
    candidates = [_field_id(row, "selected_candidate_id", "source_candidate_id", "candidate_id", "source_id")]
    candidates += [_field_id(row, "selected_alpha_candidate_id", "selected_source_candidate_id")]
    for candidate in candidates:
        if candidate is not None:
            require(candidate == chosen_id, f"{label}: selected candidate {candidate!r} != replayed {chosen_id!r}")
            return
    # The exact path key still carries a candidate ID; selection identity should
    # be explicit in JSON so that it can be traced without reverse engineering.
    fail(f"{label}: no candidate identity field")


def _json_delta_report(row: dict[str, Any], candidate: dict[str, Any], reference: dict[str, Any], label: str) -> None:
    """Check all frozen pointwise paired metric deltas and MSE ratio reports."""
    metrics_row = row.get("metrics_delta", row)
    require(isinstance(metrics_row, dict), f"{label}: missing metrics_delta object")
    for metric in ("raw_sign_accuracy", "standardized_mse", "original_units_mse",
                   "mean_3class_accuracy", "mean_macro_f1"):
        key = f"{metric}_delta" if f"{metric}_delta" in metrics_row else f"{metric}_mean_delta"
        require(key in metrics_row, f"{label}: missing paired delta {key}")
        close(metrics_row[key], candidate[metric] - reference[metric], f"{label}.{key}")
        pct_key = f"{metric}_percent_change"
        if metric in ("standardized_mse", "original_units_mse"):
            require(pct_key in metrics_row, f"{label}: missing {pct_key}")
            ref = reference[metric]
            expected = 100.0 * (candidate[metric] / ref - 1.0) if ref else None
            if expected is None:
                require(metrics_row[pct_key] is None, f"{label}.{pct_key}: expected null")
            else:
                close(metrics_row[pct_key], expected, f"{label}.{pct_key}")
    for field in ("accuracy_by_scale", "macro_f1_by_scale"):
        key = f"{field}_delta"
        require(isinstance(metrics_row.get(key), dict), f"{label}: missing {key}")
        for scale, value in candidate[field].items():
            close(metrics_row[key].get(scale), value - reference[field][scale], f"{label}.{key}.{scale}")


def _audit_old_threeway_source(group: dict[str, Any], seed: int, deployed_test: np.ndarray,
                               raw_validation: np.ndarray) -> dict[str, Any] | None:
    """Reopen the frozen source evaluator artifact and compare its deployed path."""
    meta = (group.get("group", {}).get("source_provenance")
            or group.get("source_provenance") or group.get("source") or {})
    source_root_value = meta.get("evaluator_root") or meta.get("source_evaluator_root")
    if not source_root_value:
        return None
    source_root = Path(source_root_value)
    source_results_path = source_root / "results.json"
    source_npz_path = source_root / "predictions.npz"
    if not source_results_path.is_file() or not source_npz_path.is_file():
        return None
    source_results = json.loads(source_results_path.read_text(encoding="utf-8"))
    source_manifest_path = source_root / "output_manifest.json"
    require(source_manifest_path.is_file(), f"old source root lacks output manifest: {source_root}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    require(source_manifest.get("complete") is True, f"old source root incomplete: {source_root}")
    if meta.get("source_evaluator_results_sha256"):
        require(sha256_file(source_results_path) == meta["source_evaluator_results_sha256"],
                f"old source results hash changed: {source_root}")
    if meta.get("source_evaluator_prediction_npz_sha256"):
        require(sha256_file(source_npz_path) == meta["source_evaluator_prediction_npz_sha256"],
                f"old source prediction archive hash changed: {source_root}")
    evaluator_snapshot = Path(source_manifest.get("script_snapshot", ""))
    require(evaluator_snapshot.is_file() and sha256_file(evaluator_snapshot) == source_manifest.get("script_sha256"),
            f"old source evaluator snapshot hash mismatch: {source_root}")
    groups = source_results.get("groups", {})
    source_group_key = meta.get("source_group_key")
    if source_group_key not in groups:
        dataset = group.get("group", {}).get("dataset", group.get("dataset"))
        backbone = group.get("group", {}).get("backbone", group.get("backbone"))
        matches = [key for key, value in groups.items()
                   if value.get("dataset") == dataset and value.get("backbone") == backbone]
        if len(matches) != 1:
            return None
        source_group_key = matches[0]
    original_group = groups[source_group_key]
    run = original_group.get("seeds", {}).get(str(seed))
    if not isinstance(run, dict):
        fail(f"old source {source_group_key}: missing seed {seed}")
    old_row = run.get("test_metrics", {}).get("threeway_mean")
    require(isinstance(old_row, dict), f"old source {source_group_key}/seed{seed}: missing threeway_mean test row")
    old_key = old_row.get("prediction_key")
    require(isinstance(old_key, str), f"old source {source_group_key}/seed{seed}: missing threeway path key")
    with np.load(source_npz_path, allow_pickle=False) as old_archive:
        old_path = get_array(old_archive, old_key)
        old_val_row = run.get("validation_grids", {}).get("fixed_threeway_mean", {})
        old_val_key = old_val_row.get("prediction_key")
        require(isinstance(old_val_key, str), f"old source {source_group_key}/seed{seed}: missing raw validation triple path")
        old_val_path = get_array(old_archive, old_val_key)
    require(np.array_equal(raw_validation, old_val_path),
            f"{source_group_key}/seed{seed}: new raw learned_threeway differs from frozen source validation triple")
    require(np.array_equal(deployed_test, old_path),
            f"{source_group_key}/seed{seed}: new learned_threeway deployment differs from frozen old deployed path")
    selected_ids = run.get("test_selected_identity", {}).get("selected_test_candidate_ids", {})
    reported_id = selected_ids.get("threeway_mean")
    require(reported_id == old_row.get("candidate_id"), f"{source_group_key}/seed{seed}: old selected identity differs from test row")
    return {"source_root": str(source_root), "source_group_key": source_group_key,
            "old_candidate_id": reported_id, "raw_validation_exact_match": True,
            "deployed_test_exact_match": True}


def _find_method_rows(value: Any, label: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key, row in _records(value, label):
        method = _method_from_record(key, row)
        require(method not in rows, f"{label}: duplicate method row {method}")
        rows[method] = row
    return rows


def _candidate_source_id(row: dict[str, Any], fallback: str | None = None) -> str:
    value = _field_id(row, "source_candidate_id", "selected_source_candidate_id", "selected_candidate_id")
    if value is not None:
        return value
    candidate = row.get("candidate_id")
    if isinstance(candidate, str):
        return candidate
    if fallback:
        return fallback
    fail(f"row has no candidate source identity: {row}")


def _test_delta_windows(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: candidate[name] - reference[name] for name in (
        "raw_sign_accuracy", "standardized_mse", "original_units_mse", "mean_3class_accuracy",
        *[key for key in candidate if key.startswith("accuracy_scale_")],
    )}


def audit_seed(group_key: str, group: dict[str, Any], seed: int, run: dict[str, Any], archive: Any,
               arithmetic: str, group_meta: dict[str, Any]) -> dict[str, Any]:
    """Audit atom selection, all four raw/deployed means, identities, and test metrics."""
    prefix = f"{group_key}/seed{seed}"
    validation = run.get("validation")
    test = run.get("test")
    require(isinstance(validation, dict) and isinstance(test, dict), f"{prefix}: missing validation/test objects")
    scales = [int(x) for x in _array_for(archive, seed, ("scales",)).reshape(-1)]
    epsilon = _array_for(archive, seed, ("epsilon",)).astype(np.float64)
    norm_std = _array_for(archive, seed, ("normalization_std", "std")).astype(np.float64).reshape(-1)
    require(scales == [1, 2, 4, 8], f"{prefix}: scale set/order mismatch: {scales}")
    require(epsilon.shape == (len(scales), norm_std.size), f"{prefix}: epsilon/std shape mismatch")
    require(norm_std.size == len(group_meta.get("input_columns", [])), f"{prefix}: std/input_columns mismatch")
    validation_history = get_array(archive, f"seed{seed}__validation__split__history")
    validation_truth = get_array(archive, f"seed{seed}__validation__split__truth")
    test_history = get_array(archive, f"seed{seed}__test__split__history")
    test_truth = get_array(archive, f"seed{seed}__test__split__truth")
    validation_origins = get_array(archive, f"seed{seed}__validation__split__origins").reshape(-1)
    test_origins = get_array(archive, f"seed{seed}__test__split__origins").reshape(-1)
    require(np.all(np.diff(validation_origins) > 0) and np.all(np.diff(test_origins) > 0),
            f"{prefix}: split origins are not strictly increasing")

    val_base_row = validation.get("base")
    val_base_path, _ = _path_row(val_base_row, archive, seed, "validation", "base",
                                 f"{prefix}/val/base", "base")
    val_base_metric, val_base_windows = _reported_metrics(
        val_base_row, validation_history, validation_truth, val_base_path, scales, epsilon, norm_std,
        f"{prefix}/val/base")
    val_sources: dict[str, dict[str, Any]] = {
        "base": {"row": val_base_row, "path": val_base_path, "metric": val_base_metric,
                 "windows": val_base_windows, "candidate_id": val_base_row.get("candidate_id", "base")}
    }
    check_count = 1

    # Replay all six validation alpha candidates for each correction and the
    # alpha selected under the strict raw-sign/MSE gate.
    atom_grids = validation.get("atom_alpha_grid")
    atom_selection = validation.get("atom_selection")
    require(isinstance(atom_grids, dict) and isinstance(atom_selection, dict), f"{prefix}: missing atom grid/selection")
    selected_atom_ids: dict[str, str] = {}
    atom_selected_rows: dict[str, dict[str, Any]] = {}
    for atom in ("learned_multi", "learned_single", "prior", "flat"):
        rows = atom_grids.get(atom)
        require(isinstance(rows, list), f"{prefix}/{atom}: alpha grid must be a list")
        require([float(row["alpha"]) for row in rows] == ALPHA_GRID, f"{prefix}/{atom}: alpha grid/order mismatch")
        replay = []
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            path, _ = _path_row(row, archive, seed, "validation", row.get("candidate_id"),
                                f"{prefix}/val/{atom}/alpha={row.get('alpha')}", row.get("candidate_id"))
            measured, _ = _reported_metrics(row, validation_history, validation_truth, path, scales, epsilon,
                                            norm_std, f"{prefix}/val/{atom}/{row.get('candidate_id')}")
            row["_measured"] = measured
            row["_path"] = path
            row["_window"] = score(validation_history, validation_truth, path, scales, epsilon, norm_std)[1]
            expected_eligible = eligible(measured, val_base_metric)
            require(bool(row.get("eligible")) == expected_eligible, f"{prefix}/{atom}: alpha gate mismatch")
            require(bool(row.get("is_no_op_vs_base")) is np.array_equal(path, val_base_path),
                    f"{prefix}/{atom}: alpha no-op identity flag mismatch")
            if float(row["alpha"]) == 0.0:
                assert_path(path, val_base_path, f"{prefix}/{atom}: alpha=0 no-op", exact=True)
                require(row.get("is_no_op_vs_base") is True, f"{prefix}/{atom}: alpha=0 no-op flag missing")
                require(not expected_eligible, f"{prefix}/{atom}: alpha=0 incorrectly passes strict direction gate")
            cid = row.get("candidate_id")
            require(isinstance(cid, str), f"{prefix}/{atom}: missing candidate ID")
            by_id[cid] = row
            replay.append(row)
            check_count += 1
        chosen = selected_alpha(replay, val_base_metric)
        if chosen is None:
            chosen = next(row for row in replay if float(row["alpha"]) == 0.0)
        selected_atom_ids[atom] = chosen["candidate_id"]
        selection_row = atom_selection.get(atom)
        require(isinstance(selection_row, dict), f"{prefix}/{atom}: missing selected atom identity")
        _check_selection_ref(selection_row, chosen["candidate_id"], f"{prefix}/{atom} alpha selection")
        selected_alpha_value = selection_row.get("selected_alpha", selection_row.get("alpha"))
        close(selected_alpha_value, chosen["alpha"], f"{prefix}/{atom} selected alpha")
        require(bool(selection_row.get("eligible")) is (chosen is not None and eligible(chosen["_measured"], val_base_metric)),
                f"{prefix}/{atom}: selected-atom eligible flag mismatch")
        require(bool(selection_row.get("fallback_to_alpha_zero")) is (not any(eligible(r["_measured"], val_base_metric) for r in replay)),
                f"{prefix}/{atom}: alpha-zero fallback flag mismatch")
        selected_path, _ = _path_row(selection_row, archive, seed, "validation", atom,
                                    f"{prefix}/val/selected-{atom}")
        require(selection_row.get("source_grid_path_key") == by_id[chosen["candidate_id"]].get("path_key"),
                f"{prefix}/{atom}: selected alias does not reference selected alpha grid path")
        assert_path(selected_path, chosen["_path"], f"{prefix}/{atom} selected alias", exact=True)
        selected_metric, selected_windows = score(validation_history, validation_truth, selected_path,
                                                  scales, epsilon, norm_std)
        if "metrics" in selection_row:
            check_reported_metrics(selection_row["metrics"], selected_metric, f"{prefix}/val/selected-{atom}")
        require(chosen["candidate_id"] in by_id, f"{prefix}/{atom}: selected candidate absent from alpha grid")
        assert_path(selected_path, by_id[chosen["candidate_id"]]["_path"], f"{prefix}/{atom} selected source", exact=True)
        val_sources[atom] = {"row": selection_row, "path": selected_path, "metric": selected_metric,
                             "windows": selected_windows, "candidate_id": atom,
                             "source_candidate_id": chosen["candidate_id"]}
        atom_selected_rows[atom] = selection_row
        check_count += 1

    # Ensure the saved training prior probabilities follow the saved counts.
    prior_scope = _validate_prior(archive, seed, "validation", scales, norm_std.size)
    _validate_prior(archive, seed, "test", scales, norm_std.size)

    # Check the eight fixed standalone paths.  The four corrected standalones
    # must alias their validation-selected atom paths exactly.
    standalone_rows = _find_method_rows(validation.get("standalone_candidates"), f"{prefix}/val/standalone_candidates")
    require(set(standalone_rows) == set(STANDALONE_ORDER),
            f"{prefix}: standalone method coverage differs: {sorted(standalone_rows)}")
    standalone_metrics: dict[str, dict[str, Any]] = {}
    standalone_ids: dict[str, str] = {}
    for method in STANDALONE_ORDER:
        row = standalone_rows[method]
        path, _ = _path_row(row, archive, seed, "validation", method,
                            f"{prefix}/val/standalone/{method}", method)
        measured, windows = _reported_metrics(row, validation_history, validation_truth, path, scales,
                                              epsilon, norm_std, f"{prefix}/val/standalone/{method}")
        expected_eligible = eligible(measured, val_base_metric)
        require(bool(row.get("eligible")) is expected_eligible,
                f"{prefix}: standalone {method} eligibility mismatch")
        if method in ("learned_single", "learned_multi", "prior", "flat"):
            assert_path(path, val_sources[method]["path"], f"{prefix}: standalone {method} not selected alpha", exact=True)
        elif method == "base":
            require(not expected_eligible, f"{prefix}: base cannot strictly improve over itself")
        standalone_metrics[method] = measured
        standalone_ids[method] = row.get("candidate_id") or _candidate_source_id(row, method)
        val_sources[method] = {"row": row, "path": path, "metric": measured, "windows": windows,
                               "candidate_id": standalone_ids[method]}
        check_count += 1

    # learned_selected chooses between the two already-selected learned atoms;
    # a complete tie favors the single-scale constraint.
    learned_winner = min(("learned_single", "learned_multi"), key=lambda method: (
        -val_sources[method]["metric"]["raw_sign_accuracy"],
        val_sources[method]["metric"]["standardized_mse"],
        0 if method == "learned_single" else 1,
    ))
    learned_row = validation.get("learned_selected")
    require(isinstance(learned_row, dict), f"{prefix}: missing learned_selected")
    require(learned_row.get("member", learned_row.get("selected_member")) == learned_winner,
            f"{prefix}: learned_selected member differs from validation replay")
    learned_path, _ = _path_row(learned_row, archive, seed, "validation", "learned_selected",
                               f"{prefix}/val/learned_selected")
    require(learned_row.get("selected_candidate_id") == atom_selection[learned_winner].get("selected_candidate_id"),
            f"{prefix}: learned_selected candidate source mismatch")
    close(learned_row.get("selected_alpha"), atom_selection[learned_winner]["selected_alpha"],
          f"{prefix}: learned_selected alpha source mismatch")
    assert_path(learned_path, val_sources[learned_winner]["path"], f"{prefix}: learned_selected alias", exact=True)
    learned_metric, learned_windows = _reported_metrics(
        learned_row, validation_history, validation_truth, learned_path, scales, epsilon, norm_std,
        f"{prefix}/val/learned_selected")
    val_sources["learned_selected"] = {"row": learned_row, "path": learned_path, "metric": learned_metric,
                                       "windows": learned_windows,
                                       "candidate_id": learned_row.get("candidate_id"),
                                       "source_candidate_id": val_sources[learned_winner]["candidate_id"]}
    check_count += 1

    # Standalone selection is replayed over all eight frozen paths with the
    # strict joint validation gate. No test metric enters this choice.
    eligible_standalones = [method for method in STANDALONE_ORDER
                            if eligible(standalone_metrics[method], val_base_metric)]
    if eligible_standalones:
        standalone_winner = min(eligible_standalones, key=lambda method: (
            -standalone_metrics[method]["raw_sign_accuracy"],
            standalone_metrics[method]["standardized_mse"],
            STANDALONE_ORDER.index(method),
        ))
    else:
        standalone_winner = "base"
    standalone_row = validation.get("standalone_selected")
    require(isinstance(standalone_row, dict), f"{prefix}: missing standalone_selected")
    selected_source_id = _candidate_source_id(standalone_row, standalone_winner)
    expected_source_id = (selected_atom_ids[standalone_winner]
                          if standalone_winner in selected_atom_ids else standalone_winner)
    require(selected_source_id == expected_source_id,
            f"{prefix}: standalone_selected source {selected_source_id!r} != replayed {expected_source_id!r}")
    standalone_selected_path, _ = _path_row(standalone_row, archive, seed, "validation", "standalone_selected",
                                            f"{prefix}/val/standalone_selected")
    assert_path(standalone_selected_path, val_sources[standalone_winner]["path"],
                f"{prefix}: standalone_selected alias", exact=True)
    standalone_selected_metric, standalone_selected_windows = _reported_metrics(
        standalone_row, validation_history, validation_truth, standalone_selected_path,
        scales, epsilon, norm_std, f"{prefix}/val/standalone_selected")
    val_sources["standalone_selected"] = {"row": standalone_row, "path": standalone_selected_path,
                                          "metric": standalone_selected_metric,
                                          "windows": standalone_selected_windows,
                                          "candidate_id": standalone_row.get("candidate_id"),
                                          "source_candidate_id": expected_source_id}
    check_count += 1

    # Compute each raw combination in the recorded float32 arithmetic and
    # replay its independent validation gate and base fallback.
    val_combo_state: dict[str, dict[str, Any]] = {}
    combo_rows = validation.get("combinations")
    require(isinstance(combo_rows, dict), f"{prefix}: missing validation combinations")
    for combo, members in COMBOS.items():
        row = combo_rows.get(combo)
        require(isinstance(row, dict) and isinstance(row.get("raw"), dict), f"{prefix}/{combo}: missing raw row")
        raw_row = row["raw"]
        member_paths = [val_sources[name]["path"] for name in members]
        raw_expected = fixed_mean(member_paths, arithmetic)
        raw_path, _ = _path_row(raw_row, archive, seed, "validation", f"{combo}_raw",
                                f"{prefix}/val/{combo}/raw", f"{combo}_raw")
        assert_path(raw_path, raw_expected, f"{prefix}/{combo} raw fixed-mean formula", exact=True)
        raw_metric, raw_windows = _reported_metrics(raw_row, validation_history, validation_truth, raw_path,
                                                   scales, epsilon, norm_std, f"{prefix}/val/{combo}/raw")
        gate = row.get("gate", raw_row.get("gate"))
        require(isinstance(gate, dict), f"{prefix}/{combo}: missing validation gate")
        passed = eligible(raw_metric, val_base_metric)
        require(gate.get("eligible") is passed, f"{prefix}/{combo}: strict validation gate mismatch")
        require(bool(raw_row.get("eligible", passed)) is passed, f"{prefix}/{combo}: raw eligibility mismatch")
        if "reason" in gate and not passed:
            require("raw_sign" in str(gate["reason"]) or "mse" in str(gate["reason"]).lower(),
                    f"{prefix}/{combo}: fallback reason is not tied to strict gate")
        deployed_row = row.get("deployed")
        require(isinstance(deployed_row, dict), f"{prefix}/{combo}: missing deployed row")
        deployed_path, _ = _path_row(deployed_row, archive, seed, "validation", f"{combo}_deployed",
                                     f"{prefix}/val/{combo}/deployed", f"{combo}_deployed")
        expected_deployed = raw_path if passed else val_base_path
        assert_path(deployed_path, expected_deployed, f"{prefix}/{combo} validation deploy/fallback", exact=True)
        expected_fallback = not passed
        require(bool(deployed_row.get("fallback_to_base")) is expected_fallback,
                f"{prefix}/{combo}: fallback flag mismatch")
        require(deployed_row.get("effective_source_candidate_id") == ("base" if expected_fallback else raw_row["candidate_id"]),
                f"{prefix}/{combo}: deployed source identity/fallback mismatch")
        require(bool(deployed_row.get("path_equals_base")) is np.array_equal(deployed_path, val_base_path),
                f"{prefix}/{combo}: validation path_equals_base identity mismatch")
        deployed_metric, deployed_windows = _reported_metrics(
            deployed_row, validation_history, validation_truth, deployed_path, scales, epsilon, norm_std,
            f"{prefix}/val/{combo}/deployed")
        unique_count = unique_path_count(member_paths)
        require(int(raw_row.get("unique_member_path_count", unique_count)) == unique_count,
                f"{prefix}/{combo}: unique member path count mismatch")
        require(bool(raw_row.get("at_least_two_distinct_members")) is (unique_count >= 2),
                f"{prefix}/{combo}: nontrivial identity flag mismatch")
        require(raw_row.get("members") == list(members), f"{prefix}/{combo}: member order/identity mismatch")
        actual_learned = bool(
            combo == "learned_threeway"
            and float(learned_row.get("selected_alpha", learned_row.get("alpha", 0.0))) > 0
            and not np.array_equal(learned_path, val_base_path)
        )
        declared_learned = raw_row.get("direction_head_changes_path",
                                       raw_row.get("includes_actual_learned_component"))
        if declared_learned is not None:
            require(bool(declared_learned) is actual_learned,
                    f"{prefix}/{combo}: actual learned-direction contribution flag mismatch")
        require(bool(raw_row.get("learned_selected_equals_base")) is np.array_equal(learned_path, val_base_path),
                f"{prefix}/{combo}: learned_selected_equals_base identity mismatch")
        expected_active = bool(combo == "learned_threeway" and actual_learned)
        require(bool(raw_row.get("learned_component_active")) is expected_active,
                f"{prefix}/{combo}: learned component active flag mismatch")
        val_combo_state[combo] = {
            "raw_path": raw_path, "raw_metric": raw_metric, "raw_windows": raw_windows,
            "raw_id": raw_row["candidate_id"], "deployed_path": deployed_path,
            "deployed_metric": deployed_metric, "deployed_windows": deployed_windows,
            "deployed_id": deployed_row["candidate_id"], "gate_passed": passed,
            "fallback": expected_fallback, "unique_member_path_count": unique_count,
            "actual_learned_component": actual_learned,
        }
        check_count += 2
    require(bool(val_combo_state["learned_threeway"]["raw_metric"] is not None), f"{prefix}: missing primary raw score")
    expected_raw_difference = not np.array_equal(val_combo_state["learned_threeway"]["raw_path"],
                                                 val_combo_state["nohead_threeway"]["raw_path"])
    lfd_raw_row = combo_rows["learned_threeway"]["raw"]
    if "raw_learned_differs_from_raw_nohead" in lfd_raw_row:
        require(bool(lfd_raw_row["raw_learned_differs_from_raw_nohead"]) is expected_raw_difference,
                f"{prefix}: raw learned/nohead path identity mismatch")
    expected_raw_equal_flag = not expected_raw_difference
    require(bool(lfd_raw_row.get("raw_learned_threeway_equals_raw_nohead_threeway")) is expected_raw_equal_flag,
            f"{prefix}: learned raw/nohead equality flag mismatch")
    require(bool(combo_rows["nohead_threeway"]["raw"].get("raw_learned_threeway_equals_raw_nohead_threeway")) is expected_raw_equal_flag,
            f"{prefix}: nohead raw learned/nohead equality flag mismatch")

    actual_alpha = float(learned_row.get("selected_alpha", learned_row.get("alpha", 0.0)))
    # Check every test method row against validation-frozen source identities.
    test_methods = test.get("methods")
    require(isinstance(test_methods, dict), f"{prefix}: missing test.methods")
    expected_test_methods = set(STANDALONE_ORDER) | set(ATOMS) | {"learned_selected", "standalone_selected"}
    require(set(test_methods) == expected_test_methods | {f"{combo}_{mode}" for combo in COMBOS for mode in ("raw", "deployed")},
            f"{prefix}: test method coverage mismatch")
    test_method_rows = {method: row for method, row in test_methods.items() if method in expected_test_methods}
    require(all(isinstance(row, dict) for row in test_method_rows.values()),
            f"{prefix}: test method rows must be objects")
    test_paths: dict[str, np.ndarray] = {}
    test_metrics: dict[str, dict[str, Any]] = {}
    test_windows: dict[str, dict[str, np.ndarray]] = {}
    test_ids: dict[str, str] = {}
    for method, row in test_method_rows.items():
        path, _ = _path_row(row, archive, seed, "test", method, f"{prefix}/test/{method}")
        measured, windows = _reported_metrics(row, test_history, test_truth, path, scales,
                                              epsilon, norm_std, f"{prefix}/test/{method}")
        test_paths[method] = path
        test_metrics[method] = measured
        test_windows[method] = windows
        test_ids[method] = row.get("candidate_id")
        check_count += 1
    require("base" in test_paths, f"{prefix}: test methods missing base")
    for method in STANDALONE_ORDER:
        require(method in test_paths, f"{prefix}: test methods missing {method}")
        expected_source = selected_atom_ids[method] if method in selected_atom_ids else method
        require(test_ids[method] == expected_source,
                f"{prefix}: test {method} does not use validation-selected source")
    require("learned_selected" in test_paths and "standalone_selected" in test_paths,
            f"{prefix}: test methods missing selected aliases")
    assert_path(test_paths["learned_selected"], test_paths[learned_winner],
                f"{prefix}: test learned_selected source", exact=True)
    assert_path(test_paths["standalone_selected"], test_paths[standalone_winner],
                f"{prefix}: test standalone_selected source", exact=True)
    require(test_ids["learned_selected"] == "learned_selected"
            and test_method_rows["learned_selected"].get("selected_member") == learned_winner,
            f"{prefix}: test learned_selected source candidate changed")
    require(test_ids["standalone_selected"] == "standalone_selected"
            and test_method_rows["standalone_selected"].get("selected_member") == standalone_winner,
            f"{prefix}: test standalone_selected source candidate changed")

    test_combo_rows = test.get("combinations")
    require(isinstance(test_combo_rows, dict), f"{prefix}: missing test.combinations")
    test_combo_state: dict[str, dict[str, Any]] = {}
    for combo, members in COMBOS.items():
        row = test_combo_rows.get(combo)
        require(isinstance(row, dict) and isinstance(row.get("raw"), dict) and isinstance(row.get("deployed"), dict),
                f"{prefix}/{combo}: missing test raw/deployed rows")
        member_paths = [test_paths[name] for name in members]
        raw_expected = fixed_mean(member_paths, arithmetic)
        raw_row = row["raw"]
        raw_path, _ = _path_row(raw_row, archive, seed, "test", f"{combo}_raw",
                                f"{prefix}/test/{combo}/raw", f"{combo}_raw")
        assert_path(raw_path, raw_expected, f"{prefix}/{combo} test raw formula", exact=True)
        raw_metric, raw_windows = _reported_metrics(raw_row, test_history, test_truth, raw_path, scales,
                                                   epsilon, norm_std, f"{prefix}/test/{combo}/raw")
        passed = val_combo_state[combo]["gate_passed"]
        deployed_row = row["deployed"]
        deployed_path, _ = _path_row(deployed_row, archive, seed, "test", f"{combo}_deployed",
                                     f"{prefix}/test/{combo}/deployed", f"{combo}_deployed")
        expected_deployed = raw_path if passed else test_paths["base"]
        assert_path(deployed_path, expected_deployed, f"{prefix}/{combo} test deployment frozen from validation", exact=True)
        require(deployed_row.get("effective_source_candidate_id") == (raw_row["candidate_id"] if passed else "base"),
                f"{prefix}/{combo}: test effective source changed after validation")
        require(bool(deployed_row.get("fallback_to_base")) is (not passed),
                f"{prefix}/{combo}: test fallback flag mismatch")
        require(bool(deployed_row.get("path_equals_base")) is np.array_equal(deployed_path, test_paths["base"]),
                f"{prefix}/{combo}: test path_equals_base identity mismatch")
        deployed_metric, deployed_windows = _reported_metrics(
            deployed_row, test_history, test_truth, deployed_path, scales, epsilon, norm_std,
            f"{prefix}/test/{combo}/deployed")
        unique_count = unique_path_count(member_paths)
        require(raw_row.get("members") == list(members), f"{prefix}/{combo}: test member order/identity mismatch")
        require(int(raw_row.get("unique_member_path_count", -1)) == unique_count,
                f"{prefix}/{combo}: test unique member path count mismatch")
        require(bool(raw_row.get("at_least_two_distinct_members")) is (unique_count >= 2),
                f"{prefix}/{combo}: test nontrivial identity mismatch")
        learned_test_active = bool(
            "learned_selected" in members and actual_alpha > 0.0
            and not np.array_equal(test_paths["learned_selected"], test_paths["base"])
        )
        require(bool(raw_row.get("learned_selected_equals_base")) is np.array_equal(
            test_paths["learned_selected"], test_paths["base"]),
            f"{prefix}/{combo}: test learned_selected_equals_base mismatch")
        require(bool(raw_row.get("learned_component_active")) is learned_test_active,
                f"{prefix}/{combo}: test learned component active mismatch")
        require(bool(deployed_row.get("learned_component_active")) is (learned_test_active and passed),
                f"{prefix}/{combo}: deployed test learned component active mismatch")
        test_combo_state[combo] = {
            "raw_path": raw_path, "raw_metric": raw_metric, "raw_windows": raw_windows,
            "raw_id": raw_row["candidate_id"], "deployed_path": deployed_path,
            "deployed_metric": deployed_metric, "deployed_windows": deployed_windows,
            "deployed_id": deployed_row["candidate_id"], "fallback": not passed,
            "gate_passed": passed,
        }
        check_count += 2

    # The primary deployed learned mean on old exploratory cells must reproduce
    # the original frozen threeway deployment exactly, bit-for-bit.
    old_match = _audit_old_threeway_source(
        group, seed, test_combo_state["learned_threeway"]["deployed_path"],
        val_combo_state["learned_threeway"]["raw_path"])
    expected_test_raw_equal = np.array_equal(test_combo_state["learned_threeway"]["raw_path"],
                                             test_combo_state["nohead_threeway"]["raw_path"])
    for combo in ("learned_threeway", "nohead_threeway"):
        for mode in ("raw", "deployed"):
            identity_row = test_combo_rows[combo][mode]
            require(bool(identity_row.get("raw_learned_threeway_equals_raw_nohead_threeway")) is expected_test_raw_equal,
                    f"{prefix}/{combo}/{mode}: test learned/nohead raw equality identity mismatch")
    if old_match is None and group_meta.get("cohort") == "exploratory":
        fail(f"{prefix}: exploratory cell lacks source-path provenance for original fixed threeway")
    strict_flags = test.get("strict_joint_flags", {})
    if isinstance(strict_flags, dict):
        for method in COMBOS:
            for mode in ("raw", "deployed"):
                row = test_combo_rows[method][mode]
                score_now = test_combo_state[method][f"{mode}_metric"]
                expected_flag = eligible(score_now, test_metrics["base"])
                key = f"{method}_{mode}"
                if key in strict_flags:
                    require(bool(strict_flags[key]) is expected_flag,
                            f"{prefix}/{key}: reported test strict-joint flag mismatch")

    comparisons = run.get("test_comparisons", {})
    expected_comparisons: dict[str, tuple[str, str]] = {}
    for combo in COMBOS:
        for mode in ("raw", "deployed"):
            candidate_name = f"{combo}_{mode}"
            expected_comparisons[f"{candidate_name}__vs__base"] = (candidate_name, "base")
            expected_comparisons[f"{candidate_name}__vs__standalone_selected"] = (candidate_name, "standalone_selected")
    expected_comparisons["learned_threeway_deployed__vs__nohead_threeway_deployed"] = (
        "learned_threeway_deployed", "nohead_threeway_deployed")
    expected_comparisons["learned_threeway_raw__vs__nohead_threeway_raw"] = (
        "learned_threeway_raw", "nohead_threeway_raw")
    paired_windows: dict[str, dict[str, np.ndarray]] = {}
    for comparison_id, (candidate_name, reference_name) in expected_comparisons.items():
        candidate_combo, candidate_mode = candidate_name.rsplit("_", 1)
        reference_combo, reference_mode = (
            reference_name.rsplit("_", 1) if reference_name.endswith(("_raw", "_deployed"))
            else (reference_name, "")
        )
        candidate_state = test_combo_state[candidate_combo]
        candidate_metric = candidate_state[f"{candidate_mode}_metric"]
        candidate_win = candidate_state[f"{candidate_mode}_windows"]
        if reference_mode:
            reference_state = test_combo_state[reference_combo]
            reference_metric = reference_state[f"{reference_mode}_metric"]
            reference_win = reference_state[f"{reference_mode}_windows"]
        else:
            reference_metric = test_metrics[reference_name]
            reference_win = test_windows[reference_name]
        comparison = comparisons.get(comparison_id) if isinstance(comparisons, dict) else None
        require(isinstance(comparison, dict), f"{prefix}: missing paired test comparison {comparison_id}")
        for identity_key, expected_value in (("candidate", candidate_name), ("candidate_id", candidate_name),
                                             ("reference", reference_name), ("reference_id", reference_name)):
            if identity_key in comparison:
                require(comparison[identity_key] == expected_value,
                        f"{prefix}/{comparison_id}: {identity_key} mismatch")
        _json_delta_report(comparison, candidate_metric, reference_metric, f"{prefix}/{comparison_id}")
        deltas = _test_delta_windows(candidate_win, reference_win)
        paired_windows[comparison_id] = deltas
        seed_delta_keys = comparison.get("seed_window_delta_npz_keys")
        require(isinstance(seed_delta_keys, dict), f"{prefix}/{comparison_id}: missing seed_window_delta_npz_keys")
        require(set(seed_delta_keys) == set(deltas),
                f"{prefix}/{comparison_id}: per-seed window-delta metric coverage mismatch")
        for metric_name, key in seed_delta_keys.items():
            require(isinstance(key, str), f"{prefix}/{comparison_id}/{metric_name}: invalid NPZ key")
            close(get_array(archive, key), deltas[metric_name],
                  f"{prefix}/{comparison_id}/{metric_name} saved per-seed window delta", atol=1e-9)
        check_count += 1

    # Audit every path reference recursively so a JSON-selected method cannot
    # point outside the preserved numeric archive.
    check_count += _audit_json_path_keys(run, archive, prefix)
    for key in ("raw_sign_accuracy", "standardized_mse", "original_units_mse", "mean_3class_accuracy"):
        require(key in test_combo_state["learned_threeway"]["deployed_metric"],
                f"{prefix}: primary deployed metric missing {key}")
    actual_alpha = float(learned_row.get("selected_alpha", learned_row.get("alpha", 0.0)))
    return {
        "seed": int(seed), "checks": check_count,
        "selected_alpha_ids": selected_atom_ids,
        "learned_selected_member": learned_winner,
        "standalone_selected_member": standalone_winner,
        "learned_selected_alpha": actual_alpha,
        "combination_arithmetic": arithmetic,
        "validation": {name: {
            "gate_passed": state["gate_passed"], "fallback_to_base": state["fallback"],
            "raw_metrics": state["raw_metric"], "deployed_metrics": state["deployed_metric"],
            "unique_member_path_count": state["unique_member_path_count"],
            "actual_learned_component": state["actual_learned_component"],
        } for name, state in val_combo_state.items()},
        "test": {name: {
            "gate_passed_on_validation": state["gate_passed"], "fallback_to_base": state["fallback"],
            "raw_metrics": state["raw_metric"], "deployed_metrics": state["deployed_metric"],
        } for name, state in test_combo_state.items()},
        "test_metrics": test_metrics,
        "paired_windows": paired_windows,
        "old_threeway_deployed_exact_match": old_match,
        "prior_validation_scope": prior_scope,
        "origins": {"validation": validation_origins.tolist(), "test": test_origins.tolist()},
    }


def _all_expected_comparisons() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for combo in COMBOS:
        for mode in ("raw", "deployed"):
            candidate = f"{combo}_{mode}"
            out[f"{candidate}__vs__base"] = (candidate, "base")
            out[f"{candidate}__vs__standalone_selected"] = (candidate, "standalone_selected")
    out["learned_threeway_deployed__vs__nohead_threeway_deployed"] = (
        "learned_threeway_deployed", "nohead_threeway_deployed")
    out["learned_threeway_raw__vs__nohead_threeway_raw"] = (
        "learned_threeway_raw", "nohead_threeway_raw")
    return out


def _bootstrap_array_key(spec: dict[str, Any], archive: Any, metric: str,
                         field: str, seed: int | None = None) -> str | None:
    names = ("seedmean_window_delta_npz_keys", "per_seed_window_delta_npz_keys",
             "bootstrap_input_npz_keys")
    for name in names:
        container = spec.get(name)
        if not isinstance(container, dict):
            continue
        if field == "seedmean":
            key = container.get(metric)
            if isinstance(key, str):
                return key
        elif seed is not None:
            if str(seed) in container and isinstance(container[str(seed)], dict):
                key = container[str(seed)].get(metric)
                if isinstance(key, str):
                    return key
            if metric in container and isinstance(container[metric], dict):
                key = container[metric].get(str(seed))
                if isinstance(key, str):
                    return key
    metric_row = spec.get("metrics", {}).get(metric, {}) if isinstance(spec.get("metrics"), dict) else {}
    key_name = "seedmean_window_delta_npz_key" if field == "seedmean" else "seed_window_delta_npz_keys"
    value = metric_row.get(key_name) if isinstance(metric_row, dict) else None
    if field == "per_seed" and isinstance(metric_row, dict):
        value = metric_row.get("per_seed_window_delta_npz_keys", value)
    if field == "seedmean" and isinstance(value, str):
        return value
    if field == "per_seed" and seed is not None and isinstance(value, dict):
        key = value.get(str(seed))
        if isinstance(key, str):
            return key
    return None


def audit_group_bootstrap(group_key: str, group: dict[str, Any], seed_audits: dict[int, dict[str, Any]],
                          archive: Any) -> dict[str, Any]:
    bootstrap = group.get("time_block_bootstrap_after_seed_mean")
    require(isinstance(bootstrap, dict), f"{group_key}: missing seed-mean bootstrap block")
    require(int(bootstrap.get("block_size_windows", bootstrap.get("block_size", -1))) == 8,
            f"{group_key}: bootstrap block size is not 8")
    require(int(bootstrap.get("repeats", -1)) == 2000, f"{group_key}: bootstrap repeats are not 2000")
    require(int(bootstrap.get("seed", bootstrap.get("random_seed", -1))) == 0,
            f"{group_key}: bootstrap seed is not 0")
    specs = bootstrap.get("comparisons")
    require(isinstance(specs, dict), f"{group_key}: missing bootstrap comparisons")
    expected_comparisons = _all_expected_comparisons()
    require(set(specs) == set(expected_comparisons),
            f"{group_key}: bootstrap comparison coverage differs: missing={sorted(set(expected_comparisons)-set(specs))}, extra={sorted(set(specs)-set(expected_comparisons))}")
    seed_ids = sorted(seed_audits)
    origins_count = len(seed_audits[seed_ids[0]]["origins"]["test"])
    index_key = bootstrap.get("bootstrap_indices_npz_key")
    require(isinstance(index_key, str), f"{group_key}: missing bootstrap index key")
    expected_block = min(8, origins_count)
    blocks_needed = int(math.ceil(origins_count / expected_block))
    rng = np.random.default_rng(0)
    starts = rng.integers(0, origins_count - expected_block + 1,
                          size=(2000, blocks_needed))
    expected_indices = (starts[:, :, None] + np.arange(expected_block, dtype=np.int64)[None, None, :]).reshape(2000, -1)[:, :origins_count]
    require(np.array_equal(get_array(archive, index_key), expected_indices),
            f"{group_key}: saved moving-block bootstrap indices differ from frozen seed-0 rule")
    intervals: dict[str, Any] = {}
    checks = 0
    for comparison_id, identities in expected_comparisons.items():
        spec = specs[comparison_id]
        require(isinstance(spec, dict), f"{group_key}/{comparison_id}: bootstrap spec must be an object")
        if "candidate_id" in spec:
            require(spec["candidate_id"] == identities[0], f"{group_key}/{comparison_id}: CI candidate identity mismatch")
        if "reference_id" in spec:
            require(spec["reference_id"] == identities[1], f"{group_key}/{comparison_id}: CI reference identity mismatch")
        metric_rows = spec.get("metrics")
        require(isinstance(metric_rows, dict), f"{group_key}/{comparison_id}: missing metric intervals")
        expected_metrics = set(seed_audits[seed_ids[0]]["paired_windows"][comparison_id])
        require(set(metric_rows) == expected_metrics,
                f"{group_key}/{comparison_id}: CI metric coverage mismatch")
        intervals[comparison_id] = {}
        for metric in sorted(expected_metrics):
            seed_deltas = [np.asarray(seed_audits[seed]["paired_windows"][comparison_id][metric], dtype=np.float64)
                           for seed in seed_ids]
            require(len({len(delta) for delta in seed_deltas}) == 1,
                    f"{group_key}/{comparison_id}/{metric}: per-seed window lengths differ")
            expected_seedmean = np.stack(seed_deltas, axis=0).mean(axis=0)
            for seed in seed_ids:
                key = _bootstrap_array_key(spec, archive, metric, "per_seed", seed)
                require(key is not None, f"{group_key}/{comparison_id}/{metric}: missing saved seed {seed} bootstrap input key")
                close(get_array(archive, key), seed_deltas[seed_ids.index(seed)],
                      f"{group_key}/{comparison_id}/{metric}: seed {seed} bootstrap input", atol=1e-9)
            mean_key = _bootstrap_array_key(spec, archive, metric, "seedmean")
            require(mean_key is not None, f"{group_key}/{comparison_id}/{metric}: missing seed-mean bootstrap input key")
            close(get_array(archive, mean_key), expected_seedmean,
                  f"{group_key}/{comparison_id}/{metric}: seed-mean bootstrap input", atol=1e-9)
            expected_ci = bootstrap_ci(expected_seedmean, 8, 2000, 0)
            reported = metric_rows[metric]
            require(isinstance(reported, dict), f"{group_key}/{comparison_id}/{metric}: interval must be an object")
            ci_summary = reported.get("bootstrap_ci", reported)
            require(isinstance(ci_summary, dict),
                    f"{group_key}/{comparison_id}/{metric}: missing bootstrap_ci summary")
            close(ci_summary.get("mean"), expected_ci["mean"],
                  f"{group_key}/{comparison_id}/{metric}: seed-mean point estimate")
            close(ci_summary.get("ci95"), expected_ci["ci95"],
                  f"{group_key}/{comparison_id}/{metric}: moving-block CI")
            require(bool(reported.get("seed_mean_before_block_resampling")) is True,
                    f"{group_key}/{comparison_id}/{metric}: bootstrap order is not seed-mean first")
            require(int(reported.get("repeats", -1)) == 2000
                    and int(reported.get("random_seed", -1)) == 0,
                    f"{group_key}/{comparison_id}/{metric}: per-metric bootstrap configuration mismatch")
            require(reported.get("bootstrap_indices_npz_key") == index_key,
                    f"{group_key}/{comparison_id}/{metric}: bootstrap index identity mismatch")
            if "windows" in reported:
                require(int(reported["windows"]) == len(expected_seedmean),
                        f"{group_key}/{comparison_id}/{metric}: reported window count mismatch")
            intervals[comparison_id][metric] = expected_ci
            checks += 1 + len(seed_ids)
    return {"intervals": intervals, "checks": checks,
            "definition": "per-seed paired window deltas averaged across fitted seeds, then non-circular 8-window moving-block bootstrap; seed 0; 2000 resamples"}


def _group_metric_summary(seed_audits: dict[int, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_ids = sorted(seed_audits)
    method_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in seed_ids:
        state = seed_audits[seed]
        metrics = {name: value for name, value in state["test_metrics"].items()}
        for combo, row in state["test"].items():
            metrics[f"{combo}_raw"] = row["raw_metrics"]
            metrics[f"{combo}_deployed"] = row["deployed_metrics"]
        for method, metric in metrics.items():
            method_metrics.setdefault(method, {})[str(seed)] = metric
    summaries = {}
    for method, per_seed in method_metrics.items():
        require(set(map(int, per_seed)) == set(seed_ids), f"group metric {method}: incomplete seed coverage")
        record: dict[str, Any] = {"n_seeds": len(seed_ids), "per_seed": per_seed}
        for key in ("raw_sign_accuracy", "standardized_mse", "original_units_mse",
                    "mean_3class_accuracy", "mean_macro_f1"):
            values = np.asarray([per_seed[str(seed)][key] for seed in seed_ids], dtype=np.float64)
            record[key] = float(values.mean())
            record[key + "_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        for field in ("accuracy_by_scale", "macro_f1_by_scale"):
            scales = sorted(per_seed[str(seed_ids[0])][field], key=lambda value: int(value))
            record[field] = {
                scale: float(np.mean([per_seed[str(seed)][field][scale] for seed in seed_ids]))
                for scale in scales
            }
        summaries[method] = record

    pair_summaries = {}
    for comparison_id, (candidate, reference) in _all_expected_comparisons().items():
        seed_rows = []
        for seed in seed_ids:
            row = seed_audits[seed]
            if candidate == "base" or candidate == "standalone_selected":
                candidate_metric = row["test_metrics"][candidate]
            else:
                combo, mode = candidate.rsplit("_", 1)
                candidate_metric = row["test"][combo][f"{mode}_metrics"]
            if reference in ("base", "standalone_selected"):
                reference_metric = row["test_metrics"][reference]
            else:
                combo, mode = reference.rsplit("_", 1)
                reference_metric = row["test"][combo][f"{mode}_metrics"]
            deltas = {key: float(candidate_metric[key] - reference_metric[key]) for key in (
                "raw_sign_accuracy", "standardized_mse", "original_units_mse",
                "mean_3class_accuracy", "mean_macro_f1")}
            deltas["standardized_mse_percent_change"] = (
                float(100.0 * (candidate_metric["standardized_mse"] / reference_metric["standardized_mse"] - 1.0))
                if reference_metric["standardized_mse"] else None
            )
            deltas["original_units_mse_percent_change"] = (
                float(100.0 * (candidate_metric["original_units_mse"] / reference_metric["original_units_mse"] - 1.0))
                if reference_metric["original_units_mse"] else None
            )
            deltas["joint"] = eligible(candidate_metric, reference_metric)
            deltas["candidate_metrics"] = candidate_metric
            deltas["reference_metrics"] = reference_metric
            seed_rows.append({"seed": int(seed), **deltas})
        candidate_means = summaries[candidate]
        reference_means = summaries[reference]
        direction_delta = candidate_means["raw_sign_accuracy"] - reference_means["raw_sign_accuracy"]
        mse_pct = 100.0 * (candidate_means["standardized_mse"] / reference_means["standardized_mse"] - 1.0)
        pair_summaries[comparison_id] = {
            "candidate": candidate, "reference": reference, "per_seed": seed_rows,
            "mean_raw_sign_delta": float(direction_delta),
            "mean_raw_sign_delta_pp": float(direction_delta * 100.0),
            "mean_standardized_mse_percent_change": float(mse_pct),
            "mean_joint": bool(direction_delta > 0 and candidate_means["standardized_mse"] <= reference_means["standardized_mse"] + STRICT_MSE_TOLERANCE),
            "joint_seed_count": sum(bool(row["joint"]) for row in seed_rows),
            "raw_sign_delta_seed_std": float(np.std([row["raw_sign_accuracy"] for row in seed_rows], ddof=1)) if len(seed_rows) > 1 else 0.0,
            "standardized_mse_delta_seed_std": float(np.std([row["standardized_mse"] for row in seed_rows], ddof=1)) if len(seed_rows) > 1 else 0.0,
        }
    return summaries, pair_summaries


def _verify_group_metadata(group_key: str, group_meta: dict[str, Any], seeds: dict[str, Any]) -> dict[str, Any]:
    dataset = str(group_meta.get("dataset", ""))
    backbone = str(group_meta.get("backbone", "")).lower()
    cohort = group_meta.get("cohort") or ("additional_validation" if len(seeds) == 5 else "exploratory")
    require(backbone in ("linear", "dlinear"), f"{group_key}: unexpected backbone {backbone!r}")
    seed_ids = sorted(int(seed) for seed in seeds)
    expected_seed_ids = list(range(5)) if cohort == "additional_validation" else [0, 1, 2]
    require(seed_ids == expected_seed_ids, f"{group_key}: expected seeds {expected_seed_ids}, got {seed_ids}")
    input_columns = group_meta.get("input_columns")
    require(isinstance(input_columns, list) and input_columns, f"{group_key}: missing input_columns")
    if cohort == "additional_validation" and dataset.lower() in ("electricity", "traffic"):
        require(len(input_columns) == 8, f"{group_key}: fresh verification must use exactly eight channels")
    split = group_meta.get("split_metadata")
    require(isinstance(split, dict), f"{group_key}: missing split_metadata")
    bounds = split.get("split_bounds", split.get("bounds"))
    require(isinstance(bounds, list) and len(bounds) == 5, f"{group_key}: missing five split boundaries")
    bounds = [int(value) for value in bounds]
    require(bounds[0] == 0 and all(b > a for a, b in zip(bounds, bounds[1:])),
            f"{group_key}: split bounds are invalid")
    ratios = split.get("split_ratios", [0.6, 0.15, 0.1, 0.15])
    close(ratios, [0.6, 0.15, 0.1, 0.15], f"{group_key}: data split ratios")
    require(int(split.get("lookback", 96)) == 96 and int(split.get("horizon", 24)) == 24,
            f"{group_key}: lookback/horizon differ from protocol")
    require(int(split.get("eval_stride", 24)) == 24, f"{group_key}: eval stride differs from protocol")
    require(group_meta.get("combination_arithmetic") in (None, "sum_divide", "sum_reciprocal_multiply"),
            f"{group_key}: unsupported combination arithmetic metadata")
    return {"dataset": dataset, "backbone": backbone, "cohort": cohort,
            "input_columns": input_columns, "seeds": seed_ids,
            "split_bounds": bounds, "lookback": int(split.get("lookback", 96)),
            "horizon": int(split.get("horizon", 24)), "eval_stride": int(split.get("eval_stride", 24))}


def _verify_origins(group_key: str, seed: int, group_meta: dict[str, Any],
                    split_name: str, origins: np.ndarray) -> None:
    split_meta = group_meta["split_metadata"]
    bounds = [int(value) for value in split_meta.get("split_bounds", split_meta.get("bounds"))]
    index = 2 if split_name == "validation" else 3
    lookback = int(split_meta.get("lookback", 96))
    horizon = int(split_meta.get("horizon", 24))
    stride = int(split_meta.get("eval_stride", 24))
    start, end = bounds[index], bounds[index + 1]
    expected = np.arange(max(start - 1, lookback - 1), end - horizon, stride, dtype=np.int64)
    require(np.array_equal(origins, expected),
            f"{group_key}/seed{seed}/{split_name}: origins differ from recorded temporal split")


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _verify_hash(path: Path, expected: str | None, label: str) -> str:
    require(path.is_file(), f"{label}: missing file {path}")
    actual = sha256_file(path)
    if expected:
        require(actual == expected, f"{label}: sha256 mismatch for {path}")
    return actual


def _verify_source_provenance(root: Path, group_meta: dict[str, Any], group_key: str) -> dict[str, Any]:
    provenance = group_meta.get("source_provenance") or group_meta.get("source")
    require(isinstance(provenance, dict), f"{group_key}: missing source_provenance")
    require(provenance.get("reload_verified") is True,
            f"{group_key}: checkpoint reload verification is not true")
    verified = {}
    pairs = (
        ("source_config_path", ("source_config_sha256", "config_sha256")),
        ("source_results_path", ("source_results_sha256", "results_sha256")),
        ("source_checkpoint_path", ("source_checkpoint_sha256", "checkpoint_sha256")),
        ("source_prediction_npz_path", ("source_prediction_npz_sha256",)),
        ("data_file", ("data_sha256",)),
    )
    for path_name, hash_names in pairs:
        path_value = provenance.get(path_name)
        hash_name = next((name for name in hash_names if isinstance(provenance.get(name), str)), None)
        expected = provenance.get(hash_name) if hash_name else None
        require(isinstance(path_value, str) and isinstance(expected, str),
                f"{group_key}: missing provenance pair {path_name}/{hash_names}")
        path = _resolve_path(root, path_value)
        verified[hash_name] = _verify_hash(path, expected, f"{group_key}/{path_name}")
    config_path = _resolve_path(root, provenance["source_config_path"])
    results_path = _resolve_path(root, provenance["source_results_path"])
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    source_results = json.loads(results_path.read_text(encoding="utf-8"))
    require(source_config.get("smoke", False) is False, f"{group_key}: source config is smoke")
    require(source_config.get("data", {}).get("smoke", False) is False, f"{group_key}: source data config is smoke")
    require(source_results.get("smoke", False) is False, f"{group_key}: source results are smoke")
    require(source_results.get("checkpoint_reload_verified") is True,
            f"{group_key}: source results do not confirm checkpoint reload")
    require(source_results.get("data", {}).get("smoke_truncated", False) is False,
            f"{group_key}: source data were smoke truncated")
    columns = source_config.get("data", {}).get("value_columns")
    declared_columns = group_meta.get("input_columns")
    if columns is not None and declared_columns is not None:
        require(list(columns) == list(declared_columns), f"{group_key}: source config columns differ from group metadata")
    configured_seeds = source_config.get("seeds")
    if configured_seeds is not None:
        expected_seed_ids = group_meta.get("seed_ids")
        if expected_seed_ids is not None:
            require(sorted(map(int, configured_seeds)) == sorted(map(int, expected_seed_ids)),
                    f"{group_key}: source config seed list differs from group metadata")
    snapshot_value = provenance.get("evaluator_snapshot")
    evaluator_hash = provenance.get("evaluator_sha256")
    if snapshot_value is not None:
        require(isinstance(snapshot_value, str) and isinstance(evaluator_hash, str),
                f"{group_key}: incomplete evaluator snapshot provenance")
        verified["evaluator_sha256"] = _verify_hash(
            _resolve_path(root, snapshot_value), evaluator_hash, f"{group_key}/evaluator snapshot")
    evaluator_root_value = provenance.get("evaluator_root")
    if evaluator_root_value:
        evaluator_root = _resolve_path(root, evaluator_root_value)
        root_manifest = evaluator_root / "output_manifest.json"
        root_results = evaluator_root / "results.json"
        root_npz = evaluator_root / "predictions.npz"
        if group_meta.get("cohort") == "exploratory":
            for path in (root_manifest, root_results, root_npz):
                require(path.is_file(), f"{group_key}: missing frozen source-root artifact {path}")
            manifest = json.loads(root_manifest.read_text(encoding="utf-8"))
            require(manifest.get("complete") is True, f"{group_key}: frozen source evaluator root incomplete")
            if provenance.get("evaluator_manifest_sha256"):
                _verify_hash(root_manifest, provenance["evaluator_manifest_sha256"], f"{group_key}/source evaluator manifest")
            for path, key in ((root_results, "source_evaluator_results_sha256"),
                              (root_npz, "source_evaluator_prediction_npz_sha256")):
                if provenance.get(key):
                    _verify_hash(path, provenance[key], f"{group_key}/{key}")
            source_snapshot = Path(manifest.get("script_snapshot", ""))
            require(source_snapshot.is_file() and sha256_file(source_snapshot) == manifest.get("script_sha256"),
                    f"{group_key}: frozen source evaluator snapshot hash mismatch")
        verified["source_evaluator_root"] = str(evaluator_root)
    # Verify optional original/prepared data path pairs recorded for the
    # first-eight-channel derived dataset; never parse test values here.
    for path_name, hash_name in (
        ("original_data_file", "original_data_sha256"),
        ("original_data_path", "original_data_sha256"),
        ("prepared_data_file", "prepared_data_sha256"),
        ("derived_data_file", "derived_data_sha256"),
        ("derived_data_path", "derived_data_sha256"),
    ):
        if provenance.get(path_name) is not None:
            require(isinstance(provenance.get(hash_name), str), f"{group_key}: missing {hash_name}")
            verified[hash_name] = _verify_hash(_resolve_path(root, provenance[path_name]),
                                               provenance[hash_name], f"{group_key}/{path_name}")
    return verified


def _verify_output_files(root: Path, result: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(manifest.get("complete") is True, "output manifest complete flag is not true")
    groups = result.get("groups")
    require(isinstance(groups, dict) and groups, "aggregate results has no groups")
    requested = manifest.get("requested_groups")
    completed = manifest.get("completed_groups")
    if requested is not None or completed is not None:
        require(set(requested or []) == set(completed or []), "manifest requested/completed groups differ")
        require(set(groups) == set(completed or []), "aggregate results groups differ from completed manifest groups")
    outputs: dict[str, dict[str, Any]] = {}
    for group_key, group in groups.items():
        artifacts = group.get("artifacts")
        require(isinstance(artifacts, dict), f"{group_key}: missing per-cell artifacts block")
        require(isinstance(artifacts.get("results_json"), str), f"{group_key}: missing artifacts.results_json")
        require(isinstance(artifacts.get("predictions_npz"), str), f"{group_key}: missing artifacts.predictions_npz")
        require(isinstance(artifacts.get("manifest_json"), str), f"{group_key}: missing artifacts.manifest_json")
        resolved = {name: _resolve_path(root, artifacts[name])
                    for name in ("results_json", "predictions_npz", "manifest_json")}
        for name, path in resolved.items():
            require(path.is_file(), f"{group_key}: missing output artifact {path}")
        npz_sha = artifacts.get("predictions_npz_sha256")
        require(isinstance(npz_sha, str), f"{group_key}: missing predictions_npz_sha256")
        _verify_hash(resolved["predictions_npz"], npz_sha, f"{group_key}/output predictions_npz")
        cell_manifest = json.loads(resolved["manifest_json"].read_text(encoding="utf-8"))
        require(cell_manifest.get("complete") is True, f"{group_key}: cell manifest incomplete")
        cell_artifacts = cell_manifest.get("artifacts")
        require(isinstance(cell_artifacts, dict), f"{group_key}: cell manifest has no artifact hashes")
        for name in ("results_json", "predictions_npz"):
            record = cell_artifacts.get(name)
            require(isinstance(record, dict) and isinstance(record.get("path"), str)
                    and isinstance(record.get("sha256"), str),
                    f"{group_key}: malformed cell manifest artifact {name}")
            path = _resolve_path(root, record["path"])
            require(path == resolved[name], f"{group_key}: group and cell manifest disagree on {name} path")
            _verify_hash(path, record["sha256"], f"{group_key}/cell {name}")
        require(cell_manifest.get("group_key") == group_key,
                f"{group_key}: cell manifest group key mismatch")
        cell_results = json.loads(resolved["results_json"].read_text(encoding="utf-8"))
        require(cell_results == group, f"{group_key}: aggregate group differs from per-cell results")
        manifest_group_artifacts = manifest.get("cell_artifacts", {}).get(group_key)
        require(manifest_group_artifacts == artifacts,
                f"{group_key}: root manifest and aggregate artifact references differ")
        outputs[group_key] = {**resolved, "cell_manifest": cell_manifest, "cell_results": cell_results}
    return groups, outputs


def _check_summary_rows(group_key: str, group: dict[str, Any],
                        method_summaries: dict[str, Any]) -> int:
    """Recompute every evaluator test summary mean, sample std, and seed vector."""
    summary = group.get("group_test_summary", group.get("test_summary_mean_sample_std"))
    require(isinstance(summary, dict), f"{group_key}: missing evaluator test mean/sample-std summary")
    require(set(summary) == set(method_summaries),
            f"{group_key}: summary method coverage mismatch; missing={sorted(set(method_summaries)-set(summary))}, extra={sorted(set(summary)-set(method_summaries))}")
    checked = 0
    for method, actual in method_summaries.items():
        row = summary.get(method)
        require(isinstance(row, dict), f"{group_key}/{method}: summary row missing")
        for metric in ("raw_sign_accuracy", "standardized_mse", "original_units_mse",
                       "mean_3class_accuracy", "mean_macro_f1"):
            metric_row = row.get(metric)
            require(isinstance(metric_row, dict), f"{group_key}/{method}: missing nested summary for {metric}")
            close(metric_row.get("mean"), actual[metric], f"{group_key}/group_summary/{method}.{metric}.mean")
            close(metric_row.get("sample_std"), actual[metric + "_seed_std"],
                  f"{group_key}/group_summary/{method}.{metric}.sample_std")
            per_seed = metric_row.get("per_seed")
            expected_per_seed = [actual["per_seed"][str(seed)][metric] for seed in sorted(map(int, actual["per_seed"]))]
            close(per_seed, expected_per_seed, f"{group_key}/group_summary/{method}.{metric}.per_seed")
            checked += 3
        for field in ("accuracy_by_scale", "macro_f1_by_scale"):
            scale_rows = row.get(field)
            require(isinstance(scale_rows, dict) and set(scale_rows) == set(actual[field]),
                    f"{group_key}/{method}: {field} scale coverage mismatch")
            for scale, mean_value in actual[field].items():
                scale_row = scale_rows[scale]
                require(isinstance(scale_row, dict), f"{group_key}/{method}/{field}/{scale}: summary row missing")
                per_seed = [actual["per_seed"][str(seed)][field][scale]
                            for seed in sorted(map(int, actual["per_seed"]))]
                close(scale_row.get("mean"), mean_value, f"{group_key}/{method}/{field}/{scale}.mean")
                expected_std = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
                close(scale_row.get("sample_std"), expected_std,
                      f"{group_key}/{method}/{field}/{scale}.sample_std")
                close(scale_row.get("per_seed"), per_seed,
                      f"{group_key}/{method}/{field}/{scale}.per_seed")
                checked += 3
    return checked


def audit_root(root: Path, expected_cohort: str | None = None) -> dict[str, Any]:
    root = root.expanduser().resolve()
    results_path = root / "results.json"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = root / "output_manifest.json"
    require(results_path.is_file() and manifest_path.is_file(), f"root lacks results.json/manifest.json: {root}")
    result = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups, output_artifacts = _verify_output_files(root, result, manifest)
    if expected_cohort:
        actual_cohort = manifest.get("cohort", result.get("cohort"))
        require(actual_cohort == expected_cohort,
                f"expected cohort {expected_cohort}, got {actual_cohort}")
    snapshot_value = manifest.get("evaluator_snapshot", manifest.get("script_snapshot"))
    snapshot_hash = manifest.get("evaluator_sha256", manifest.get("script_sha256"))
    if snapshot_value is not None:
        require(isinstance(snapshot_hash, str), "root manifest misses evaluator snapshot hash")
        _verify_hash(_resolve_path(root, snapshot_value), snapshot_hash, "root evaluator snapshot")
    if manifest.get("results_sha256"):
        _verify_hash(results_path, manifest["results_sha256"], "aggregate results")
    report_groups = []
    total_checks = 0
    for group_key, group in groups.items():
        meta = dict(group)
        seeds = group.get("seeds")
        require(isinstance(seeds, dict) and seeds, f"{group_key}: missing seed results")
        meta["seed_ids"] = sorted(int(seed) for seed in seeds)
        if "source_provenance" not in meta and isinstance(meta.get("source"), dict):
            meta["source_provenance"] = meta["source"]
        if not meta.get("cohort"):
            meta["cohort"] = manifest.get("cohort") or result.get("group_cohorts", {}).get(group_key)
        normalized = _verify_group_metadata(group_key, meta, seeds)
        meta["cohort"] = normalized["cohort"]
        provenance_hashes = _verify_source_provenance(root, meta, group_key)
        arithmetic = _mean_operation(group)
        source_provenance = meta.get("source_provenance", {})
        declared_device = (source_provenance.get("source_device") or source_provenance.get("device")
                           or meta.get("source_device") or meta.get("device"))
        if normalized["cohort"] == "additional_validation" and normalized["dataset"].lower() in ("electricity", "traffic"):
            require(arithmetic == "sum_reciprocal_multiply",
                    f"{group_key}: new CUDA fixed means must use saved reciprocal-multiply float32 arithmetic")
        artifacts = output_artifacts[group_key]
        cell_results = json.loads(artifacts["results_json"].read_text(encoding="utf-8"))
        require(cell_results == group, f"{group_key}: aggregate cell record differs from per-cell result file")
        seed_audits: dict[int, dict[str, Any]] = {}
        with np.load(artifacts["predictions_npz"], allow_pickle=False) as archive:
            for seed_key, run in sorted(seeds.items(), key=lambda item: int(item[0])):
                seed = int(seed_key)
                require(int(run.get("seed", seed)) == seed, f"{group_key}: seed key/value mismatch")
                audit = audit_seed(group_key, group, seed, run, archive, arithmetic, meta)
                for split_name in ("validation", "test"):
                    _verify_origins(group_key, seed, meta, split_name,
                                    np.asarray(audit["origins"][split_name], dtype=np.int64))
                seed_audits[seed] = audit
            # All fitted seeds must see identical evaluation windows; they are
            # paired model refits over one fixed dataset, not independent data.
            first_seed = sorted(seed_audits)[0]
            for seed in sorted(seed_audits)[1:]:
                for split_name in ("validation", "test"):
                    require(seed_audits[seed]["origins"][split_name] == seed_audits[first_seed]["origins"][split_name],
                            f"{group_key}: origins differ across fitted seeds for {split_name}")
            bootstrap_report = audit_group_bootstrap(group_key, group, seed_audits, archive)
            total_checks += bootstrap_report["checks"]
        method_summaries, paired_summaries = _group_metric_summary(seed_audits)
        summary_checks = _check_summary_rows(group_key, group, method_summaries)
        total_checks += summary_checks
        seed_checks = sum(audit["checks"] for audit in seed_audits.values())
        total_checks += seed_checks
        report_groups.append({
            "group_key": group_key,
            "dataset": normalized["dataset"],
            "backbone": normalized["backbone"],
            "cohort": normalized["cohort"],
            "input_columns": normalized["input_columns"],
            "seeds": {str(seed): {key: value for key, value in audit.items()
                                  if key not in ("paired_windows", "origins", "test_metrics")}
                      for seed, audit in seed_audits.items()},
            "method_mean_sample_std": method_summaries,
            "paired_comparisons": paired_summaries,
            "time_block_bootstrap_after_seed_mean": bootstrap_report,
            "source_hashes": provenance_hashes,
            "combination_arithmetic": arithmetic,
            "source_device": declared_device,
            "source_snapshot_sha256": source_provenance.get("evaluator_sha256"),
            "prior_check_scope": "saved train-prior counts to saved prior probabilities; not rebuilt from CSV labels",
            "checks": seed_checks + bootstrap_report["checks"] + summary_checks,
        })
    require(len(report_groups) == len(groups), "audit result omitted groups")
    cohort_counts: dict[str, int] = {}
    for row in report_groups:
        cohort_counts[row["cohort"]] = cohort_counts.get(row["cohort"], 0) + 1
    return {
        "passed": True,
        "audit": "independent_fixed_threeway_numpy",
        "root": str(root),
        "cohort_counts": cohort_counts,
        "n_groups": len(report_groups),
        "checks": total_checks,
        "protocol_sha256": sha256_file(Path(__file__).with_name("fixed_threeway_verification_protocol_2026-09-28.md")),
        "input_artifacts": {
            "results_json": {"path": str(results_path), "sha256": sha256_file(results_path)},
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        },
        "bootstrap": {"block_size_windows": 8, "repeats": 2000, "seed": 0,
                      "order": "paired window difference; average across fitted seeds first; non-circular moving blocks"},
        "prior_check_scope": "saved training class counts are checked against saved prior probabilities; no CSV label rebuild",
        "groups": report_groups,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="frozen fixed-threeway aggregate output root")
    parser.add_argument("--report", type=Path, help="audit JSON destination; defaults to ROOT/independent_audit.json")
    parser.add_argument("--expected-cohort", help="optional cohort assertion")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    report = audit_root(root, args.expected_cohort)
    source = Path(__file__).resolve()
    snapshot = root / source.name
    if snapshot != source:
        shutil.copy2(source, snapshot)
    report["auditor"] = {"source": str(source), "snapshot": str(snapshot), "sha256": sha256_file(source)}
    destination = args.report or root / "independent_audit.json"
    _write_json(destination, report)
    print(json.dumps({"passed": report["passed"], "groups": report["n_groups"],
                      "checks": report["checks"], "report": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
