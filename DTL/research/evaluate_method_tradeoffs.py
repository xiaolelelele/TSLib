"""Paired per-method direction/MSE evidence from saved trend_lab artifacts.

This reads JSON and NPZ artifacts only. It never loads a checkpoint or pickle.
Each dataset/backbone group is summarized independently; methods are never
averaged together and training seeds are averaged before the time-block
bootstrap is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EPS_TOL = 1e-8
BOOTSTRAP_REPEATS = 2000
BLOCK_SIZE = 8
BOOTSTRAP_SEED = 20260927

CLASSIFIER_ONLY = {"direction_head", "majority", "last_direction"}
MODEL_TYPES = {
    "base": "shared_base_forecaster",
    "difference": "difference_mse_forecaster",
    "direction_head": "classifier_only_independent_head",
    "projection": "posthoc_multiscale_projection",
    "projection_single": "posthoc_single_scale_projection",
    "residual": "learned_residual_adapter",
    "residual_direction": "direction_supervised_residual_adapter",
    "event": "event_hazard_trajectory_adapter",
    "event_nohazard": "nohazard_trajectory_adapter",
    "last_value": "naive_last_value_forecaster",
    "last_slope": "naive_last_slope_forecaster",
    "majority": "classifier_only_training_majority",
    "last_direction": "classifier_only_last_history_direction",
}


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _seed_key(seed: int, name: str) -> str:
    return f"seed{seed}_{name}"


def _find_artifact_roots(inputs: list[Path]) -> list[Path]:
    roots: set[Path] = set()
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file() and path.name == "results.json":
            roots.add(path.parent)
        elif path.is_dir() and (path / "results.json").is_file():
            roots.add(path)
        elif path.is_dir():
            for result_path in path.rglob("results.json"):
                candidate = result_path.parent
                if not (candidate / "config.json").is_file():
                    continue
                # Full experiment roots can contain a separately labelled
                # accelerator smoke check; it is not one of the dataset cells.
                if any("smoke" in part.lower() for part in candidate.relative_to(path).parts):
                    continue
                roots.add(candidate)
        else:
            raise FileNotFoundError(f"input has no results.json: {path}")
    if not roots:
        raise FileNotFoundError("no results.json/config.json artifact pairs found")
    return sorted(roots)


def _dataset_name(config: dict[str, Any], result: dict[str, Any], root: Path) -> str:
    data = config.get("data") or {}
    source = data.get("csv_path") or data.get("path")
    if source:
        name = Path(str(source)).stem
        if name:
            return name
    metadata = result.get("data") or {}
    if metadata.get("scenario"):
        return str(metadata["scenario"])
    return str(metadata.get("kind") or root.parent.name)


def _model_type(method: str) -> str:
    return MODEL_TYPES.get(method, "unknown_method_type")


def _direction_labels(history: np.ndarray, future: np.ndarray,
                      scales: list[int], epsilon: np.ndarray) -> np.ndarray:
    """Return [window,scale,horizon,channel] labels: down=0, flat=1, up=2."""
    history = np.asarray(history, dtype=np.float64)
    future = np.asarray(future, dtype=np.float64)
    joined = np.concatenate((history, future), axis=1)
    steps = np.arange(future.shape[1]) + history.shape[1]
    slopes = np.stack([(joined[:, steps] - joined[:, steps - k]) / k
                       for k in scales], axis=1)
    eps = np.asarray(epsilon, dtype=np.float64)[None, :, None, :]
    return np.where(slopes > eps, 2, np.where(slopes < -eps, 0, 1))


def _per_window_metrics(history: np.ndarray, truth: np.ndarray, pred: np.ndarray,
                        scales: list[int], epsilon: np.ndarray,
                        channel_std: np.ndarray | None) -> dict[str, np.ndarray]:
    history = np.asarray(history, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    true_labels = _direction_labels(history, truth, scales, epsilon)
    pred_labels = _direction_labels(history, pred, scales, epsilon)
    correct = (true_labels == pred_labels).mean(axis=(2, 3))
    raw_truth = np.sign(np.diff(np.concatenate((history[:, -1:, :], truth), axis=1), axis=1))
    raw_pred = np.sign(np.diff(np.concatenate((history[:, -1:, :], pred), axis=1), axis=1))
    err2 = np.square(pred - truth)
    out = {
        "raw_sign_accuracy": (raw_truth == raw_pred).mean(axis=(1, 2)),
        "mean_scales_accuracy": correct.mean(axis=1),
        "mse": err2.mean(axis=(1, 2)),
    }
    for i, scale in enumerate(scales):
        out[f"accuracy_scale_{scale}"] = correct[:, i]
    if channel_std is not None and len(channel_std) == err2.shape[-1]:
        out["original_mse"] = (err2 * np.square(channel_std)[None, None, :]).mean(axis=(1, 2))
    return out


def _stats(values: list[float | None]) -> dict[str, float | int | None]:
    clean = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if not len(clean):
        return {"n": 0, "mean": None, "std": None}
    return {"n": int(len(clean)), "mean": float(clean.mean()),
            "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0}


def _moving_block_ci(values: np.ndarray, block_size: int = BLOCK_SIZE,
                     repeats: int = BOOTSTRAP_REPEATS,
                     seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Bootstrap a seed-averaged per-window paired difference in time blocks."""
    v = np.asarray(values, dtype=float).reshape(-1)
    if not len(v) or not np.isfinite(v).all():
        return {"mean": None, "ci95": [None, None], "windows": int(len(v))}
    rng = np.random.default_rng(seed)
    n = len(v)
    block = min(block_size, n)
    needed = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(repeats, needed))
    indices = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(repeats, -1)[:, :n]
    boot = v[indices].mean(axis=1)
    return {"mean": float(v.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "windows": int(n)}


def _record_metrics(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("metrics") or {}


def _classifier_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = _record_metrics(record)
    probability = metrics.get("probability") or {}
    if probability:
        return probability
    return record.get("direction_metrics", {}).get("overall") or {}


def _scale_classifier_accuracy(record: dict[str, Any], scale: int) -> float | None:
    metrics = _record_metrics(record)
    prob = metrics.get("probability") or {}
    by_scale = prob.get("by_scale") or record.get("direction_metrics", {}).get("by_scale") or {}
    item = by_scale.get(str(scale)) or by_scale.get(scale) or {}
    return _finite_number(item.get("accuracy"))


def _raw_seed(root: Path, result: dict[str, Any], archive: Any,
              seed: int, method: str, record: dict[str, Any],
              scales: list[int], epsilon: np.ndarray,
              channel_std: np.ndarray | None) -> dict[str, Any]:
    prefix = f"seed{seed}_"
    keys = set(archive.files)
    truth_key, history_key = prefix + "truth", prefix + "history"
    pred_key = prefix + method
    raw: dict[str, Any] = {"seed": seed, "method": method}
    if truth_key not in keys or history_key not in keys:
        raw["trajectory_metrics"] = None
        return raw
    if pred_key in keys and method not in CLASSIFIER_ONLY:
        per_window = _per_window_metrics(archive[history_key], archive[truth_key], archive[pred_key],
                                         scales, epsilon, channel_std)
        calculated = {
            name: float(values.mean()) for name, values in per_window.items()
        }
        metric_record = _record_metrics(record)
        original_record = record.get("original_units_metrics") or {}
        # Use the exact saved scalar for seed-level summaries and deltas; retain
        # NPZ-derived window values for the paired block interval.
        if _finite_number(metric_record.get("mse")) is not None:
            calculated["mse"] = _finite_number(metric_record.get("mse"))
        if _finite_number(original_record.get("mse")) is not None:
            calculated["original_mse"] = _finite_number(original_record.get("mse"))
            raw["original_units_mse_source"] = "original_units_metrics.mse"
        elif "original_mse" in calculated:
            raw["original_units_mse_source"] = "NPZ errors weighted by training standard deviation"
        consistency = {}
        checks = [("mse", metric_record.get("mse")),
                  ("raw_sign_accuracy", metric_record.get("raw_sign_accuracy"))]
        for scale in scales:
            scale_record = (metric_record.get("scales") or {}).get(str(scale)) or {}
            checks.append((f"accuracy_scale_{scale}", scale_record.get("accuracy")))
        if original_record.get("mse") is not None and "original_mse" in per_window:
            checks.append(("original_mse", original_record.get("mse")))
        for name, reported in checks:
            value = _finite_number(reported)
            if value is None or name not in calculated:
                continue
            difference = float(calculated[name] - value)
            consistency[name] = difference
            if not np.isclose(calculated[name], value, rtol=1e-5, atol=2e-6):
                raise ValueError(f"NPZ/JSON metric mismatch at {root}, seed={seed}, method={method}, metric={name}: {calculated[name]} vs {value}")
        raw["npz_json_metric_max_abs_difference"] = max((abs(v) for v in consistency.values()), default=None)
        raw["trajectory_metrics"] = calculated
        raw["_per_window"] = per_window
        base_key = prefix + "base"
        raw["_prediction_equals_base"] = bool(base_key in keys and np.array_equal(archive[pred_key], archive[base_key]))
    else:
        raw["trajectory_metrics"] = None
        raw["_prediction_equals_base"] = method == "direction_head" and pred_key in keys and prefix + "base" in keys and np.array_equal(archive[pred_key], archive[prefix + "base"])

    classifier = _classifier_record(record)
    by_scale = {}
    for scale in scales:
        accuracy = _scale_classifier_accuracy(record, scale)
        if accuracy is not None:
            by_scale[str(scale)] = accuracy
    raw["classifier_metrics"] = {
        "accuracy": _finite_number(classifier.get("accuracy")),
        "macro_f1": _finite_number(classifier.get("macro_f1")),
        "by_scale_accuracy": by_scale,
    } if classifier else None
    for key in ("events", "regimes"):
        raw[key] = _record_metrics(record).get(key)
    raw["parameters"] = record.get("parameters") if isinstance(record.get("parameters"), dict) else None
    raw["fit_seconds"] = _finite_number(record.get("fit_seconds"))
    raw["inference_seconds"] = _finite_number(record.get("inference_seconds"))
    raw["shared_base_inference_seconds"] = _finite_number(record.get("shared_base_inference_seconds"))
    raw["selected_alpha"] = _finite_number(record.get("selected_alpha"))
    return raw


def _base_parameter_total(run: dict[str, Any]) -> int | None:
    base = (run.get("methods") or {}).get("base") or {}
    params = base.get("parameters") or {}
    try:
        return int(params["total"])
    except (KeyError, TypeError, ValueError):
        return None


def _pipeline_parameter_counts(method: str, records: list[dict[str, Any]],
                               base_counts: dict[int, int | None]) -> dict[str, Any]:
    extra = []
    total = []
    for raw in records:
        params = raw.get("parameters")
        n = params.get("total") if isinstance(params, dict) else None
        try:
            n = int(n) if n is not None else None
        except (TypeError, ValueError):
            n = None
        if n is not None:
            extra.append(n)
            if method in {"base", "difference", "last_value", "last_slope", "majority", "last_direction"}:
                total.append(n)
            elif base_counts.get(int(raw["seed"])) is not None:
                total.append(n + int(base_counts[int(raw["seed"])]))
    return {"reported_method_parameters": _stats([float(x) for x in extra]),
            "estimated_pipeline_parameters": _stats([float(x) for x in total]),
            "pipeline_parameter_note": (
                "For adapter/projection methods, estimated pipeline count adds the shared base parameters to the reported adapter/head count; for standalone forecasters it uses the reported count."
            )}


def _paired_seed_delta(candidate: dict[str, float], baseline: dict[str, float],
                       metrics: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in metrics:
        if name not in candidate or name not in baseline:
            continue
        delta = float(candidate[name] - baseline[name])
        out[name] = {"candidate": float(candidate[name]), "base": float(baseline[name]),
                     "delta": delta}
        if name in {"mse", "original_mse"}:
            out[name]["relative_delta_percent"] = (
                float(100.0 * delta / baseline[name]) if baseline[name] != 0 else None
            )
        else:
            out[name]["delta_percentage_points"] = float(100.0 * delta)
    return out


def _summarize_group(group: dict[str, Any]) -> dict[str, Any]:
    scales = group["scales"]
    metrics_to_summarize = ["raw_sign_accuracy", "mean_scales_accuracy"] + [f"accuracy_scale_{k}" for k in scales] + ["mse", "original_mse"]
    by_method: dict[str, list[dict[str, Any]]] = group["methods"]
    base_by_seed = {r["seed"]: r.get("trajectory_metrics") for r in by_method.get("base", [])}
    base_per_window = {r["seed"]: r.get("_per_window") for r in by_method.get("base", [])}
    seeds = group["seeds"]
    summary: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for method, records in by_method.items():
        trajectory_eligible = method not in CLASSIFIER_ONLY
        metric_stats: dict[str, Any] = {}
        valid_traj = [r.get("trajectory_metrics") for r in records if r.get("trajectory_metrics") is not None]
        for name in metrics_to_summarize:
            metric_stats[name] = _stats([r.get("trajectory_metrics", {}).get(name) if r.get("trajectory_metrics") else None for r in records])

        classifier_stats: dict[str, Any] | None = None
        class_records = [r.get("classifier_metrics") for r in records if r.get("classifier_metrics")]
        if class_records:
            classifier_stats = {
                "accuracy": _stats([r.get("accuracy") for r in class_records]),
                "macro_f1": _stats([r.get("macro_f1") for r in class_records]),
                "by_scale_accuracy": {
                    str(k): _stats([r.get("by_scale_accuracy", {}).get(str(k)) for r in class_records])
                    for k in scales
                },
            }

        paired = []
        if trajectory_eligible:
            for record in records:
                seed = record["seed"]
                candidate = record.get("trajectory_metrics")
                baseline = base_by_seed.get(seed)
                if candidate is None or baseline is None:
                    continue
                delta = _paired_seed_delta(candidate, baseline, metrics_to_summarize)
                raw_delta = delta.get("raw_sign_accuracy", {}).get("delta")
                mean_delta = delta.get("mean_scales_accuracy", {}).get("delta")
                mse_delta = delta.get("mse", {}).get("delta")
                same_path = bool(record.get("_prediction_equals_base"))
                mse_nonincrease = mse_delta is not None and mse_delta <= EPS_TOL
                improves_raw = raw_delta is not None and raw_delta > 0 and not same_path
                improves_mean = mean_delta is not None and mean_delta > 0 and not same_path
                scale_joint = {}
                for k in scales:
                    name = f"accuracy_scale_{k}"
                    d = delta.get(name, {}).get("delta")
                    scale_joint[str(k)] = bool(d is not None and d > 0 and mse_nonincrease and not same_path)
                detail = {
                    "dataset": group["dataset"], "backbone": group["backbone"],
                    "method": method, "seed": seed,
                    "prediction_exactly_equals_base": same_path,
                    "mse_nonincrease": bool(mse_nonincrease),
                    "raw_sign_improved": bool(improves_raw),
                    "mean_scales_improved": bool(improves_mean),
                    "joint_success_raw_sign": bool(improves_raw and mse_nonincrease),
                    "joint_success_mean_scales": bool(improves_mean and mse_nonincrease),
                    "joint_success_by_scale": scale_joint,
                    "deltas": delta,
                }
                paired.append(detail)
                paired_rows.append(detail)

        mean_deltas: dict[str, Any] = {}
        mean_joint: dict[str, bool | None] = {}
        if trajectory_eligible and valid_traj:
            bmean = {name: _stats([base_by_seed[s].get(name) for s in seeds if base_by_seed.get(s) is not None])["mean"]
                     for name in metrics_to_summarize}
            for name in metrics_to_summarize:
                m = metric_stats[name]["mean"]
                b = bmean.get(name)
                if m is not None and b is not None:
                    delta = float(m - b)
                    item: dict[str, Any] = {"candidate_mean": m, "base_mean": b, "delta": delta}
                    if name in {"mse", "original_mse"}:
                        item["relative_delta_percent"] = float(100 * delta / b) if b != 0 else None
                    else:
                        item["delta_percentage_points"] = float(100 * delta)
                    mean_deltas[name] = item
            mean_mse_delta = mean_deltas.get("mse", {}).get("delta")
            nonincrease = mean_mse_delta is not None and mean_mse_delta <= EPS_TOL
            for name, key in [("raw_sign_accuracy", "raw_sign"), ("mean_scales_accuracy", "mean_scales")]:
                d = mean_deltas.get(name, {}).get("delta")
                mean_joint[key] = bool(d is not None and d > 0 and nonincrease)
            mean_joint["by_scale"] = {}
            for k in scales:
                d = mean_deltas.get(f"accuracy_scale_{k}", {}).get("delta")
                mean_joint["by_scale"][str(k)] = bool(d is not None and d > 0 and nonincrease)

        # Seed-average per-window differences first, then resample 8-window blocks.
        confidence_intervals: dict[str, Any] = {}
        if trajectory_eligible and method != "base":
            names = ["raw_sign_accuracy", "mean_scales_accuracy"] + [f"accuracy_scale_{k}" for k in scales] + ["mse", "original_mse"]
            for name in names:
                differences = []
                for record in records:
                    seed = record["seed"]
                    c = (record.get("_per_window") or {}).get(name)
                    b = (base_per_window.get(seed) or {}).get(name)
                    if c is not None and b is not None and len(c) == len(b):
                        differences.append(np.asarray(c, dtype=float) - np.asarray(b, dtype=float))
                if differences and len({len(x) for x in differences}) == 1:
                    seed_averaged = np.mean(np.stack(differences, axis=0), axis=0)
                    confidence_intervals[name] = _moving_block_ci(seed_averaged)

        # Event scores and turning-neighborhood rates are reported for event methods.
        event_stats = {}
        event_values = [(r.get("events") or {}) for r in records]
        if any(event_values):
            for name in ("precision", "recall", "f1", "mean_absolute_time_error", "true_count", "pred_count"):
                event_stats[name] = _stats([_finite_number(e.get(name)) for e in event_values])
        turn_stats = _stats([_finite_number((r.get("regimes") or {}).get("turn_neighborhood_direction_error_rate")) for r in records])
        event_regime_per_seed = [
            {
                "seed": r["seed"],
                "mse": (r.get("trajectory_metrics") or {}).get("mse"),
                "original_mse": (r.get("trajectory_metrics") or {}).get("original_mse"),
                "raw_sign_accuracy": (r.get("trajectory_metrics") or {}).get("raw_sign_accuracy"),
                "mean_scales_accuracy": (r.get("trajectory_metrics") or {}).get("mean_scales_accuracy"),
                "event_f1": _finite_number((r.get("events") or {}).get("f1")),
                "event_precision": _finite_number((r.get("events") or {}).get("precision")),
                "event_recall": _finite_number((r.get("events") or {}).get("recall")),
                "turn_neighborhood_direction_error_rate": _finite_number((r.get("regimes") or {}).get("turn_neighborhood_direction_error_rate")),
                "turn_neighborhood_count": (r.get("regimes") or {}).get("turn_neighborhood_count"),
            }
            for r in records
        ] if method in {"event", "event_nohazard"} else None

        params = _pipeline_parameter_counts(method, records, group["base_parameter_counts"])
        fit_stats = _stats([r.get("fit_seconds") for r in records])
        inference_stats = _stats([r.get("inference_seconds") for r in records])
        shared_base_inference_stats = _stats([r.get("shared_base_inference_seconds") for r in records])
        zero_alpha = [r for r in records if r.get("selected_alpha") is not None and r.get("selected_alpha") == 0]
        summary[method] = {
            "model_type": _model_type(method),
            "classifier_only": not trajectory_eligible,
            "trajectory_comparison_eligible": trajectory_eligible,
            "trajectory_shared_with_base": bool(method == "direction_head" and records and all(r.get("_prediction_equals_base") for r in records)),
            "seed_count": len(records),
            "trajectory_metrics_mean_std_across_seeds": metric_stats if trajectory_eligible else None,
            "classifier_metrics_mean_std_across_seeds": classifier_stats,
            "mean_criterion_paired_to_base": mean_deltas if trajectory_eligible else None,
            "mean_joint_success": mean_joint if trajectory_eligible else None,
            "per_seed_paired_results": paired if trajectory_eligible and method != "base" else [],
            "per_seed_joint_success_seed_count": {
                "raw_sign": sum(bool(p["joint_success_raw_sign"]) for p in paired),
                "mean_scales": sum(bool(p["joint_success_mean_scales"]) for p in paired),
                "by_scale": {str(k): sum(bool(p["joint_success_by_scale"].get(str(k))) for p in paired) for k in scales},
                "n_seeds": len(paired),
            } if trajectory_eligible and method != "base" else None,
            "per_seed_joint_success_rate": {
                "raw_sign": (sum(bool(p["joint_success_raw_sign"]) for p in paired) / len(paired)) if paired else None,
                "mean_scales": (sum(bool(p["joint_success_mean_scales"]) for p in paired) / len(paired)) if paired else None,
                "by_scale": {
                    str(k): (sum(bool(p["joint_success_by_scale"].get(str(k))) for p in paired) / len(paired)) if paired else None
                    for k in scales
                },
            } if trajectory_eligible and method != "base" else None,
            "per_seed_direction_improvement_count": {
                "raw_sign": sum(bool(p["raw_sign_improved"]) for p in paired),
                "mean_scales": sum(bool(p["mean_scales_improved"]) for p in paired),
                "n_seeds": len(paired),
            } if trajectory_eligible and method != "base" else None,
            "paired_window_block_bootstrap_ci95": confidence_intervals,
            "event_metrics_mean_std_across_seeds": event_stats or None,
            "turn_neighborhood_direction_error_rate_mean_std": turn_stats if turn_stats["n"] else None,
            "event_and_turn_neighborhood_per_seed": event_regime_per_seed,
            "parameters": params,
            "reported_fit_seconds_mean_std": fit_stats,
            "reported_inference_seconds_mean_std": inference_stats,
            "shared_base_inference_seconds_mean_std": shared_base_inference_stats,
            "selected_alpha_zero_seed_count": len(zero_alpha),
            "selected_alpha_zero_seed_count_total": len([r for r in records if r.get("selected_alpha") is not None]),
            "all_trajectory_predictions_exactly_base": bool(trajectory_eligible and records and all(r.get("_prediction_equals_base") for r in records)),
        }

    # Direct paired comparison of the two event variants, retaining each setting.
    event_pair: dict[str, Any] | None = None
    event_records = {name: {r["seed"]: r for r in by_method.get(name, [])}
                     for name in ("event", "event_nohazard")}
    if event_records["event"] and event_records["event_nohazard"]:
        comparisons = []
        for seed in seeds:
            a, b = event_records["event"].get(seed), event_records["event_nohazard"].get(seed)
            if not a or not b:
                continue
            am, bm = a.get("trajectory_metrics") or {}, b.get("trajectory_metrics") or {}
            ae, be = a.get("events") or {}, b.get("events") or {}
            ar = a.get("regimes") or {}
            br = b.get("regimes") or {}
            comparisons.append({
                "seed": seed,
                "mse_delta_event_minus_nohazard": (am["mse"] - bm["mse"]) if "mse" in am and "mse" in bm else None,
                "mse_relative_delta_event_minus_nohazard_percent": (100.0 * (am["mse"] - bm["mse"]) / bm["mse"]) if "mse" in am and "mse" in bm and bm["mse"] != 0 else None,
                "raw_sign_accuracy_delta": (am["raw_sign_accuracy"] - bm["raw_sign_accuracy"]) if "raw_sign_accuracy" in am and "raw_sign_accuracy" in bm else None,
                "mean_scales_accuracy_delta": (am["mean_scales_accuracy"] - bm["mean_scales_accuracy"]) if "mean_scales_accuracy" in am and "mean_scales_accuracy" in bm else None,
                "original_mse_delta_event_minus_nohazard": (am["original_mse"] - bm["original_mse"]) if "original_mse" in am and "original_mse" in bm else None,
                "event_f1_delta": (_finite_number(ae.get("f1")) - _finite_number(be.get("f1"))) if _finite_number(ae.get("f1")) is not None and _finite_number(be.get("f1")) is not None else None,
                "turn_neighborhood_error_rate_delta": (_finite_number(ar.get("turn_neighborhood_direction_error_rate")) - _finite_number(br.get("turn_neighborhood_direction_error_rate"))) if _finite_number(ar.get("turn_neighborhood_direction_error_rate")) is not None and _finite_number(br.get("turn_neighborhood_direction_error_rate")) is not None else None,
            })
        event_pair = {
            "contrast": "event_minus_event_nohazard",
            "per_seed": comparisons,
            "mean_across_seeds": {
                key: _stats([row.get(key) for row in comparisons])
                for key in ("mse_delta_event_minus_nohazard", "mse_relative_delta_event_minus_nohazard_percent", "original_mse_delta_event_minus_nohazard", "raw_sign_accuracy_delta", "mean_scales_accuracy_delta", "event_f1_delta", "turn_neighborhood_error_rate_delta")
            },
            "interpretation_note": "Negative MSE and turning-neighborhood error deltas favor event; positive direction-accuracy and event-F1 deltas favor event. Event F1 is computed per seed from saved result metrics.",
        }
    return {"methods": summary, "event_vs_nohazard": event_pair,
            "paired_seed_rows": paired_rows}


def analyze(roots: list[Path], output_dir: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for root in roots:
        result = json.loads((root / "results.json").read_text(encoding="utf-8"))
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if result.get("checkpoint_reload_verified") is not True:
            raise ValueError(f"checkpoint reload is not verified; refusing incomplete/unverified result: {root}")
        if result.get("smoke") or (result.get("data") or {}).get("smoke_truncated"):
            raise ValueError(f"refusing smoke/truncated result: {root}")
        dataset = _dataset_name(config, result, root)
        backbone = str(config.get("backbone") or result.get("backbone") or "unknown")
        key = (dataset, backbone)
        group = grouped.setdefault(key, {
            "dataset": dataset, "backbone": backbone,
            "scales": [int(k) for k in result.get("scales", config.get("scales", []))],
            "sources": [], "seeds": [], "methods": {}, "base_parameter_counts": {},
            "configs": [], "checkpoint_reload_verified": True,
            "_origins": None, "_reference_truth": None, "_reference_history": None,
        })
        if group["scales"] != [int(k) for k in result.get("scales", config.get("scales", []))]:
            raise ValueError(f"scale mismatch within dataset/backbone group {key}")
        group["sources"].append(str(root.resolve()))
        group["configs"].append({"backbone": backbone, "training": config.get("training"), "data": config.get("data")})
        epsilon = np.asarray(result.get("epsilon"), dtype=float)
        normalization = result.get("normalization") or {}
        channel_std = np.asarray(normalization["std"], dtype=float) if normalization.get("std") is not None else None
        runs = result.get("runs") or []
        if not runs:
            raise ValueError(f"no per-seed runs in {root / 'results.json'}")
        with np.load(root / "predictions.npz", allow_pickle=False) as archive:
            if "origins" in archive.files:
                origins = np.asarray(archive["origins"])
                if group["_origins"] is None:
                    group["_origins"] = origins.copy()
                elif not np.array_equal(group["_origins"], origins):
                    raise ValueError(f"test origins differ within dataset/backbone group {key}")
            for run in runs:
                seed = int(run["seed"])
                if seed in group["seeds"]:
                    raise ValueError(f"duplicate dataset/backbone/seed {key + (seed,)}; pass one authoritative result per seed")
                group["seeds"].append(seed)
                group["base_parameter_counts"][seed] = _base_parameter_total(run)
                truth_key, history_key = _seed_key(seed, "truth"), _seed_key(seed, "history")
                if truth_key in archive.files and history_key in archive.files:
                    truth, history = np.asarray(archive[truth_key]), np.asarray(archive[history_key])
                    if group["_reference_truth"] is None:
                        group["_reference_truth"] = truth.copy()
                        group["_reference_history"] = history.copy()
                    elif not np.array_equal(group["_reference_truth"], truth) or not np.array_equal(group["_reference_history"], history):
                        raise ValueError(f"test truth/history differ within dataset/backbone group {key}")
                methods = run.get("methods") or {}
                for method, record in methods.items():
                    raw = _raw_seed(root, result, archive, seed, method, record,
                                    group["scales"], epsilon, channel_std)
                    group["methods"].setdefault(method, []).append(raw)

    output_groups = []
    paired_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    compact_delta_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        group["seeds"].sort()
        summarized = _summarize_group(group)
        paired_rows.extend(summarized.pop("paired_seed_rows"))
        method_summaries = summarized["methods"]
        for method, row in method_summaries.items():
            metrics = row.get("trajectory_metrics_mean_std_across_seeds") or {}
            deltas = row.get("mean_criterion_paired_to_base") or {}
            joint = row.get("per_seed_joint_success_seed_count") or {}
            param = row.get("parameters") or {}
            pipeline = param.get("estimated_pipeline_parameters") or {}
            reported = param.get("reported_method_parameters") or {}
            csv_row = {
                "dataset": group["dataset"], "backbone": group["backbone"], "method": method,
                "model_type": row["model_type"], "classifier_only": row["classifier_only"],
                "trajectory_comparison_eligible": row["trajectory_comparison_eligible"],
                "seed_count": row["seed_count"],
                "reported_method_parameters_mean": reported.get("mean"),
                "estimated_pipeline_parameters_mean": pipeline.get("mean"),
                "fit_seconds_mean": (row.get("reported_fit_seconds_mean_std") or {}).get("mean"),
                "raw_sign_accuracy_mean": (metrics.get("raw_sign_accuracy") or {}).get("mean"),
                "raw_sign_accuracy_std": (metrics.get("raw_sign_accuracy") or {}).get("std"),
                "mean_scales_accuracy_mean": (metrics.get("mean_scales_accuracy") or {}).get("mean"),
                "mean_scales_accuracy_std": (metrics.get("mean_scales_accuracy") or {}).get("std"),
                "raw_sign_delta_pp_mean": (deltas.get("raw_sign_accuracy") or {}).get("delta_percentage_points"),
                "mean_scales_delta_pp_mean": (deltas.get("mean_scales_accuracy") or {}).get("delta_percentage_points"),
                "mse_mean": (metrics.get("mse") or {}).get("mean"),
                "mse_std": (metrics.get("mse") or {}).get("std"),
                "mse_relative_delta_percent_mean": (deltas.get("mse") or {}).get("relative_delta_percent"),
                "original_mse_mean": (metrics.get("original_mse") or {}).get("mean"),
                "original_mse_std": (metrics.get("original_mse") or {}).get("std"),
                "joint_success_raw_sign_seeds": joint.get("raw_sign"),
                "joint_success_mean_scales_seeds": joint.get("mean_scales"),
                "joint_success_seed_denominator": joint.get("n_seeds"),
                "joint_success_raw_sign_rate": ((row.get("per_seed_joint_success_rate") or {}).get("raw_sign")),
                "joint_success_mean_scales_rate": ((row.get("per_seed_joint_success_rate") or {}).get("mean_scales")),
                "mean_joint_raw_sign": ((row.get("mean_joint_success") or {}).get("raw_sign")),
                "mean_joint_mean_scales": ((row.get("mean_joint_success") or {}).get("mean_scales")),
                "classifier_accuracy_mean": ((row.get("classifier_metrics_mean_std_across_seeds") or {}).get("accuracy") or {}).get("mean"),
                "classifier_macro_f1_mean": ((row.get("classifier_metrics_mean_std_across_seeds") or {}).get("macro_f1") or {}).get("mean"),
                "event_f1_mean": ((row.get("event_metrics_mean_std_across_seeds") or {}).get("f1") or {}).get("mean"),
                "turn_neighborhood_error_rate_mean": ((row.get("turn_neighborhood_direction_error_rate_mean_std") or {}).get("mean")),
            }
            for scale in group["scales"]:
                stat = metrics.get(f"accuracy_scale_{scale}") or {}
                delta = deltas.get(f"accuracy_scale_{scale}") or {}
                csv_row[f"accuracy_scale_{scale}_mean"] = stat.get("mean")
                csv_row[f"accuracy_scale_{scale}_std"] = stat.get("std")
                csv_row[f"accuracy_scale_{scale}_delta_pp_mean"] = delta.get("delta_percentage_points")
                csv_row[f"joint_success_scale_{scale}_seeds"] = (joint.get("by_scale") or {}).get(str(scale))
            summary_rows.append(csv_row)
            if method in {"difference", "projection", "projection_single", "residual", "residual_direction", "event", "event_nohazard"}:
                compact_delta_rows.append({
                    "dataset": group["dataset"], "backbone": group["backbone"], "method": method,
                    "seed_count": row["seed_count"],
                    "raw_sign_delta_percentage_points": (deltas.get("raw_sign_accuracy") or {}).get("delta_percentage_points"),
                    "raw_sign_joint_success_seeds": joint.get("raw_sign"),
                    "mean_scales_delta_percentage_points": (deltas.get("mean_scales_accuracy") or {}).get("delta_percentage_points"),
                    "mean_scales_joint_success_seeds": joint.get("mean_scales"),
                    "standardized_mse_relative_delta_percent": (deltas.get("mse") or {}).get("relative_delta_percent"),
                    "mean_criterion_joint_raw_sign": (row.get("mean_joint_success") or {}).get("raw_sign"),
                    "mean_criterion_joint_mean_scales": (row.get("mean_joint_success") or {}).get("mean_scales"),
                })
        output_groups.append({
            "dataset": group["dataset"], "backbone": group["backbone"], "sources": group["sources"],
            "seeds": group["seeds"], "scales": group["scales"],
            "checkpoint_reload_verified": group["checkpoint_reload_verified"], "methods": method_summaries,
            "event_vs_nohazard": summarized.get("event_vs_nohazard"),
            "configuration_metadata": group["configs"],
        })

    report = {
        "schema_version": 1,
        "analysis": "paired method-level direction accuracy and MSE tradeoffs",
        "groups": output_groups,
        "criteria": {
            "mean_scale_accuracy": "Arithmetic mean of per-window 3-class accuracy at the predeclared scales; method-specific only.",
            "raw_sign_accuracy": "Unthresholded adjacent-step sign agreement; zero is its own sign class.",
            "strict_joint_success": "Direction delta must be strictly positive and standardized MSE delta <= 1e-8; +1% is never treated as unchanged. Exact no-change paths cannot count as direction improvement.",
            "original_units_mse": "Reported as a supplementary metric; variables have different units, so larger-scale channels can dominate it.",
            "classifier_only": "direction_head, majority, and last_direction report classifier metrics separately; classifier gains are excluded from trajectory joint-success counts.",
            "parameter_counts": "Reported method parameters may be adapter/head-only. Estimated pipeline count adds the shared base for projection/adapter pipelines; use these counts to qualify fairness.",
        },
        "completion_checks": [
            "every included cell has checkpoint_reload_verified=true",
            "smoke or truncated data are rejected",
        ],
        "bootstrap": {
            "method": "moving blocks of consecutive test windows; average paired per-window differences across fitted seeds first, then resample windows",
            "block_size_windows": BLOCK_SIZE, "replicates": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
            "interpretation": "Conditional interval for the fixed dataset/test segment and the fitted seeds, not independent dataset repetitions; seeds are not resampled as independent examples and groups are not pooled across datasets/backbones.",
        },
        "mse_tolerance_absolute": EPS_TOL,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "method_tradeoffs.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(output_dir / "method_summary.csv", summary_rows)
    _write_csv(output_dir / "tradeoff_delta_table.csv", compact_delta_rows)
    _write_paired_csv(output_dir / "paired_seed_deltas.csv", paired_rows)
    (output_dir / "evidence.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_paired_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for row in rows:
        item = {k: row.get(k) for k in (
            "dataset", "backbone", "method", "seed", "prediction_exactly_equals_base",
            "mse_nonincrease", "raw_sign_improved", "mean_scales_improved",
            "joint_success_raw_sign", "joint_success_mean_scales")}
        for name, rec in row.get("deltas", {}).items():
            item[name + "_delta"] = rec.get("delta")
            item[name + "_change"] = rec.get("relative_delta_percent") if "relative_delta_percent" in rec else rec.get("delta_percentage_points")
            item[name + "_change_unit"] = "relative_percent" if "relative_delta_percent" in rec else "percentage_points"
        for scale, ok in row.get("joint_success_by_scale", {}).items():
            item[f"joint_success_scale_{scale}"] = ok
        flat.append(item)
    _write_csv(path, flat)


def _fmt_pm(stat: dict[str, Any] | None, factor: float = 1.0, digits: int = 3) -> str:
    if not stat or stat.get("mean") is None:
        return "—"
    mean = float(stat["mean"]) * factor
    std = float(stat["std"] or 0.0) * factor
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Method tradeoff evidence",
        "",
        "Each row is one fixed dataset × backbone × method. Values are mean ± sample SD across the fitted seeds; no methods or datasets are pooled.",
        "",
        "Joint success requires a strictly positive direction delta and standardized MSE delta ≤ 1e-8. Raw sign and 3-class mean-scales criteria are separate. Classifier-only methods are not trajectory gains.",
        "Original-unit MSE is supplementary because the channels have different units. Parameter counts distinguish reported adapter/head counts from an estimated full pipeline count.",
        "",
    ]
    for group in report["groups"]:
        lines += [f"## {group['dataset']} × {group['backbone']}", "",
                  f"Seeds: {group['seeds']}; scales: {group['scales']}.", "",
                  "| Method | Type | raw sign | 3-class mean | k=1 | k=2 | k=4 | k=8 | norm MSE | original MSE | joint raw | joint 3-class | params pipeline |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for method, row in group["methods"].items():
            metric = row.get("trajectory_metrics_mean_std_across_seeds") or {}
            joint = row.get("per_seed_joint_success_seed_count") or {}
            params = ((row.get("parameters") or {}).get("estimated_pipeline_parameters") or {}).get("mean")
            param_text = f"{params:.0f}" if params is not None else "—"
            if row.get("classifier_only"):
                cls = row.get("classifier_metrics_mean_std_across_seeds") or {}
                lines.append(f"| {method} | classifier only | — | — | — | — | — | — | — | — | — | — | {param_text} (classifier acc {_fmt_pm(cls.get('accuracy'), 100, 2)}%) |")
                continue
            acc = lambda name: _fmt_pm(metric.get(name), 100, 2) + "%"
            lines.append("| " + " | ".join([
                method, row.get("model_type", ""), acc("raw_sign_accuracy"), acc("mean_scales_accuracy"),
                *[acc(f"accuracy_scale_{k}") for k in group["scales"]],
                _fmt_pm(metric.get("mse"), 1, 4), _fmt_pm(metric.get("original_mse"), 1, 4),
                f"{joint.get('raw_sign', '—')}/{joint.get('n_seeds', '—')}",
                f"{joint.get('mean_scales', '—')}/{joint.get('n_seeds', '—')}", param_text,
            ]) + " |")
        lines.append("")
        lines += [
            "Compact paired changes vs base (direction changes in percentage points; standardized MSE change in relative percent):",
            "",
            "| Method | raw-sign Δ pp | raw-sign joint seeds | 3-class mean Δ pp | 3-class joint seeds | standardized MSE Δ % | mean raw joint | mean 3-class joint |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
        compact_methods = ("difference", "projection", "projection_single", "residual", "residual_direction", "event", "event_nohazard")
        for method in compact_methods:
            row = group["methods"].get(method)
            if not row:
                continue
            d = row.get("mean_criterion_paired_to_base") or {}
            joint = row.get("per_seed_joint_success_seed_count") or {}
            mean_joint = row.get("mean_joint_success") or {}
            raw_d = (d.get("raw_sign_accuracy") or {}).get("delta_percentage_points")
            mean_d = (d.get("mean_scales_accuracy") or {}).get("delta_percentage_points")
            mse_d = (d.get("mse") or {}).get("relative_delta_percent")
            den = joint.get("n_seeds")
            def pp(value: Any) -> str:
                return "—" if value is None else f"{float(value):+.3f}"
            lines.append("| " + " | ".join([
                method, pp(raw_d), f"{joint.get('raw_sign', '—')}/{den if den is not None else '—'}",
                pp(mean_d), f"{joint.get('mean_scales', '—')}/{den if den is not None else '—'}",
                "—" if mse_d is None else f"{float(mse_d):+.3f}",
                str(mean_joint.get("raw_sign", "—")), str(mean_joint.get("mean_scales", "—")),
            ]) + " |")
        lines += ["", "Paired mean changes with 95% window-block intervals vs base:"]
        for method, row in group["methods"].items():
            if row.get("classifier_only") or method == "base":
                continue
            d = row.get("mean_criterion_paired_to_base") or {}
            raw = d.get("raw_sign_accuracy", {})
            mean = d.get("mean_scales_accuracy", {})
            mse = d.get("mse", {})
            cis = row.get("paired_window_block_bootstrap_ci95") or {}
            def ci(name: str, factor: float = 1.0) -> str:
                x = cis.get(name, {}).get("ci95")
                return "" if not x or x[0] is None else f"; 95% block CI [{x[0]*factor:.3f}, {x[1]*factor:.3f}]"
            lines.append(
                f"- `{method}`: raw sign {raw.get('delta_percentage_points', float('nan')):+.3f} pp{ci('raw_sign_accuracy', 100)}; "
                f"mean-scales {mean.get('delta_percentage_points', float('nan')):+.3f} pp{ci('mean_scales_accuracy', 100)}; "
                f"standardized MSE {mse.get('relative_delta_percent', float('nan')):+.3f}% "
                f"(strict joint seeds raw {((row.get('per_seed_joint_success_seed_count') or {}).get('raw_sign', '—'))}/"
                f"{((row.get('per_seed_joint_success_seed_count') or {}).get('n_seeds', '—'))}, 3-class "
                f"{((row.get('per_seed_joint_success_seed_count') or {}).get('mean_scales', '—'))}/"
                f"{((row.get('per_seed_joint_success_seed_count') or {}).get('n_seeds', '—'))})."
            )
        event_pair = group.get("event_vs_nohazard")
        if event_pair:
            means = event_pair.get("mean_across_seeds") or {}
            lines += ["", "Event vs event_nohazard (event minus nohazard; per-seed paired):"]
            for metric in ("mse_delta_event_minus_nohazard", "original_mse_delta_event_minus_nohazard", "raw_sign_accuracy_delta", "mean_scales_accuracy_delta", "event_f1_delta", "turn_neighborhood_error_rate_delta"):
                stat = means.get(metric)
                if stat and stat.get("mean") is not None:
                    lines.append(f"- {metric}: {_fmt_pm(stat, 1, 5)}")
        lines.append("")
        lines.append("Evidence source artifacts:")
        for source in group["sources"]:
            lines.append(f"- `{source}`")
        lines.append("")
    lines += [
        "## Interpretation limits",
        "",
        f"Confidence intervals use {BLOCK_SIZE}-window moving blocks and {BOOTSTRAP_REPEATS} replicates after averaging paired per-window differences across fitted seeds. They are conditional on each fixed test segment and the trained seeds; they do not treat seeds as independent test datasets.",
        "Original-unit MSE can be dominated by high-scale channels. Reported adapter/head parameter counts omit the shared base unless the pipeline estimate is shown.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="one or more artifact roots, results.json files, or parent folders")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory for method_tradeoffs.json, method_summary.csv, paired_seed_deltas.csv, and evidence.md")
    args = parser.parse_args()
    roots = _find_artifact_roots(args.inputs)
    report = analyze(roots, args.output_dir.resolve())
    print(json.dumps({"groups": [{"dataset": g["dataset"], "backbone": g["backbone"],
                                   "methods": list(g["methods"])} for g in report["groups"]],
                      "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
