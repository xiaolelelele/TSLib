"""Recheck four pre-registered fixed prediction means without weight search.

The evaluator consumes the three frozen fusion roots for exploratory groups
and completed training cells for the five-seed Electricity/Traffic validation.
It never changes the frozen fusion evaluator or trains a model.
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
sys.path.insert(0, str(WORKSPACE / "DTL" / "research"))

import evaluate_trend_fusions as frozen

ATOMS = ("learned_multi", "learned_single", "prior", "flat")
STANDALONE_ORDER = (
    "base", "difference", "residual", "residual_direction",
    "learned_single", "learned_multi", "prior", "flat",
)
COMBINATIONS: dict[str, tuple[str, ...]] = {
    "learned_threeway": ("learned_selected", "flat", "difference"),
    "nohead_threeway": ("base", "flat", "difference"),
    "base_difference_mean": ("base", "difference"),
    "flat_difference_mean": ("flat", "difference"),
}
BOOTSTRAP_METRICS = (
    "raw_sign_accuracy", "standardized_mse", "original_units_mse",
    "mean_3class_accuracy", "accuracy_scale_1", "accuracy_scale_2",
    "accuracy_scale_4", "accuracy_scale_8",
)
BOOTSTRAP_BLOCK = 8
BOOTSTRAP_REPEATS = 2000
STRICT_MSE_TOLERANCE = 1e-8


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
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    if not token:
        raise ValueError(f"invalid safe key from {value!r}")
    return token


def _alpha_token(alpha: float) -> str:
    return f"{float(alpha):.8g}".replace("-", "m").replace(".", "p")


def _key(seed: int, split: str, candidate: str, field: str = "path") -> str:
    return f"seed{seed}__{split}__{candidate}__{field}"


def _to_np(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_tensor(value: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _save_path(npz_data: dict[str, np.ndarray], seed: int, split: str,
               candidate: str, path: torch.Tensor | np.ndarray) -> str:
    key = _key(seed, split, candidate)
    array = _to_np(path).astype(np.float32, copy=False)
    if key in npz_data and not np.array_equal(npz_data[key], array):
        raise AssertionError(f"attempt to replace an existing prediction path: {key}")
    npz_data[key] = array
    return key


def _save_split(npz_data: dict[str, np.ndarray], seed: int, split: str,
                history: torch.Tensor | np.ndarray, truth: torch.Tensor | np.ndarray,
                origins: np.ndarray) -> None:
    for field, value in (("history", history), ("truth", truth), ("origins", origins)):
        npz_data[_key(seed, split, "split", field)] = _to_np(value).astype(
            np.int64 if field == "origins" else np.float32, copy=False
        )


def _same_path(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> bool:
    return bool(np.array_equal(_to_np(a), _to_np(b)))


def _fixed_mean(paths: list[torch.Tensor], divisor: int, arithmetic: str) -> torch.Tensor:
    if not paths:
        raise ValueError("cannot average no paths")
    if all(_same_path(paths[0], path) for path in paths[1:]):
        return paths[0]
    total = paths[0]
    for path in paths[1:]:
        total = total + path
    if divisor == 3 and arithmetic == "sum_reciprocal_multiply":
        return total * float(np.float32(1.0 / 3.0))
    return total / float(divisor)


def _unique_path_count(paths: list[torch.Tensor]) -> int:
    unique: list[np.ndarray] = []
    for path in paths:
        array = _to_np(path)
        if not any(np.array_equal(array, seen) for seen in unique):
            unique.append(array)
    return len(unique)


def _eligible(metrics: dict[str, Any], base_metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["raw_sign_accuracy"] > base_metrics["raw_sign_accuracy"]
        and metrics["standardized_mse"] <= base_metrics["standardized_mse"] + STRICT_MSE_TOLERANCE
    )


def _score(history: torch.Tensor, truth: torch.Tensor, prediction: torch.Tensor,
           true_labels: np.ndarray, true_raw_sign: np.ndarray, scales: list[int],
           epsilon: torch.Tensor, std: np.ndarray
           ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics, windows, _ = frozen._score_path(
        history, truth, prediction, true_labels, true_raw_sign, scales, epsilon, std
    )
    metrics["mse"] = metrics["standardized_mse"]
    return metrics, windows


def _metric_delta(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for key in candidate:
        output[key + "_delta"] = float(candidate[key] - reference[key])
        if key in ("standardized_mse", "original_units_mse"):
            output[key + "_percent_change"] = (
                float(100.0 * (candidate[key] / reference[key] - 1.0))
                if reference[key] != 0 else None
            )
    return output


def _bootstrap_indices(n: int) -> np.ndarray:
    block = min(BOOTSTRAP_BLOCK, n)
    if block < 1:
        return np.empty((BOOTSTRAP_REPEATS, 0), dtype=np.int64)
    needed = int(math.ceil(n / block))
    rng = np.random.default_rng(0)
    starts = rng.integers(0, n - block + 1, size=(BOOTSTRAP_REPEATS, needed))
    return (starts[:, :, None] + np.arange(block, dtype=np.int64)[None, None, :]).reshape(
        BOOTSTRAP_REPEATS, -1
    )[:, :n]


def _bootstrap_ci(values: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    boot = flat[indices].mean(axis=1)
    return {
        "mean": float(flat.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "windows": int(len(flat)),
    }


def _resolve_data_path(cell_dir: Path, value: str) -> Path:
    return frozen._resolve_data_path(cell_dir, value)


def _cell_provenance(cell_dir: Path, config: dict[str, Any], source_results: dict[str, Any],
                     evaluator: dict[str, Any] | None, expected: dict[str, Any] | None = None
                     ) -> dict[str, Any]:
    if config.get("smoke", False) or source_results.get("smoke", False):
        raise ValueError(f"smoke-trained source is not accepted: {cell_dir}")
    if source_results.get("checkpoint_reload_verified") is not True:
        raise ValueError(f"checkpoint_reload_verified is not true: {cell_dir}")
    if source_results.get("data", {}).get("smoke_truncated", False):
        raise ValueError(f"source data were smoke-truncated: {cell_dir}")
    data_path = _resolve_data_path(cell_dir, str(config["data"]["csv_path"]))
    data_hash = _sha256(data_path)
    file_hashes = {
        "config_sha256": _sha256(cell_dir / "config.json"),
        "results_sha256": _sha256(cell_dir / "results.json"),
        "checkpoint_sha256": _sha256(cell_dir / "checkpoint.pt"),
        "source_prediction_npz_sha256": _sha256(cell_dir / "predictions.npz"),
        "data_sha256": data_hash,
    }
    for name, expected_hash in (expected or {}).items():
        if name.endswith("_sha256") and expected_hash and file_hashes.get(name) != expected_hash:
            raise AssertionError(f"source {name} changed for {cell_dir}")
    evaluator_info = evaluator or {}
    return {
        "source_cell_dir": str(cell_dir),
        "source_config_path": str(cell_dir / "config.json"),
        "source_results_path": str(cell_dir / "results.json"),
        "source_checkpoint_path": str(cell_dir / "checkpoint.pt"),
        "source_prediction_npz_path": str(cell_dir / "predictions.npz"),
        "data_file": str(data_path),
        "data_sha256": data_hash,
        **file_hashes,
        "reload_verified": True,
        "evaluator_root": evaluator_info.get("root"),
        "evaluator_snapshot": evaluator_info.get("snapshot"),
        "evaluator_sha256": evaluator_info.get("sha256"),
        "evaluator_manifest_sha256": evaluator_info.get("manifest_sha256"),
    }


def _load_old_root(root: Path, wanted_groups: set[str] | None = None
                   ) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    root = root.expanduser().resolve()
    manifest_path = root / "output_manifest.json"
    results_path = root / "results.json"
    npz_path = root / "predictions.npz"
    for path in (manifest_path, results_path, npz_path):
        if not path.is_file():
            raise FileNotFoundError(f"old evaluator root is missing {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError(f"old evaluator root is incomplete: {root}")
    if manifest.get("cohort") not in ("exploratory", "additional_validation"):
        raise ValueError(f"unexpected old evaluator cohort at {root}")
    requested = set(manifest.get("requested_groups", []))
    completed = set(manifest.get("completed_groups", []))
    if requested != completed:
        raise ValueError(f"old evaluator root group manifest is incomplete: {root}")
    snapshot = Path(manifest.get("script_snapshot", ""))
    if not snapshot.is_file() or _sha256(snapshot) != manifest.get("script_sha256"):
        raise AssertionError(f"old evaluator snapshot hash mismatch: {root}")
    result = json.loads(results_path.read_text(encoding="utf-8"))
    if set(result.get("groups", {})) != requested:
        raise AssertionError(f"old evaluator results/manifest groups mismatch: {root}")
    selected_groups = requested if wanted_groups is None else wanted_groups
    if not selected_groups.issubset(requested):
        raise KeyError(f"requested groups {sorted(selected_groups - requested)} are absent from {root}")
    with np.load(npz_path, allow_pickle=False) as archive:
        selected_keys = [key for key in archive.files
                         if any(key.startswith(group_key + "__") for group_key in selected_groups)]
        arrays = {key: archive[key] for key in selected_keys}
    return manifest, result, arrays


def _old_cell_provenance(root: Path, root_manifest: dict[str, Any], root_results: dict[str, Any],
                         group: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cell_dir = Path(group["seeds"]["0"]["reconstruction"]["cell_dir"]).resolve()
    config = json.loads((cell_dir / "config.json").read_text(encoding="utf-8"))
    source_results = json.loads((cell_dir / "results.json").read_text(encoding="utf-8"))
    evaluator = {
        "root": str(root),
        "snapshot": root_manifest.get("script_snapshot"),
        "sha256": root_manifest.get("script_sha256"),
        "manifest_sha256": _sha256(root / "output_manifest.json"),
    }
    expected = {
        "config_sha256": group["source"].get("config_sha256"),
        "checkpoint_sha256": group["source"].get("checkpoint_sha256"),
        "source_prediction_npz_sha256": group["source"].get("source_prediction_npz_sha256"),
        "data_sha256": group["source"].get("data_sha256"),
    }
    provenance = _cell_provenance(cell_dir, config, source_results, evaluator, expected)
    provenance["source_evaluator_prediction_npz_sha256"] = _sha256(root / "predictions.npz")
    provenance["source_evaluator_results_sha256"] = _sha256(root / "results.json")
    provenance["source_evaluator_audit_path"] = str(root / "independent_audit.json")
    if (root / "independent_audit.json").is_file():
        provenance["source_evaluator_audit_sha256"] = _sha256(root / "independent_audit.json")
    if group.get("source", {}).get("script_sha256") != root_manifest.get("script_sha256"):
        raise AssertionError(f"source evaluator script identity mismatch for {group['group_key']}")
    return config, source_results, provenance


def _source_prediction(npz: dict[str, np.ndarray], key: str, device: torch.device) -> torch.Tensor:
    if key not in npz:
        raise KeyError(f"source evaluator prediction key is missing: {key}")
    return _as_tensor(npz[key], device)


def _old_seed_inputs(group_key: str, run: dict[str, Any], source_npz: dict[str, np.ndarray],
                     seed: int, device: torch.device, source_device: str, arithmetic: str
                     ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data = run["reconstruction"]["data"]
    x: dict[str, dict[str, torch.Tensor]] = {}
    origins: dict[str, np.ndarray] = {}
    for split in ("validation", "test"):
        prefix = f"{group_key}__seed{seed}__{split}__split__"
        for field in ("history", "truth"):
            source_key = prefix + field
            if source_key not in source_npz:
                raise KeyError(f"old evaluator misses split key {source_key}")
        x[split] = {
            "history": _as_tensor(source_npz[prefix + "history"], device),
            "truth": _as_tensor(source_npz[prefix + "truth"], device),
        }
        origins[split] = np.asarray(source_npz[prefix + "origins"], dtype=np.int64)

    run_validation = run["validation_selection"]
    method_paths: dict[str, dict[str, torch.Tensor]] = {"validation": {}, "test": {}}
    for method in STANDALONE_ORDER:
        val_key = next(
            row["prediction_key"] for row in run["validation_grids"]["standalone_candidates"]
            if row["candidate_id"] == method
        )
        test_key = run["test_metrics"][method]["prediction_key"]
        method_paths["validation"][method] = _source_prediction(source_npz, val_key, device)
        method_paths["test"][method] = _source_prediction(source_npz, test_key, device)
    for atom in ATOMS:
        val_key = run_validation["atom_selection"][atom]["prediction_key"]
        test_key = run["test_metrics"][atom]["prediction_key"]
        method_paths["validation"][atom] = _source_prediction(source_npz, val_key, device)
        method_paths["test"][atom] = _source_prediction(source_npz, test_key, device)
    for split in ("validation", "test"):
        for method in ("learned_selected", "standalone_selected"):
            method_paths[split][method] = _source_prediction(
                source_npz, run["test_metrics"][method]["prediction_key"] if split == "test"
                else run_validation[method]["prediction_key"], device,
            )

    atom_alpha_paths: dict[str, list[dict[str, Any]]] = {}
    for atom, rows in run["validation_grids"]["atom_alpha_grids"].items():
        atom_alpha_paths[atom] = []
        for row in rows:
            atom_alpha_paths[atom].append({
                "candidate_id": row["candidate_id"],
                "alpha": float(row["alpha"]),
                "path": _source_prediction(source_npz, row["prediction_key"], device),
                "source_metrics": row["metrics"],
            })

    probability_sources: dict[str, np.ndarray] = {}
    for atom in ATOMS:
        row = run["validation_grids"]["atom_alpha_grids"][atom][0]
        probability_key = row.get("probability_key")
        if probability_key and probability_key in source_npz:
            probability_sources[f"validation__{atom}"] = source_npz[probability_key]
    for atom in ATOMS:
        candidate = run["test_metrics"][atom]["candidate_id"]
        probability_key = frozen._npz_key(group_key, seed, "test", candidate, "probabilities")
        if probability_key in source_npz:
            probability_sources[f"test__{atom}"] = source_npz[probability_key]

    if run["reconstruction"].get("test_model_paths_match_saved_npz_rtol_1e_5_atol_1e_6") is not True:
        raise AssertionError(f"source test paths were not reload-verified for {group_key}/seed{seed}")
    metadata = {
        "data": x,
        "origins": origins,
        "method_paths": method_paths,
        "atom_alpha_paths": atom_alpha_paths,
        "probability_sources": probability_sources,
        "scales": list(data["scales"]),
        "epsilon": torch.as_tensor(data["epsilon"], dtype=torch.float32, device=device),
        "mean": np.asarray(data["normalization"]["mean"], dtype=np.float32),
        "std": np.asarray(data["normalization"]["std"], dtype=np.float32),
        "prior_counts": source_npz.get(f"{group_key}__train_prior_class_counts"),
        "prior_probabilities": source_npz.get(f"{group_key}__prior_probabilities"),
        "flat_probabilities": source_npz.get(f"{group_key}__flat_probabilities"),
        "source_validation_selection": run_validation,
        "source_fixed_threeway_validation": run_validation["fixed_threeway_mean"],
        "source_fixed_threeway_test": run["test_metrics"]["threeway_mean"],
        "source_npz": source_npz,
        "source_device": source_device,
        "arithmetic": arithmetic,
        "seed_run": run,
    }
    return metadata, {k: np.asarray(v) for k, v in probability_sources.items()}


def _window_metrics_delta(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]
                          ) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(candidate[name], dtype=np.float64) - np.asarray(reference[name], dtype=np.float64)
        for name in BOOTSTRAP_METRICS
        if name in candidate and name in reference
    }


def _metric_record_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    output = _metric_delta(
        {name: float(candidate[name]) for name in (
            "raw_sign_accuracy", "standardized_mse", "original_units_mse",
            "mean_3class_accuracy", "mean_macro_f1",
        )},
        {name: float(reference[name]) for name in (
            "raw_sign_accuracy", "standardized_mse", "original_units_mse",
            "mean_3class_accuracy", "mean_macro_f1",
        )},
    )
    for field in ("accuracy_by_scale", "macro_f1_by_scale"):
        output[field + "_delta"] = {
            scale: float(candidate[field][scale] - reference[field][scale])
            for scale in candidate[field]
        }
    return output


def _emit_comparison(comparison_id: str, seed: int, candidate_id: str, reference_id: str,
                     candidate_metrics: dict[str, Any], reference_metrics: dict[str, Any],
                     candidate_windows: dict[str, np.ndarray], reference_windows: dict[str, np.ndarray],
                     npz_data: dict[str, np.ndarray]) -> dict[str, Any]:
    window_delta = _window_metrics_delta(candidate_windows, reference_windows)
    keys = {}
    for metric, delta in window_delta.items():
        key = f"seed{seed}__test__comparison__{comparison_id}__{metric}_window_delta"
        npz_data[key] = delta.astype(np.float64)
        keys[metric] = key
    return {
        "candidate_id": candidate_id,
        "reference_id": reference_id,
        "metrics_delta": _metric_record_delta(candidate_metrics, reference_metrics),
        "seed_window_delta_npz_keys": keys,
    }


def _evaluate_seed(group_key: str, seed: int, prepared: dict[str, Any],
                   npz_data: dict[str, np.ndarray], device: torch.device
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
    scales = [int(value) for value in prepared["scales"]]
    epsilon = _as_tensor(prepared["epsilon"], device)
    std = np.asarray(prepared["std"], dtype=np.float32)
    arithmetic = prepared["arithmetic"]
    data = prepared["data"]
    val_x, val_y = data["validation"]["history"], data["validation"]["truth"]
    test_x, test_y = data["test"]["history"], data["test"]["truth"]
    _save_split(npz_data, seed, "validation", val_x, val_y, prepared["origins"]["validation"])
    npz_data[f"seed{seed}__metadata__scales"] = np.asarray(scales, dtype=np.int64)
    npz_data[f"seed{seed}__metadata__epsilon"] = _to_np(epsilon).astype(np.float32, copy=False)
    npz_data[f"seed{seed}__metadata__normalization_mean"] = np.asarray(prepared["mean"], dtype=np.float32)
    npz_data[f"seed{seed}__metadata__normalization_std"] = std
    if prepared.get("prior_counts") is not None:
        npz_data[f"seed{seed}__metadata__prior_class_counts"] = np.asarray(prepared["prior_counts"], dtype=np.int64)
    if prepared.get("prior_train_labels") is not None:
        npz_data[f"seed{seed}__metadata__prior_train_labels"] = np.asarray(prepared["prior_train_labels"], dtype=np.int64)
    if prepared.get("prior_probabilities") is not None:
        npz_data[f"seed{seed}__metadata__prior_probabilities"] = np.asarray(prepared["prior_probabilities"], dtype=np.float32)
    if prepared.get("flat_probabilities") is not None:
        npz_data[f"seed{seed}__metadata__flat_probabilities"] = np.asarray(prepared["flat_probabilities"], dtype=np.float32)
    for source_name, value in prepared.get("probability_sources", {}).items():
        split, atom = source_name.split("__", maxsplit=1)
        npz_data[f"seed{seed}__{split}__probabilities__{atom}__probabilities"] = np.asarray(value, dtype=np.float32)

    val_true_labels, val_true_sign = frozen._metric_labels(val_x, val_y, scales, epsilon)
    val_paths = dict(prepared["method_paths"]["validation"])
    test_paths = dict(prepared["method_paths"]["test"])
    alpha_entries = prepared["atom_alpha_paths"]
    val_windows: dict[str, dict[str, np.ndarray]] = {}
    test_windows: dict[str, dict[str, np.ndarray]] = {}
    val_metrics: dict[str, dict[str, Any]] = {}
    test_metrics: dict[str, dict[str, Any]] = {}
    for method, path in val_paths.items():
        _save_path(npz_data, seed, "validation", method, path)

    base_path = val_paths["base"]
    base_val_metrics, base_val_windows = _score(
        val_x, val_y, base_path, val_true_labels, val_true_sign, scales, epsilon, std
    )
    val_metrics["base"], val_windows["base"] = base_val_metrics, base_val_windows
    validation_grid: dict[str, list[dict[str, Any]]] = {}
    selected_atoms: dict[str, torch.Tensor] = {}
    selected_alphas: dict[str, float] = {}
    atom_selection: dict[str, Any] = {}

    # Re-score every stored alpha path on validation and apply the frozen selector.
    for atom in ATOMS:
        rows = []
        entry_by_id = {entry["candidate_id"]: entry for entry in alpha_entries[atom]}
        for entry in alpha_entries[atom]:
            candidate_id = str(entry["candidate_id"])
            prediction = entry["path"]
            metric, window = _score(
                val_x, val_y, prediction, val_true_labels, val_true_sign, scales, epsilon, std
            )
            path_key = _save_path(npz_data, seed, "validation", candidate_id, prediction)
            rows.append({
                "candidate_id": candidate_id,
                "alpha": float(entry["alpha"]),
                "path_key": path_key,
                "metrics": metric,
                "eligible": _eligible(metric, base_val_metrics),
                "is_no_op_vs_base": _same_path(prediction, base_path),
            })
            val_metrics[candidate_id], val_windows[candidate_id] = metric, window
        chosen, has_eligible = frozen._chosen_alpha(
            [{"candidate_id": row["candidate_id"], "alpha": row["alpha"], "metrics": row["metrics"]}
             for row in rows], base_val_metrics
        )
        selected = entry_by_id[chosen["candidate_id"]]
        selected_atoms[atom] = selected["path"]
        selected_alphas[atom] = float(selected["alpha"])
        alias_key = _save_path(npz_data, seed, "validation", atom, selected["path"])
        atom_selection[atom] = {
            "selected_candidate_id": chosen["candidate_id"],
            "selected_alpha": selected_alphas[atom],
            "path_key": alias_key,
            "source_grid_path_key": next(row["path_key"] for row in rows
                                          if row["candidate_id"] == chosen["candidate_id"]),
            "eligible": bool(has_eligible),
            "fallback_to_alpha_zero": not bool(has_eligible),
            "is_no_op_vs_base": _same_path(selected["path"], base_path),
            "selection_gate": "raw_sign_accuracy > base and standardized_mse <= base + 1e-8",
            "tie_break": "raw-sign accuracy descending, standardized MSE ascending, alpha ascending",
        }
        validation_grid[atom] = rows
        val_paths[atom] = selected["path"]
        val_metrics[atom], val_windows[atom] = val_metrics[chosen["candidate_id"]], val_windows[chosen["candidate_id"]]

    learned_candidates = []
    for tie, atom in enumerate(("learned_single", "learned_multi")):
        learned_candidates.append({
            "candidate_id": atom,
            "source_candidate_id": atom_selection[atom]["selected_candidate_id"],
            "alpha": selected_alphas[atom],
            "metrics": val_metrics[atom],
            "tie_value": tie,
        })
    learned_candidates.sort(key=lambda row: (
        -row["metrics"]["raw_sign_accuracy"], row["metrics"]["standardized_mse"], row["tie_value"]
    ))
    learned_member = learned_candidates[0]["candidate_id"]
    learned_selected_val = selected_atoms[learned_member]
    val_paths["learned_selected"] = learned_selected_val
    learned_val_key = _save_path(npz_data, seed, "validation", "learned_selected", learned_selected_val)
    val_metrics["learned_selected"], val_windows["learned_selected"] = val_metrics[learned_member], val_windows[learned_member]

    standalone_rows = []
    for method in STANDALONE_ORDER:
        metric, window = _score(
            val_x, val_y, val_paths[method], val_true_labels, val_true_sign, scales, epsilon, std
        )
        val_metrics[method], val_windows[method] = metric, window
        standalone_rows.append({
            "candidate_id": method,
            "path_key": _key(seed, "validation", method),
            "metrics": metric,
            "eligible": _eligible(metric, base_val_metrics),
            "is_no_op_vs_base": _same_path(val_paths[method], base_path),
        })
    standalone_choice, standalone_eligible_rows = frozen._choose_candidate(
        [{"candidate_id": row["candidate_id"], "metrics": row["metrics"]} for row in standalone_rows],
        base_val_metrics, list(STANDALONE_ORDER),
    )
    standalone_fallback = standalone_choice is None
    if standalone_fallback:
        standalone_member = "base"
    else:
        standalone_member = standalone_choice["candidate_id"]
    val_paths["standalone_selected"] = val_paths[standalone_member]
    standalone_val_key = _save_path(npz_data, seed, "validation", "standalone_selected", val_paths["standalone_selected"])
    val_metrics["standalone_selected"], val_windows["standalone_selected"] = (
        val_metrics[standalone_member], val_windows[standalone_member]
    )

    source_choice = prepared.get("source_validation_selection")
    if source_choice:
        for atom in ATOMS:
            expected = source_choice["atom_selection"][atom]
            if (str(expected["selected_candidate_id"]) != atom_selection[atom]["selected_candidate_id"]
                    or float(expected["selected_alpha"]) != selected_alphas[atom]):
                raise AssertionError(f"recomputed alpha selector differs for {group_key}/seed{seed}/{atom}")
        if source_choice["learned_selected"]["selected_member"] != learned_member:
            raise AssertionError(f"learned_selected identity differs from frozen source for {group_key}/seed{seed}")
        if source_choice["standalone_selected"]["selected_member"] != standalone_member:
            raise AssertionError(f"standalone identity differs from frozen source for {group_key}/seed{seed}")

    # Every validation combination and gate is frozen before any test labels or metrics are computed.
    validation_combinations: dict[str, Any] = {}
    validation_combo_cache: dict[str, dict[str, Any]] = {}
    for combo, members in COMBINATIONS.items():
        member_paths = [val_paths[member] for member in members]
        raw_path = _fixed_mean(member_paths, len(members), arithmetic)
        raw_metrics, raw_windows = _score(
            val_x, val_y, raw_path, val_true_labels, val_true_sign, scales, epsilon, std
        )
        gate = _eligible(raw_metrics, base_val_metrics)
        deployed_path = raw_path if gate else base_path
        deployed_metrics = raw_metrics if gate else base_val_metrics
        deployed_windows = raw_windows if gate else base_val_windows
        raw_id, deployed_id = f"{combo}_raw", f"{combo}_deployed"
        raw_key = _save_path(npz_data, seed, "validation", raw_id, raw_path)
        deployed_key = _save_path(npz_data, seed, "validation", deployed_id, deployed_path)
        unique_count = _unique_path_count(member_paths)
        learned_active = _eligible_learned_component(
            combo, val_paths, selected_alphas[learned_member], members
        )
        common = {
            "members": list(members),
            "unique_member_path_count": unique_count,
            "at_least_two_distinct_members": unique_count >= 2,
            "learned_selected_equals_base": _same_path(val_paths["learned_selected"], base_path),
            "learned_component_active": learned_active,
        }
        gate_row = {
            "eligible": bool(gate),
            "raw_sign_accuracy_strict_improvement": bool(
                raw_metrics["raw_sign_accuracy"] > base_val_metrics["raw_sign_accuracy"]
            ),
            "standardized_mse_within_tolerance": bool(
                raw_metrics["standardized_mse"] <= base_val_metrics["standardized_mse"] + STRICT_MSE_TOLERANCE
            ),
            "mse_tolerance": STRICT_MSE_TOLERANCE,
            "reference_candidate_id": "base",
        }
        validation_combinations[combo] = {
            "raw": {"candidate_id": raw_id, "path_key": raw_key, "metrics": raw_metrics, **common},
            "gate": gate_row,
            "deployed": {
                "candidate_id": deployed_id,
                "path_key": deployed_key,
                "effective_source_candidate_id": raw_id if gate else "base",
                "metrics": deployed_metrics,
                "fallback_to_base": not bool(gate),
                "path_equals_base": _same_path(deployed_path, base_path),
                **{**common, "learned_component_active": learned_active and bool(gate)},
            },
        }
        validation_combo_cache[combo] = {
            "members": members,
            "gate": bool(gate),
            "raw_path": raw_path,
            "deployed_path": deployed_path,
            "raw_metrics": raw_metrics,
            "deployed_metrics": deployed_metrics,
            "raw_windows": raw_windows,
            "deployed_windows": deployed_windows,
            "unique_count": unique_count,
            "learned_active": learned_active,
        }

    for combo in ("learned_threeway", "nohead_threeway"):
        equal = _same_path(
            validation_combo_cache["learned_threeway"]["raw_path"],
            validation_combo_cache["nohead_threeway"]["raw_path"],
        )
        for status in ("raw", "deployed"):
            validation_combinations[combo][status]["raw_learned_threeway_equals_raw_nohead_threeway"] = equal

    # All validation alpha/member/combo choices are now frozen. Only now form or score test paths.
    if prepared.get("test_atoms_need_projection", False):
        for atom in ATOMS:
            atom_scales = scales if atom != "learned_single" else [1]
            atom_epsilon = epsilon if atom != "learned_single" else epsilon[:1]
            prob = prepared["probabilities"]["test"][atom]
            test_paths[atom] = test_paths["base"] if selected_alphas[atom] == 0.0 else frozen.project_trajectory(
                test_x, test_paths["base"], prob.expand(len(test_x), -1, -1, -1, -1),
                atom_scales, atom_epsilon, strength=selected_alphas[atom],
                steps=frozen.PROJECTION_STEPS, max_rms=frozen.MAX_RMS,
            )
    test_paths["learned_selected"] = test_paths[learned_member]
    test_paths["standalone_selected"] = test_paths[standalone_member]
    _save_split(npz_data, seed, "test", test_x, test_y, prepared["origins"]["test"])
    for method, path in test_paths.items():
        _save_path(npz_data, seed, "test", method, path)

    test_true_labels, test_true_sign = frozen._metric_labels(test_x, test_y, scales, epsilon)
    test_methods: dict[str, Any] = {}
    for method in (*STANDALONE_ORDER, *ATOMS, "learned_selected", "standalone_selected"):
        if method in test_metrics:
            continue
        metric, window = _score(
            test_x, test_y, test_paths[method], test_true_labels, test_true_sign, scales, epsilon, std
        )
        test_metrics[method], test_windows[method] = metric, window
        candidate = atom_selection[method]["selected_candidate_id"] if method in ATOMS else method
        test_methods[method] = {
            "candidate_id": candidate,
            "path_key": _key(seed, "test", method),
            "metrics": metric,
        }
    test_methods["learned_selected"]["selected_member"] = learned_member
    test_methods["standalone_selected"]["selected_member"] = standalone_member
    test_metrics["base"] = test_metrics["base"]

    test_combinations: dict[str, Any] = {}
    test_combo_raw_paths: dict[str, torch.Tensor] = {}
    for combo, cache in validation_combo_cache.items():
        members = cache["members"]
        member_paths = [test_paths[member] for member in members]
        raw_path = _fixed_mean(member_paths, len(members), arithmetic)
        test_combo_raw_paths[combo] = raw_path
        raw_metrics, raw_windows = _score(
            test_x, test_y, raw_path, test_true_labels, test_true_sign, scales, epsilon, std
        )
        deployed_path = raw_path if cache["gate"] else test_paths["base"]
        if cache["gate"]:
            deployed_metrics, deployed_windows = raw_metrics, raw_windows
        else:
            deployed_metrics, deployed_windows = test_metrics["base"], test_windows["base"]
        raw_id, deployed_id = f"{combo}_raw", f"{combo}_deployed"
        raw_key = _save_path(npz_data, seed, "test", raw_id, raw_path)
        deployed_key = _save_path(npz_data, seed, "test", deployed_id, deployed_path)
        test_paths[raw_id], test_paths[deployed_id] = raw_path, deployed_path
        raw_metrics["strict_joint_pass"] = _eligible(raw_metrics, test_metrics["base"])
        deployed_metrics["strict_joint_pass"] = _eligible(deployed_metrics, test_metrics["base"])
        deployed_metrics["strict_user_joint_pass"] = deployed_metrics["strict_joint_pass"]
        unique_count = _unique_path_count(member_paths)
        learned_active = _eligible_learned_component(
            combo, test_paths, selected_alphas[learned_member], members
        )
        common = {
            "members": list(members),
            "unique_member_path_count": unique_count,
            "at_least_two_distinct_members": unique_count >= 2,
            "learned_selected_equals_base": _same_path(test_paths["learned_selected"], test_paths["base"]),
            "learned_component_active": learned_active,
        }
        test_combinations[combo] = {
            "raw": {"candidate_id": raw_id, "path_key": raw_key, "metrics": raw_metrics, **common},
            "deployed": {
                "candidate_id": deployed_id,
                "path_key": deployed_key,
                "effective_source_candidate_id": raw_id if cache["gate"] else "base",
                "metrics": deployed_metrics,
                "fallback_to_base": not cache["gate"],
                "path_equals_base": _same_path(deployed_path, test_paths["base"]),
                "validation_gate_eligible": cache["gate"],
                **{**common, "learned_component_active": learned_active and cache["gate"]},
            },
        }
        test_metrics[raw_id], test_windows[raw_id] = raw_metrics, raw_windows
        test_metrics[deployed_id], test_windows[deployed_id] = deployed_metrics, deployed_windows
        test_methods[raw_id] = {"candidate_id": raw_id, "path_key": raw_key, "metrics": raw_metrics}
        test_methods[deployed_id] = {"candidate_id": deployed_id, "path_key": deployed_key, "metrics": deployed_metrics}

    for combo in ("learned_threeway", "nohead_threeway"):
        equal = _same_path(test_combo_raw_paths["learned_threeway"], test_combo_raw_paths["nohead_threeway"])
        for status in ("raw", "deployed"):
            test_combinations[combo][status]["raw_learned_threeway_equals_raw_nohead_threeway"] = equal

    # Old threeway paths are required to reproduce exactly, including source-device arithmetic.
    source_fixed = prepared.get("source_fixed_threeway_test")
    source_fixed_validation = prepared.get("source_fixed_threeway_validation")
    if source_fixed is not None:
        source_test_path = _source_prediction(prepared["source_npz"], source_fixed["prediction_key"], device)
        if not _same_path(test_paths["learned_threeway_deployed"], source_test_path):
            raise AssertionError(f"deployed learned_threeway differs from frozen source test path for {group_key}/seed{seed}")
        source_val_path = _source_prediction(prepared["source_npz"], source_fixed_validation["prediction_key"], device)
        if not _same_path(validation_combo_cache["learned_threeway"]["raw_path"], source_val_path):
            raise AssertionError(f"raw learned_threeway differs from frozen source validation path for {group_key}/seed{seed}")

    strict_flags = {
        method: bool(_eligible(test_metrics[method], test_metrics["base"]))
        for method in (*STANDALONE_ORDER, *ATOMS, "learned_selected", "standalone_selected",
                       *[f"{name}_{status}" for name in COMBINATIONS for status in ("raw", "deployed")])
    }
    compare_specs = []
    for combo in COMBINATIONS:
        for status in ("raw", "deployed"):
            candidate = f"{combo}_{status}"
            for reference in ("base", "standalone_selected"):
                compare_specs.append((f"{candidate}__vs__{reference}", candidate, reference))
    compare_specs.extend([
        ("learned_threeway_deployed__vs__nohead_threeway_deployed", "learned_threeway_deployed", "nohead_threeway_deployed"),
        ("learned_threeway_raw__vs__nohead_threeway_raw", "learned_threeway_raw", "nohead_threeway_raw"),
    ])
    test_comparisons = {
        comparison_id: _emit_comparison(
            comparison_id, seed, candidate, reference,
            test_metrics[candidate], test_metrics[reference],
            test_windows[candidate], test_windows[reference], npz_data,
        )
        for comparison_id, candidate, reference in compare_specs
    }

    validation_result = {
        "base": {"candidate_id": "base", "path_key": _key(seed, "validation", "base"), "metrics": base_val_metrics},
        "atom_alpha_grid": validation_grid,
        "atom_selection": atom_selection,
        "learned_selected": {
            "selected_member": learned_member,
            "selected_candidate_id": atom_selection[learned_member]["selected_candidate_id"],
            "selected_alpha": selected_alphas[learned_member],
            "path_key": learned_val_key,
            "metrics": val_metrics["learned_selected"],
            "eligible": _eligible(val_metrics["learned_selected"], base_val_metrics),
            "learned_selected_equals_base": _same_path(learned_selected_val, base_path),
        },
        "standalone_candidates": standalone_rows,
        "standalone_selected": {
            "selected_member": standalone_member,
            "candidate_id": standalone_member,
            "source_candidate_id": atom_selection[standalone_member]["selected_candidate_id"] if standalone_member in ATOMS else standalone_member,
            "path_key": standalone_val_key,
            "metrics": val_metrics["standalone_selected"],
            "eligible_candidate_ids": [row["candidate_id"] for row in standalone_eligible_rows],
            "fallback_to_base": bool(standalone_fallback),
            "learned_selected_equals_base": _same_path(learned_selected_val, base_path),
        },
        "combinations": validation_combinations,
    }
    test_result = {
        "methods": test_methods,
        "base": test_methods["base"],
        "standalone_selected": test_methods["standalone_selected"],
        "combinations": test_combinations,
        "strict_joint_flags": strict_flags,
    }
    return {"seed": seed, "validation": validation_result, "test": test_result,
            "test_comparisons": test_comparisons}, test_windows


def _eligible_learned_component(combo: str, paths: dict[str, torch.Tensor],
                                alpha: float, members: tuple[str, ...]) -> bool:
    if "learned_selected" not in members:
        return False
    return bool(alpha > 0.0 and not _same_path(paths["learned_selected"], paths["base"]))


def _validate_new_config(cell_dir: Path, config: dict[str, Any], seeds: list[int]) -> None:
    if config.get("smoke", False):
        raise ValueError(f"smoke-trained cell cannot be scored: {cell_dir}")
    if str(config.get("backbone", "")).lower() not in ("linear", "dlinear"):
        raise ValueError(f"unexpected backbone in {cell_dir}: {config.get('backbone')}")
    if list(config.get("seeds", [])) != seeds:
        raise ValueError(f"cell seeds differ from explicit CLI seeds in {cell_dir}")
    if list(config.get("scales", [])) != frozen.SCALES:
        raise ValueError(f"fixed scales differ from protocol in {cell_dir}")
    if list(map(float, config.get("alpha_grid", []))) != frozen.ALPHA_GRID:
        raise ValueError(f"fixed alpha grid differs from protocol in {cell_dir}")
    if list(map(float, config.get("temperature_grid", []))) != [0.5, 1.0, 2.0, 4.0]:
        raise ValueError(f"fixed calibration temperature grid differs from protocol in {cell_dir}")
    if float(config.get("epsilon_quantile", -1)) != 0.2:
        raise ValueError(f"fixed train-only epsilon quantile differs from protocol in {cell_dir}")
    if int(config.get("projection_steps", -1)) != frozen.PROJECTION_STEPS or float(
        config.get("max_rms", -1)
    ) != frozen.MAX_RMS:
        raise ValueError(f"fixed projection settings differ from protocol in {cell_dir}")
    training = config.get("training", {})
    expected_training = {
        "epochs": 40, "patience": 8, "batch_size": 64,
        "learning_rate": 0.001, "direction_weight": 0.1,
        "difference_weight": 1.0,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training setting {key} differs from protocol in {cell_dir}")
    if int(config.get("hidden", -1)) != 32:
        raise ValueError(f"hidden dimension differs from protocol in {cell_dir}")
    data_cfg = config.get("data", {})
    columns = data_cfg.get("value_columns", [])
    if list(map(str, columns)) != [str(index) for index in range(8)]:
        raise ValueError(f"expected exactly the fixed first eight numeric columns in {cell_dir}")
    if int(data_cfg.get("lookback", -1)) != 96 or int(data_cfg.get("horizon", -1)) != 24:
        raise ValueError(f"fixed lookback/horizon differ from protocol in {cell_dir}")
    if list(map(float, data_cfg.get("split_ratios", []))) != [0.6, 0.15, 0.1, 0.15]:
        raise ValueError(f"split ratios differ from protocol in {cell_dir}")
    if int(data_cfg.get("train_stride", -1)) != 4 or int(data_cfg.get("eval_stride", -1)) != 24:
        raise ValueError(f"window strides differ from protocol in {cell_dir}")


def _new_seed_inputs(cell_dir: Path, config: dict[str, Any], checkpoint: dict[str, Any],
                     source_predictions: dict[str, np.ndarray], source_results: dict[str, Any],
                     seed: int, device: torch.device, provenance: dict[str, Any]
                     ) -> dict[str, Any]:
    arrays, data_info = frozen._load_seed_arrays(
        cell_dir, config, checkpoint, source_predictions, seed, device
    )
    model_paths, learned_probs = frozen._seed_model_paths(
        checkpoint, arrays, device, seed, source_predictions
    )
    prior_prob, flat_prob, prior_counts = frozen._prior_probabilities(
        arrays["train_x"], arrays["train_y"], list(config["scales"]), arrays["epsilon"]
    )
    prior_train_labels = frozen.direction_labels(
        frozen.multiscale_slopes(arrays["train_x"], arrays["train_y"], list(config["scales"])),
        arrays["epsilon"],
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
    split_arrays = {
        "validation": {"history": arrays["val_x"], "truth": arrays["val_y"]},
        "test": {"history": arrays["test_x"], "truth": arrays["test_y"]},
    }
    method_paths = {"validation": {}, "test": {}}
    for split in ("validation", "test"):
        for method in ("base", "difference", "residual", "residual_direction"):
            method_paths[split][method] = model_paths[f"{split}__{method}"]
    alpha_paths = {atom: [] for atom in ATOMS}
    base_val = method_paths["validation"]["base"]
    for atom in ATOMS:
        atom_scales = list(config["scales"]) if atom != "learned_single" else [1]
        atom_epsilon = arrays["epsilon"] if atom != "learned_single" else arrays["epsilon"][:1]
        atom_prob = probabilities[atom]
        for alpha in frozen.ALPHA_GRID:
            candidate_id = f"atom__{atom}__alpha_{_alpha_token(alpha)}"
            path = base_val if alpha == 0.0 else frozen.project_trajectory(
                arrays["val_x"], base_val,
                atom_prob.expand(len(arrays["val_x"]), -1, -1, -1, -1),
                atom_scales, atom_epsilon, strength=alpha,
                steps=frozen.PROJECTION_STEPS, max_rms=frozen.MAX_RMS,
            )
            alpha_paths[atom].append({"candidate_id": candidate_id, "alpha": alpha, "path": path})
    probability_sources = {}
    for atom in ATOMS:
        probability_sources[f"validation__{atom}"] = _to_np(probabilities[atom])
        probability_sources[f"test__{atom}"] = _to_np(test_probabilities[atom])
    return {
        "data": split_arrays,
        "origins": {
            "validation": np.asarray(data_info["split_origins"]["validation"], dtype=np.int64),
            "test": np.asarray(data_info["split_origins"]["test"], dtype=np.int64),
        },
        "method_paths": method_paths,
        "atom_alpha_paths": alpha_paths,
        "test_atoms_need_projection": True,
        "probabilities": {"validation": probabilities, "test": test_probabilities},
        "probability_sources": probability_sources,
        "scales": list(config["scales"]),
        "epsilon": arrays["epsilon"],
        "mean": _to_np(arrays["mean"]),
        "std": _to_np(arrays["std"]),
        "prior_counts": _to_np(prior_counts),
        "prior_train_labels": _to_np(prior_train_labels),
        "prior_probabilities": _to_np(prior_prob[0]),
        "flat_probabilities": _to_np(flat_prob[0]),
        "arithmetic": "sum_reciprocal_multiply" if device.type == "cuda" else "sum_divide",
        "source_validation_selection": None,
        "source_fixed_threeway_validation": None,
        "source_fixed_threeway_test": None,
        "source_npz": source_predictions,
        "data_info": data_info,
        "source_results": source_results,
        "provenance": provenance,
    }


def _group_bootstrap(group_key: str, seeds: list[int], seed_results: dict[str, Any],
                     npz_data: dict[str, np.ndarray]) -> dict[str, Any]:
    if not seeds:
        raise ValueError("cannot bootstrap an empty seed set")
    comparison_ids = list(seed_results[str(seeds[0])]["test_comparisons"])
    indices = _bootstrap_indices(len(npz_data[_key(seeds[0], "test", "split", "origins")]))
    indices_key = f"{_safe(group_key)}__bootstrap__block_indices"
    npz_data[indices_key] = indices
    comparisons: dict[str, Any] = {}
    for comparison_id in comparison_ids:
        seed_rows = [seed_results[str(seed)]["test_comparisons"][comparison_id] for seed in seeds]
        candidate_id = seed_rows[0]["candidate_id"]
        reference_id = seed_rows[0]["reference_id"]
        if any(row["candidate_id"] != candidate_id or row["reference_id"] != reference_id for row in seed_rows):
            raise AssertionError(f"comparison identity changed across seeds: {group_key}/{comparison_id}")
        metrics = {}
        for metric in BOOTSTRAP_METRICS:
            keys = [row["seed_window_delta_npz_keys"].get(metric) for row in seed_rows]
            if any(key is None for key in keys):
                continue
            values = [np.asarray(npz_data[key], dtype=np.float64) for key in keys]
            if any(array.shape != values[0].shape for array in values[1:]):
                raise AssertionError(f"window count changed across seeds: {group_key}/{comparison_id}/{metric}")
            seed_mean = np.mean(np.stack(values, axis=0), axis=0, dtype=np.float64)
            seedmean_key = f"{_safe(group_key)}__{comparison_id}__seedmean_window_delta__{metric}"
            npz_data[seedmean_key] = seed_mean
            ci = _bootstrap_ci(seed_mean, indices)
            metrics[metric] = {
                **ci,
                "bootstrap_ci": {"mean": ci["mean"], "ci95": ci["ci95"]},
                "seedmean_window_delta_npz_key": seedmean_key,
                "seed_window_delta_npz_keys": {str(seed): key for seed, key in zip(seeds, keys)},
                "bootstrap_indices_npz_key": indices_key,
                "block_size": min(BOOTSTRAP_BLOCK, int(len(seed_mean))),
                "repeats": BOOTSTRAP_REPEATS,
                "random_seed": 0,
                "seed_mean_before_block_resampling": True,
            }
        comparisons[comparison_id] = {
            "candidate_id": candidate_id,
            "reference_id": reference_id,
            "candidate": candidate_id,
            "reference": reference_id,
            "metrics": metrics,
        }
    return {
        "method": "paired window deltas averaged over fitted seeds, then non-circular moving block bootstrap",
        "block_size": BOOTSTRAP_BLOCK,
        "repeats": BOOTSTRAP_REPEATS,
        "random_seed": 0,
        "seed_mean_before_block_resampling": True,
        "bootstrap_indices_npz_key": indices_key,
        "comparisons": comparisons,
    }


def _test_summary(seed_results: dict[str, Any]) -> dict[str, Any]:
    methods = list(seed_results[next(iter(seed_results))]["test"]["methods"])
    scalar_fields = (
        "raw_sign_accuracy", "standardized_mse", "original_units_mse",
        "mean_3class_accuracy", "mean_macro_f1",
    )
    output = {}
    for method in methods:
        rows = [seed_results[str(seed)]["test"]["methods"][method]["metrics"] for seed in sorted(map(int, seed_results))]
        summary = {}
        for field in scalar_fields:
            values = np.asarray([row[field] for row in rows], dtype=np.float64)
            summary[field] = {"mean": float(values.mean()),
                              "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                              "per_seed": values.tolist()}
        for field in ("accuracy_by_scale", "macro_f1_by_scale"):
            scales = rows[0][field]
            summary[field] = {
                scale: {"mean": float(np.mean([row[field][scale] for row in rows])),
                        "sample_std": float(np.std([row[field][scale] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
                        "per_seed": [float(row[field][scale]) for row in rows]}
                for scale in scales
            }
        output[method] = summary
    return output


def _old_group(group_key: str, source_root: Path, root_manifest: dict[str, Any],
               root_results: dict[str, Any], source_npz: dict[str, np.ndarray],
               seeds: list[int], device: torch.device, npz_data: dict[str, np.ndarray]
               ) -> dict[str, Any]:
    group = root_results["groups"][group_key]
    config, source_results, provenance = _old_cell_provenance(
        source_root, root_manifest, root_results, group
    )
    if list(config.get("seeds", [])) != seeds:
        raise ValueError(f"old cell seeds differ from explicit CLI seeds: {group_key}")
    source_device = str(group.get("device", "cpu")).lower()
    arithmetic = "sum_reciprocal_multiply" if source_device.startswith("cuda") else "sum_divide"
    provenance.update({
        "source_group_device": source_device,
        "combination_arithmetic": arithmetic,
        "source_cohort": group.get("cohort"),
        "source_group_key": group_key,
        "source_data_metadata": group.get("group_data", {}).get("data_metadata"),
        "source_prior_audit_scope": "reused independently audited train-only counts/probabilities from the frozen evaluator; no new CSV label recomputation",
        "source_data_hash_manifest_sha256": _sha256(source_root / "data_sha256.txt") if (source_root / "data_sha256.txt").is_file() else None,
    })
    seed_results: dict[str, Any] = {}
    seed_windows: dict[str, Any] = {}
    for seed in seeds:
        run = group["seeds"][str(seed)]
        prepared, _ = _old_seed_inputs(
            group_key, run, source_npz, seed, device, source_device, arithmetic
        )
        result, windows = _evaluate_seed(group_key, seed, prepared, npz_data, device)
        seed_results[str(seed)] = result
        seed_windows[str(seed)] = windows
    info = group["reconstruction"]["data"] if "reconstruction" in group else group["group_data"]
    data_cfg = config.get("data", {})
    return {
        "group_key": group_key,
        "dataset": group["dataset"],
        "backbone": group["backbone"],
        "cohort": "exploratory",
        "device_for_evaluation": str(device),
        "source_device": source_device,
        "combination_arithmetic": arithmetic,
        "input_columns": list(data_cfg.get("value_columns", [])),
        "seeds": seed_results,
        "validation_selection_protocol": {
            "gate": "raw_sign_accuracy strictly greater than base and standardized_mse <= base + 1e-8",
            "alpha_grid": frozen.ALPHA_GRID,
            "standalone_tie_order": STANDALONE_ORDER,
        },
        "split_metadata": {
            "lookback": info.get("lookback"),
            "horizon": info.get("horizon"),
            "channels": info.get("channels"),
            "split_bounds": info.get("data_metadata", {}).get("split_bounds"),
            "split_ratios": info.get("data_metadata", {}).get("split_ratios"),
            "train_stride": info.get("data_metadata", {}).get("train_stride"),
            "eval_stride": info.get("data_metadata", {}).get("eval_stride"),
            "validation_origins_npz_key_template": "seed{seed}__validation__split__origins",
            "test_origins_npz_key_template": "seed{seed}__test__split__origins",
        },
        "source": provenance,
        "test_summary_mean_sample_std": _test_summary(seed_results),
        "time_block_bootstrap_after_seed_mean": _group_bootstrap(group_key, seeds, seed_results, npz_data),
    }


def _read_sha256_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            digest, filename = parts
            values[str(Path(filename).resolve())] = digest
    return values


def _new_group(cell_dir: Path, seeds: list[int], device: torch.device,
               npz_data: dict[str, np.ndarray]) -> dict[str, Any]:
    cell_dir = cell_dir.expanduser().resolve()
    config = json.loads((cell_dir / "config.json").read_text(encoding="utf-8"))
    source_results = json.loads((cell_dir / "results.json").read_text(encoding="utf-8"))
    _validate_new_config(cell_dir, config, seeds)
    if str(config.get("device", "")).lower() != device.type:
        raise ValueError(f"new-cell evaluation device must match its frozen training device {config.get('device')}: {cell_dir}")
    if source_results.get("checkpoint_reload_verified") is not True:
        raise ValueError(f"checkpoint reload not verified for {cell_dir}")
    if source_results.get("data", {}).get("smoke_truncated", False):
        raise ValueError(f"source cell is smoke-truncated: {cell_dir}")
    audit_path = cell_dir / "independent_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"independent training audit missing: {audit_path}")
    source_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if source_audit.get("verified") is not True or source_audit.get("seeds") != seeds:
        raise ValueError(f"independent training audit failed/incomplete for {cell_dir}")
    checkpoint_path = cell_dir / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if set(checkpoint) != {str(seed) for seed in seeds}:
        raise ValueError(f"checkpoint seed keys are incomplete or unexpected: {cell_dir}")
    with np.load(cell_dir / "predictions.npz", allow_pickle=False) as archive:
        source_predictions = {key: archive[key] for key in archive.files}
    data_path = _resolve_data_path(cell_dir, str(config["data"]["csv_path"]))
    if source_audit.get("source_sha256") != _sha256(data_path):
        raise AssertionError(f"source CSV hash differs from independent audit: {cell_dir}")
    data_root = cell_dir.parents[2]
    matrix_path = data_root / "matrix_verification.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"matrix verification manifest missing: {matrix_path}")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if (matrix.get("wrapper_exit_code") != 0
            or matrix.get("phase") != "full_matrix"
            or matrix.get("protocol_sha256") != "d1b90f4005927c36558d7aa38cb71e50385dbdc08a0eed8ec6e7e10832581852"
            or matrix.get("preflight", {}).get("test_metrics_requested_or_computed") is not False):
        raise ValueError(f"training matrix manifest is incomplete or mismatched: {matrix_path}")
    evaluator_provenance = _cell_provenance(cell_dir, config, source_results, None)
    evaluator_provenance.update({
        "independent_training_audit_path": str(audit_path),
        "independent_training_audit_sha256": _sha256(audit_path),
        "independent_training_audit_verified": True,
        "training_matrix_manifest_path": str(matrix_path),
        "training_matrix_manifest_sha256": _sha256(matrix_path),
        "training_matrix_manifest_verified": True,
        "source_prior_audit_scope": "recomputed from reconstructed train windows; train labels, counts, and probabilities are saved for independent recomputation",
    })
    dataset, backbone, group_key = frozen._cell_identity(cell_dir, config)
    run_seeds = source_results.get("seeds", [])
    if list(map(int, run_seeds)) != seeds:
        raise ValueError(f"source results seed rows are incomplete or out of order: {cell_dir}")
    seed_results: dict[str, Any] = {}
    seed_windows: dict[str, Any] = {}
    for seed in seeds:
        prepared = _new_seed_inputs(
            cell_dir, config, checkpoint[str(seed)], source_predictions, source_results,
            seed, device, evaluator_provenance,
        )
        result, windows = _evaluate_seed(group_key, seed, prepared, npz_data, device)
        seed_results[str(seed)] = result
        seed_windows[str(seed)] = windows
    data_info = prepared["data_info"]
    data_meta = data_info.get("data_metadata", {})
    config_data = config["data"]
    source_data_hash_manifest = data_root / "data_sha256.txt"
    manifest_hashes = _read_sha256_manifest(source_data_hash_manifest)
    original_data_candidates = [
        path for path in manifest_hashes if Path(path).name.lower() == f"{dataset.lower()}.csv"
    ]
    original_data_path = original_data_candidates[0] if len(original_data_candidates) == 1 else None
    if original_data_path and _sha256(Path(original_data_path)) != manifest_hashes[original_data_path]:
        raise AssertionError(f"original CSV differs from its frozen hash manifest: {original_data_path}")
    selected_columns = matrix.get("preflight", {}).get("datasets", {}).get(dataset, {}).get("selected_columns", [])
    if list(map(str, selected_columns)) != list(map(str, config_data.get("value_columns", []))):
        raise AssertionError(f"selected input columns differ from matrix preflight: {dataset}")
    provenance = {
        **evaluator_provenance,
        "original_data_path": original_data_path,
        "original_data_sha256": manifest_hashes.get(original_data_path) if original_data_path else None,
        "derived_data_path": str(data_path),
        "derived_data_sha256": _sha256(data_path),
        "derived_data_columns": list(config_data.get("value_columns", [])),
        "source_data_hash_manifest_path": str(source_data_hash_manifest) if source_data_hash_manifest.is_file() else None,
        "source_data_hash_manifest_sha256": _sha256(source_data_hash_manifest) if source_data_hash_manifest.is_file() else None,
        "source_cohort": "additional_validation",
        "source_matrix_verified": True,
    }
    return {
        "group_key": group_key,
        "dataset": dataset,
        "backbone": backbone,
        "cohort": "additional_validation",
        "device_for_evaluation": str(device),
        "source_device": str(config.get("device", "unknown")),
        "combination_arithmetic": "sum_reciprocal_multiply" if str(config.get("device", "")).startswith("cuda") else "sum_divide",
        "input_columns": list(config_data.get("value_columns", [])),
        "seeds": seed_results,
        "validation_selection_protocol": {
            "gate": "raw_sign_accuracy strictly greater than base and standardized_mse <= base + 1e-8",
            "alpha_grid": frozen.ALPHA_GRID,
            "standalone_tie_order": STANDALONE_ORDER,
        },
        "split_metadata": {
            "lookback": int(config_data["lookback"]),
            "horizon": int(config_data["horizon"]),
            "channels": int(config_data["channels"]),
            "input_columns": list(config_data.get("value_columns", [])),
            "split_bounds": source_results["data"].get("split_bounds"),
            "split_ratios": list(config_data["split_ratios"]),
            "train_stride": int(config_data["train_stride"]),
            "eval_stride": int(config_data["eval_stride"]),
            "validation_origins_npz_key_template": "seed{seed}__validation__split__origins",
            "test_origins_npz_key_template": "seed{seed}__test__split__origins",
            "data_metadata": data_meta,
        },
        "source": provenance,
        "test_summary_mean_sample_std": _test_summary(seed_results),
        "time_block_bootstrap_after_seed_mean": _group_bootstrap(group_key, seeds, seed_results, npz_data),
    }


def _write_group_artifacts(output_dir: Path, group: dict[str, Any], npz_data: dict[str, np.ndarray],
                           evaluator_hash: str, protocol_hash: str) -> dict[str, Any]:
    group_key = group["group_key"]
    group_dir = output_dir / "cell_artifacts" / _safe(group_key)
    group_dir.mkdir(parents=True, exist_ok=False)
    results_path = group_dir / "results.json"
    prediction_path = group_dir / "predictions.npz"
    manifest_path = group_dir / "manifest.json"
    _write_json(results_path, group)
    np.savez_compressed(prediction_path, **npz_data)
    cell_manifest = {
        "schema_version": 1,
        "complete": True,
        "group_key": group_key,
        "dataset": group["dataset"],
        "backbone": group["backbone"],
        "cohort": group["cohort"],
        "seeds": sorted(map(int, group["seeds"])),
        "evaluator_sha256": evaluator_hash,
        "protocol_sha256": protocol_hash,
        "source": group["source"],
        "artifacts": {
            "results_json": {"path": str(results_path.relative_to(output_dir)), "sha256": _sha256(results_path)},
            "predictions_npz": {"path": str(prediction_path.relative_to(output_dir)), "sha256": _sha256(prediction_path)},
        },
        "prediction_key_count": len(npz_data),
    }
    _write_json(manifest_path, cell_manifest)
    group["artifacts"] = {
        "results_json": str(results_path.relative_to(output_dir)),
        "predictions_npz": str(prediction_path.relative_to(output_dir)),
        "manifest_json": str(manifest_path.relative_to(output_dir)),
        "predictions_npz_sha256": _sha256(prediction_path),
    }
    _write_json(results_path, group)
    cell_manifest["artifacts"]["results_json"]["sha256"] = _sha256(results_path)
    _write_json(manifest_path, cell_manifest)
    return group["artifacts"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-evaluator-root", action="append", type=Path, default=[])
    parser.add_argument("--cell", action="append", type=Path, default=[])
    parser.add_argument("--group", action="append", default=[], help="restrict old evaluator roots to these group keys")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort", choices=("exploratory", "additional_validation", "engineering_smoke"), required=True)
    parser.add_argument("--run-scope", choices=("complete", "engineering_smoke"), default="complete")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--expected-evaluator-sha256", default=None,
                        help="required for new-cell scoring; proves the reviewed evaluator was frozen")
    return parser


def main() -> None:
    args = _parser().parse_args()
    seeds = list(args.seeds)
    if len(set(seeds)) != len(seeds) or seeds != sorted(seeds):
        raise ValueError("--seeds must be unique and sorted")
    if not args.old_evaluator_root and not args.cell:
        raise ValueError("provide at least one --old-evaluator-root or --cell")
    if args.cell and args.cohort != "additional_validation":
        raise ValueError("fresh training cells must be labeled additional_validation")
    if args.old_evaluator_root and args.cohort not in ("exploratory", "engineering_smoke"):
        raise ValueError("old evaluator roots must be labeled exploratory or engineering_smoke")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    protocol_path = WORKSPACE / "DTL" / "research" / "fixed_threeway_verification_protocol_2026-09-28.md"
    protocol_sha = _sha256(protocol_path)
    expected_protocol_sha = "d1b90f4005927c36558d7aa38cb71e50385dbdc08a0eed8ec6e7e10832581852"
    if protocol_sha != expected_protocol_sha:
        raise AssertionError(f"frozen protocol hash changed: {protocol_sha}")
    evaluator_snapshot = output_dir / "evaluate_fixed_threeway.py"
    shutil.copy2(Path(__file__).resolve(), evaluator_snapshot)
    protocol_snapshot = output_dir / protocol_path.name
    shutil.copy2(protocol_path, protocol_snapshot)
    evaluator_hash = _sha256(evaluator_snapshot)
    if evaluator_hash != _sha256(Path(__file__).resolve()):
        raise AssertionError("evaluator snapshot differs from executing source")
    if args.cell and args.expected_evaluator_sha256 != evaluator_hash:
        raise ValueError("fresh-cell scoring requires --expected-evaluator-sha256 equal to this frozen source hash")

    output_manifest = {
        "schema_version": 1,
        "complete": False,
        "cohort": args.cohort,
        "run_scope": args.run_scope,
        "requested_groups": [],
        "completed_groups": [],
        "seeds": seeds,
        "device": args.device,
        "torch_num_threads": args.torch_threads,
        "protocol_path": str(protocol_path),
        "protocol_snapshot": str(protocol_snapshot),
        "protocol_sha256": protocol_sha,
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_snapshot": str(evaluator_snapshot),
        "evaluator_sha256": evaluator_hash,
        "old_evaluator_roots": [str(path.expanduser().resolve()) for path in args.old_evaluator_root],
        "requested_cells": [str(path.expanduser().resolve()) for path in args.cell],
        "cell_artifacts": {},
    }
    _write_json(output_dir / "manifest.json", output_manifest)
    root_results = {"schema_version": 1, "protocol_sha256": protocol_sha,
                    "evaluator_sha256": evaluator_hash, "groups": {}}
    group_to_source: dict[str, tuple[Path, dict[str, Any], dict[str, Any], dict[str, np.ndarray]]] = {}

    try:
        for root_arg in args.old_evaluator_root:
            root = root_arg.expanduser().resolve()
            root_manifest_path = root / "output_manifest.json"
            if not root_manifest_path.is_file():
                raise FileNotFoundError(f"old evaluator root has no output manifest: {root}")
            root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
            root_results_doc = json.loads((root / "results.json").read_text(encoding="utf-8"))
            available = set(root_results_doc.get("groups", {}))
            wanted = set(args.group) if args.group else available
            wanted &= available
            if args.group and not wanted:
                continue
            _, _, source_arrays = _load_old_root(root, wanted)
            for group_key in wanted:
                if group_key in group_to_source:
                    raise ValueError(f"duplicate old source group: {group_key}")
                group_to_source[group_key] = (root, root_manifest, root_results_doc, source_arrays)
        missing_groups = set(args.group) - set(group_to_source)
        if missing_groups:
            raise KeyError(f"requested old groups were not found: {sorted(missing_groups)}")
        if args.cell and args.group:
            raise ValueError("--group only applies to old evaluator roots; do not mix it with --cell")
        requested_names = sorted(group_to_source) if not args.cell else []
        if args.cell and group_to_source:
            raise ValueError("one output run cannot mix old exploratory roots and new validation cells")

        for group_key in requested_names:
            root, root_manifest, result_doc, source_arrays = group_to_source[group_key]
            cell_npz: dict[str, np.ndarray] = {}
            group_result = _old_group(group_key, root, root_manifest, result_doc,
                                      source_arrays, seeds, device, cell_npz)
            group_result["artifacts"] = {}
            artifacts = _write_group_artifacts(output_dir, group_result, cell_npz, evaluator_hash, protocol_sha)
            group_result["artifacts"] = artifacts
            root_results["groups"][group_key] = group_result
            output_manifest["requested_groups"].append(group_key)
            output_manifest["completed_groups"].append(group_key)
            output_manifest["cell_artifacts"][group_key] = artifacts
            _write_json(output_dir / "results.json", root_results)
            _write_json(output_dir / "manifest.json", output_manifest)

        for cell_arg in args.cell:
            cell = cell_arg.expanduser().resolve()
            config = json.loads((cell / "config.json").read_text(encoding="utf-8"))
            _, _, group_key = frozen._cell_identity(cell, config)
            if group_key in root_results["groups"]:
                raise ValueError(f"duplicate cell group: {group_key}")
            cell_npz = {}
            group_result = _new_group(cell, seeds, device, cell_npz)
            group_result["artifacts"] = {}
            artifacts = _write_group_artifacts(output_dir, group_result, cell_npz, evaluator_hash, protocol_sha)
            group_result["artifacts"] = artifacts
            root_results["groups"][group_key] = group_result
            output_manifest["requested_groups"].append(group_key)
            output_manifest["completed_groups"].append(group_key)
            output_manifest["cell_artifacts"][group_key] = artifacts
            _write_json(output_dir / "results.json", root_results)
            _write_json(output_dir / "manifest.json", output_manifest)

        requested = set(output_manifest["requested_groups"])
        completed = set(output_manifest["completed_groups"])
        if not requested or requested != completed:
            raise AssertionError("output group set is empty or incomplete")
        output_manifest["complete"] = True
        _write_json(output_dir / "results.json", root_results)
        _write_json(output_dir / "manifest.json", output_manifest)
    except Exception as exc:
        output_manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(output_dir / "manifest.json", output_manifest)
        raise
    print(json.dumps({"output_dir": str(output_dir), "groups": sorted(output_manifest["completed_groups"]),
                      "seeds": seeds, "evaluator_sha256": evaluator_hash}, ensure_ascii=False))


if __name__ == "__main__":
    main()
