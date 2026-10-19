"""Bounded CPU inference audit for projection controls on the ETTh pilots.

This script reuses the immutable source snapshots and checkpoints from the
existing ETTh1/ETTh2 pilot runs. It does not fit models or modify trend_lab.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep this diagnostic off the shared GPUs even if the environment exposes them.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

WORKSPACE = Path(__file__).resolve().parents[2]
RUNS = {
    "ETTh1": WORKSPACE / "DTL/trend_lab_runs/etth1_pilot_20260927T074135Z",
    "ETTh2": WORKSPACE / "DTL/trend_lab_runs/etth2_pilot_20260927T074525Z",
}

# Load the exact package snapshot recorded with these checkpoints.
SNAPSHOT_PACKAGE = RUNS["ETTh1"] / "source"
if not SNAPSHOT_PACKAGE.is_dir():
    raise FileNotFoundError(f"source snapshot not found: {SNAPSHOT_PACKAGE}")
sys.path.insert(0, str(SNAPSHOT_PACKAGE))

import numpy as np
import torch

from trend_lab.data import load_data
from trend_lab.decoding import project_trajectory
from trend_lab.models import DirectionHead, build_base
from trend_lab.operators import direction_labels, fit_thresholds, multiscale_slopes
from trend_lab.metrics import evaluate_predictions


ALPHA_GRID = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
SCALES = [1, 2, 4, 8]
PROJECTION_STEPS = 40
MAX_RMS = 0.5
SEEDS = [0, 1, 2]
CONTROL_VARIANTS = [
    "prior_projection",
    "flat_projection",
    "prior_at_learned_alpha",
    "flat_at_learned_alpha",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_hash_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        row = raw.strip()
        if not row:
            continue
        pieces = row.split(maxsplit=1)
        if len(pieces) == 2:
            result[pieces[1]] = pieces[0]
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
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


def _metric_record(history: torch.Tensor, truth: torch.Tensor, prediction: torch.Tensor,
                   scales: list[int], epsilon: torch.Tensor, std: np.ndarray) -> dict[str, Any]:
    metrics = evaluate_predictions(history, truth, prediction, scales, epsilon)
    accuracies = {str(scale): float(metrics["scales"][str(scale)]["accuracy"]) for scale in scales}
    macro_f1 = {str(scale): float(metrics["scales"][str(scale)]["macro_f1"]) for scale in scales}
    return {
        "mean_3class_accuracy": float(np.mean(list(accuracies.values()))),
        "mean_macro_f1": float(np.mean(list(macro_f1.values()))),
        "raw_sign_accuracy": float(metrics["raw_sign_accuracy"]),
        "standardized_mse": float(metrics["mse"]),
        "original_units_mse": float((((prediction - truth) * torch.as_tensor(
            std, dtype=truth.dtype, device=truth.device)[None, None, :]) ** 2).mean().item()),
        "accuracy_by_scale": accuracies,
        "macro_f1_by_scale": macro_f1,
    }


def _comparison(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in ("mean_3class_accuracy", "mean_macro_f1", "raw_sign_accuracy",
                "standardized_mse", "original_units_mse"):
        left, right = float(candidate[key]), float(reference[key])
        output[key + "_delta"] = left - right
        if key.endswith("_mse"):
            output[key + "_ratio"] = left / right if right else None
            output[key + "_percent_change"] = 100.0 * (left / right - 1.0) if right else None
    output["accuracy_by_scale_delta"] = {
        scale: float(candidate["accuracy_by_scale"][scale] - reference["accuracy_by_scale"][scale])
        for scale in candidate["accuracy_by_scale"]
    }
    output["macro_f1_by_scale_delta"] = {
        scale: float(candidate["macro_f1_by_scale"][scale] - reference["macro_f1_by_scale"][scale])
        for scale in candidate["macro_f1_by_scale"]
    }
    return output


def _legacy_validation_feasible(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    return bool(
        candidate["mean_3class_accuracy"] > reference["mean_3class_accuracy"]
        and candidate["mean_macro_f1"] >= reference["mean_macro_f1"] - 1e-12
        and candidate["standardized_mse"] <= reference["standardized_mse"] * 1.01
    )


def _strict_test_passes(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, bool]:
    mse_ok = candidate["standardized_mse"] <= reference["standardized_mse"] + 1e-8
    raw_sign_ok = candidate["raw_sign_accuracy"] > reference["raw_sign_accuracy"]
    three_class_ok = candidate["mean_3class_accuracy"] > reference["mean_3class_accuracy"]
    return {
        "raw_sign_accuracy_strictly_above_base_and_mse_within_1e_8": bool(raw_sign_ok and mse_ok),
        "mean_3class_accuracy_strictly_above_base_and_mse_within_1e_8": bool(three_class_ok and mse_ok),
        "both_direction_metrics_strictly_above_base_and_mse_within_1e_8": bool(
            raw_sign_ok and three_class_ok and mse_ok
        ),
    }


def _control_probabilities(labels: torch.Tensor, kind: str) -> torch.Tensor:
    """Return [1,S,H,C,3] probabilities without looking at evaluation targets."""
    if kind == "prior":
        # labels are the complete training-segment labels [N,S,H,C].
        return torch.nn.functional.one_hot(labels, num_classes=3).float().mean(dim=0, keepdim=True)
    if kind == "flat":
        n_scales, horizon, channels = labels.shape[1:]
        probabilities = torch.zeros((1, n_scales, horizon, channels, 3), dtype=torch.float32)
        probabilities[..., 1] = 1.0
        return probabilities
    raise ValueError(f"unknown control kind {kind!r}")


def _expanded(probability: torch.Tensor, n: int) -> torch.Tensor:
    return probability.expand(n, -1, -1, -1, -1)


def _evaluate_alpha(history: torch.Tensor, truth: torch.Tensor, base: torch.Tensor,
                    probability: torch.Tensor, epsilon: torch.Tensor, scales: list[int],
                    alpha: float, std: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
    projected = project_trajectory(
        history, base, _expanded(probability, len(history)), scales, epsilon,
        strength=float(alpha), steps=PROJECTION_STEPS, max_rms=MAX_RMS,
    )
    return projected, _metric_record(history, truth, projected, scales, epsilon, std)


def _select_alpha(grid: list[dict[str, Any]], baseline: dict[str, Any]) -> tuple[float, bool]:
    eligible = [
        row for row in grid
        if row["mean_3class_accuracy"] > baseline["mean_3class_accuracy"]
        and row["mean_macro_f1"] >= baseline["mean_macro_f1"] - 1e-12
        and row["standardized_mse"] <= baseline["standardized_mse"] * 1.01
    ]
    if not eligible:
        return 0.0, False
    chosen = sorted(eligible, key=lambda row: (
        -row["mean_3class_accuracy"], row["standardized_mse"], row["alpha"]
    ))[0]
    return float(chosen["alpha"]), True


def _prepare_dataset(dataset: str) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    run_dir = RUNS[dataset]
    pilot_dir = run_dir / "pilot"
    checkpoint_path = pilot_dir / "checkpoint.pt"
    predictions_path = pilot_dir / "predictions.npz"
    results_path = pilot_dir / "results.json"
    config_path = pilot_dir / "config.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if sorted(int(seed) for seed in checkpoint) != SEEDS:
        raise AssertionError(f"unexpected checkpoint seeds for {dataset}: {sorted(checkpoint)}")

    config = copy.deepcopy(checkpoint["0"]["config"])
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    if config != saved_config:
        raise AssertionError(f"checkpoint and saved config differ for {dataset}")
    if config["scales"] != SCALES or config["alpha_grid"] != ALPHA_GRID:
        raise AssertionError(f"protocol drift in {dataset} config")
    if int(config["projection_steps"]) != PROJECTION_STEPS or float(config["max_rms"]) != MAX_RMS:
        raise AssertionError(f"projection settings differ from the fixed protocol for {dataset}")

    data_config = config["data"]
    data_path = (WORKSPACE / "DTL" / data_config["csv_path"]).resolve()
    expected_data_hashes = _read_hash_file(run_dir / "data_sha256.txt")
    expected_hash = next(iter(expected_data_hashes.values()))
    actual_hash = _sha256(data_path)
    if actual_hash != expected_hash:
        raise AssertionError(f"CSV hash mismatch for {dataset}: {actual_hash} != {expected_hash}")
    data_config["csv_path"] = str(data_path)
    data_config["smoke"] = False
    bundle = load_data(data_config)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("smoke") is not False or results.get("seeds") != SEEDS:
        raise AssertionError(f"saved run is not the expected full 3-seed pilot for {dataset}")

    scales = list(config["scales"])
    train_x = torch.as_tensor(bundle.train.x, dtype=torch.float32)
    train_y = torch.as_tensor(bundle.train.y, dtype=torch.float32)
    val_x = torch.as_tensor(bundle.validation.x, dtype=torch.float32)
    val_y = torch.as_tensor(bundle.validation.y, dtype=torch.float32)
    test_x = torch.as_tensor(bundle.test.x, dtype=torch.float32)
    test_y = torch.as_tensor(bundle.test.y, dtype=torch.float32)
    epsilon = fit_thresholds(train_x, train_y, scales, quantile=float(config["epsilon_quantile"]))
    if not torch.allclose(epsilon, checkpoint["0"]["epsilon"], rtol=0.0, atol=1e-7):
        raise AssertionError(f"recomputed epsilon differs from checkpoint for {dataset}")
    for seed in SEEDS:
        if not torch.allclose(epsilon, checkpoint[str(seed)]["epsilon"], rtol=0.0, atol=1e-7):
            raise AssertionError(f"recomputed epsilon differs from seed {seed} checkpoint for {dataset}")
    if not np.allclose(bundle.mean, np.asarray(results["normalization"]["mean"]), rtol=0, atol=1e-6):
        raise AssertionError(f"recomputed training mean differs for {dataset}")
    if not np.allclose(bundle.std, np.asarray(results["normalization"]["std"]), rtol=0, atol=1e-6):
        raise AssertionError(f"recomputed training std differs for {dataset}")
    if bundle.metadata["split_bounds"] != results["data"]["split_bounds"]:
        raise AssertionError(f"recomputed split boundaries differ for {dataset}")

    train_labels = direction_labels(multiscale_slopes(train_x, train_y, scales), epsilon)
    prior_probability = _control_probabilities(train_labels, "prior")
    flat_probability = _control_probabilities(train_labels, "flat")
    saved_archive = np.load(predictions_path, allow_pickle=False)
    saved_predictions = {key: saved_archive[key] for key in saved_archive.files}
    saved_archive.close()
    if not np.array_equal(bundle.test.origins, saved_predictions["origins"]):
        raise AssertionError(f"recreated test origins differ for {dataset}")

    dataset_protocol = {
        "source_run": str(run_dir),
        "pilot_dir": str(pilot_dir),
        "csv_path": str(data_path),
        "csv_sha256_expected": expected_hash,
        "csv_sha256_actual": actual_hash,
        "pilot_config": config,
        "data_metadata": bundle.metadata,
        "normalization": {"mean": bundle.mean.tolist(), "std": bundle.std.tolist()},
        "epsilon": epsilon.tolist(),
        "train_only_prior_probabilities_by_scale_horizon_channel": prior_probability[0].tolist(),
        "flat_probabilities_by_scale_horizon_channel": flat_probability[0].tolist(),
    }
    tensors = {
        "train_x": train_x, "train_y": train_y, "train_labels": train_labels,
        "val_x": val_x, "val_y": val_y, "test_x": test_x, "test_y": test_y,
        "epsilon": epsilon, "prior_probability": prior_probability,
        "flat_probability": flat_probability,
    }
    metadata = {
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "results": results,
        "config": config,
        "saved_predictions": saved_predictions,
        "std": bundle.std,
        "protocol": dataset_protocol,
    }
    return tensors, saved_predictions, metadata


def _run_dataset(dataset: str, output_npz: dict[str, np.ndarray]) -> dict[str, Any]:
    tensors, saved_predictions, metadata = _prepare_dataset(dataset)
    run_dir = metadata["run_dir"]
    checkpoint = metadata["checkpoint"]
    results = metadata["results"]
    config = metadata["config"]
    std = metadata["std"]
    scales = list(config["scales"])
    epsilon = tensors["epsilon"]
    val_x, val_y = tensors["val_x"], tensors["val_y"]
    test_x, test_y = tensors["test_x"], tensors["test_y"]
    base_class = build_base(config["backbone"], 96, 24, 7, int(config["hidden"]))
    head = DirectionHead(96, 24, 7, scales, int(config["hidden"]))
    dataset_result: dict[str, Any] = {
        "source_run": str(run_dir),
        "data_reconstruction": metadata["protocol"],
        "validation_grids": {},
        "runs": [],
    }

    for seed in SEEDS:
        for suffix, recreated in (("history", tensors["test_x"]), ("truth", tensors["test_y"])):
            if not np.array_equal(recreated.numpy(), saved_predictions[f"seed{seed}_{suffix}"]):
                raise AssertionError(f"recreated {suffix} does not match saved NPZ for {dataset} seed {seed}")
        print(f"[{dataset}] seed {seed}: reconstructing checkpoint inference", flush=True)
        seed_key = str(seed)
        seed_checkpoint = checkpoint[seed_key]
        seed_saved = results["runs"][seed]
        base_class.load_state_dict(seed_checkpoint["models"]["base"])
        base_class.eval()
        head.load_state_dict(seed_checkpoint["models"]["direction_head"])
        head.eval()
        with torch.no_grad():
            val_base = base_class(val_x)
            test_base = base_class(test_x)
            val_probability = torch.softmax(
                head(val_x, val_base) / float(seed_checkpoint["temperature"]), dim=-1
            )
            test_probability = torch.softmax(
                head(test_x, test_base) / float(seed_checkpoint["temperature"]), dim=-1
            )

        # Reproduce and verify the saved base/head outputs before attributing controls.
        for suffix, actual in (("base", test_base), ("direction_head", test_base)):
            expected = torch.as_tensor(saved_predictions[f"seed{seed}_{suffix}"], dtype=torch.float32)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected_probability = torch.as_tensor(
            saved_predictions[f"seed{seed}_direction_head_probs"], dtype=torch.float32
        )
        torch.testing.assert_close(test_probability, expected_probability, rtol=1e-5, atol=1e-6)
        max_base_error = float(np.max(np.abs(test_base.numpy() - saved_predictions[f"seed{seed}_base"])))
        max_head_probability_error = float(np.max(np.abs(
            test_probability.numpy() - saved_predictions[f"seed{seed}_direction_head_probs"]
        )))

        learned_alpha = float(seed_checkpoint["selected_alpha"]["projection"])
        if learned_alpha not in ALPHA_GRID:
            raise AssertionError(f"learned alpha not in fixed grid for {dataset} seed {seed}")
        learned_projection = project_trajectory(
            test_x, test_base, test_probability, scales, epsilon,
            strength=learned_alpha, steps=PROJECTION_STEPS, max_rms=MAX_RMS,
        )
        torch.testing.assert_close(
            learned_projection,
            torch.as_tensor(saved_predictions[f"seed{seed}_projection"], dtype=torch.float32),
            rtol=1e-5, atol=1e-6,
        )

        probabilities = {
            "prior": tensors["prior_probability"],
            "flat": tensors["flat_probability"],
        }
        base_val_metrics = _metric_record(val_x, val_y, val_base, scales, epsilon, std)
        base_test_metrics = _metric_record(test_x, test_y, test_base, scales, epsilon, std)
        learned_test_metrics = _metric_record(
            test_x, test_y, learned_projection, scales, epsilon, std
        )
        learned_strict_passes = _strict_test_passes(learned_test_metrics, base_test_metrics)
        saved_base_validation = seed_saved["validation"]["base"]
        saved_base_accuracy = float(np.mean([
            saved_base_validation["scales"][str(scale)]["accuracy"] for scale in scales
        ]))
        saved_base_macro_f1 = float(np.mean([
            saved_base_validation["scales"][str(scale)]["macro_f1"] for scale in scales
        ]))
        if not np.isclose(base_val_metrics["mean_3class_accuracy"], saved_base_accuracy, rtol=0, atol=1e-12):
            raise AssertionError(f"recreated validation direction accuracy differs for {dataset} seed {seed}")
        if not np.isclose(base_val_metrics["mean_macro_f1"], saved_base_macro_f1, rtol=0, atol=1e-12):
            raise AssertionError(f"recreated validation macro-F1 differs for {dataset} seed {seed}")
        if not np.isclose(base_val_metrics["standardized_mse"], saved_base_validation["mse"], rtol=1e-5, atol=1e-7):
            raise AssertionError(f"recreated validation MSE differs for {dataset} seed {seed}")

        grids: dict[str, list[dict[str, Any]]] = {}
        validation_prediction_cache: dict[str, dict[float, tuple[torch.Tensor, dict[str, Any]]]] = {}
        selected_alpha: dict[str, float] = {}
        selected_feasible: dict[str, bool] = {}
        for kind in ("prior", "flat"):
            rows: list[dict[str, Any]] = []
            validation_prediction_cache[kind] = {}
            for alpha in ALPHA_GRID:
                prediction, metric = _evaluate_alpha(
                    val_x, val_y, val_base, probabilities[kind], epsilon, scales, alpha, std
                )
                validation_prediction_cache[kind][alpha] = (prediction, metric)
                row = {"alpha": alpha, **metric}
                row["standardized_mse_ratio_to_base"] = (
                    metric["standardized_mse"] / base_val_metrics["standardized_mse"]
                    if base_val_metrics["standardized_mse"] else None
                )
                row["legacy_selection_feasible"] = bool(
                    metric["mean_3class_accuracy"] > base_val_metrics["mean_3class_accuracy"]
                    and metric["mean_macro_f1"] >= base_val_metrics["mean_macro_f1"] - 1e-12
                    and metric["standardized_mse"] <= base_val_metrics["standardized_mse"] * 1.01
                )
                rows.append(row)
            selected_alpha[kind], selected_feasible[kind] = _select_alpha(rows, base_val_metrics)
            grids[kind] = rows
            dataset_result["validation_grids"].setdefault(seed_key, {})[kind] = {
                "baseline": base_val_metrics,
                "alpha_grid": rows,
                "selected_alpha": selected_alpha[kind],
                "selected_feasible": selected_feasible[kind],
                "fallback_to_zero": not selected_feasible[kind],
                "selection_rule": (
                    "mean 3class accuracy strictly above base; mean macro-F1 >= base; "
                    "standardized MSE <= 1.01 * base; rank by accuracy descending, "
                    "then MSE ascending, then alpha ascending; choose 0 if no feasible alpha"
                ),
            }

        # Outputs contain only the alpha chosen on validation or the original learned
        # projection's validation alpha. Test outcomes never feed alpha selection.
        variant_specs = {
            "prior_projection": ("prior", selected_alpha["prior"], "independent_validation_selection"),
            "flat_projection": ("flat", selected_alpha["flat"], "independent_validation_selection"),
            "prior_at_learned_alpha": ("prior", learned_alpha, "fixed_to_saved_learned_validation_alpha"),
            "flat_at_learned_alpha": ("flat", learned_alpha, "fixed_to_saved_learned_validation_alpha"),
        }
        variants: dict[str, Any] = {}
        variant_test_predictions: dict[str, torch.Tensor] = {}
        for variant_name, (kind, alpha, selection_mode) in variant_specs.items():
            if alpha in validation_prediction_cache[kind]:
                val_prediction, val_metrics = validation_prediction_cache[kind][alpha]
            else:
                val_prediction, val_metrics = _evaluate_alpha(
                    val_x, val_y, val_base, probabilities[kind], epsilon, scales, alpha, std
                )
            test_prediction, test_metrics = _evaluate_alpha(
                test_x, test_y, test_base, probabilities[kind], epsilon, scales, alpha, std
            )
            variant_test_predictions[variant_name] = test_prediction
            independent = selection_mode == "independent_validation_selection"
            variants[variant_name] = {
                "control_probability_source": (
                    "train-only class frequencies per scale/horizon/channel"
                    if kind == "prior" else "fixed one-hot flat class at every position"
                ),
                "selection_mode": selection_mode,
                "is_independently_selected_candidate": independent,
                "selected_alpha": float(alpha),
                "selection_feasible_under_legacy_rule": (
                    selected_feasible[kind] if independent else None
                ),
                "fallback_to_zero": (
                    (not selected_feasible[kind]) if independent else False
                ),
                "validation_at_applied_alpha": val_metrics,
                "validation_alpha_is_legacy_feasible": _legacy_validation_feasible(
                    val_metrics, base_val_metrics
                ),
                "test": test_metrics,
                "test_comparison_to_base": _comparison(test_metrics, base_test_metrics),
                "test_comparison_to_learned_projection": _comparison(
                    test_metrics, learned_test_metrics
                ),
                "strict_test_passes_vs_base": _strict_test_passes(test_metrics, base_test_metrics),
                "strict_test_passes_vs_learned_projection": _strict_test_passes(
                    test_metrics, learned_test_metrics
                ),
                "strict_joint_pass_vs_base_test": _strict_test_passes(
                    test_metrics, base_test_metrics
                )["both_direction_metrics_strictly_above_base_and_mse_within_1e_8"],
            }
            prefix = f"{dataset}_{seed}"
            output_npz[f"{prefix}_test_history"] = test_x.numpy()
            output_npz[f"{prefix}_test_truth"] = test_y.numpy()
            output_npz[f"{prefix}_base"] = test_base.numpy()
            output_npz[f"{prefix}_learned_projection"] = learned_projection.numpy()
            output_npz[f"{prefix}_{variant_name}"] = test_prediction.numpy()

        output_npz[f"{dataset}_test_origins"] = saved_predictions["origins"]
        output_npz[f"{dataset}_prior_probabilities"] = probabilities["prior"][0].numpy()
        output_npz[f"{dataset}_flat_probabilities"] = probabilities["flat"][0].numpy()
        dataset_result["runs"].append({
            "seed": seed,
            "learned_projection_validation_selected_alpha": learned_alpha,
            "inference_reconstruction": {
                "checkpoint_loaded_weights_only_cpu": True,
                "torch_set_num_threads": 2,
                "base_test_max_abs_error_vs_saved_npz": max_base_error,
                "head_probability_max_abs_error_vs_saved_npz": max_head_probability_error,
                "base_and_head_within_rtol_1e-5_atol_1e-6": True,
                "learned_projection_within_rtol_1e-5_atol_1e-6": True,
                "recreated_test_history_truth_origins_match_saved": True,
                "train_label_prior_uses_only_training_segment": True,
            },
            "test_reference_metrics": {
                "base": base_test_metrics,
                "learned_projection": learned_test_metrics,
            },
            "learned_projection_test_comparison_to_base": _comparison(
                learned_test_metrics, base_test_metrics
            ),
            "learned_projection_strict_test_passes_vs_base": learned_strict_passes,
            "variants": variants,
        })
        print(f"[{dataset}] seed {seed}: controls complete; val alphas "
              f"prior={selected_alpha['prior']:g}, flat={selected_alpha['flat']:g}; "
              f"learned={learned_alpha:g}", flush=True)

    dataset_result["summary"] = _summarize_dataset(dataset_result["runs"])
    return dataset_result


def _summarize_dataset(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    def comparison_summary(comparisons: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "mean_raw_sign_accuracy_change_percentage_points": float(np.mean([
                row["raw_sign_accuracy_delta"] * 100.0 for row in comparisons
            ])),
            "mean_3class_accuracy_change_percentage_points": float(np.mean([
                row["mean_3class_accuracy_delta"] * 100.0 for row in comparisons
            ])),
            "mean_standardized_mse_percent_change": float(np.mean([
                row["standardized_mse_percent_change"] for row in comparisons
            ])),
        }

    for variant_name in CONTROL_VARIANTS:
        records = [run["variants"][variant_name] for run in runs]
        metric_fields = ("mean_3class_accuracy", "mean_macro_f1", "raw_sign_accuracy",
                         "standardized_mse", "original_units_mse")
        test_aggregate: dict[str, Any] = {}
        for field in metric_fields:
            values = np.asarray([record["test"][field] for record in records], dtype=float)
            test_aggregate[field] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "per_seed": values.tolist(),
            }
        summary[variant_name] = {
            "test_metrics": test_aggregate,
            "mean_change_vs_base": comparison_summary([
                record["test_comparison_to_base"] for record in records
            ]),
            "mean_change_vs_learned_projection": comparison_summary([
                record["test_comparison_to_learned_projection"] for record in records
            ]),
            "strict_test_pass_counts_vs_base": {
                key: sum(bool(record["strict_test_passes_vs_base"][key]) for record in records)
                for key in records[0]["strict_test_passes_vs_base"]
            },
            "strict_test_pass_counts_vs_learned_projection": {
                key: sum(bool(record["strict_test_passes_vs_learned_projection"][key]) for record in records)
                for key in records[0]["strict_test_passes_vs_learned_projection"]
            },
            "selected_alpha_per_seed": [record["selected_alpha"] for record in records],
        }
    summary["learned_projection_vs_base"] = {
        "test_metrics": {
            field: {
                "mean": float(np.mean([run["test_reference_metrics"]["learned_projection"][field]
                                        for run in runs])),
                "sample_std": float(np.std([run["test_reference_metrics"]["learned_projection"][field]
                                             for run in runs], ddof=1)),
            }
            for field in ("mean_3class_accuracy", "mean_macro_f1", "raw_sign_accuracy",
                          "standardized_mse", "original_units_mse")
        },
        "mean_change_vs_base": comparison_summary([
            run["learned_projection_test_comparison_to_base"] for run in runs
        ]),
        "strict_test_pass_counts": {
            key: sum(bool(run["learned_projection_strict_test_passes_vs_base"][key]) for run in runs)
            for key in runs[0]["learned_projection_strict_test_passes_vs_base"]
        },
        "per_seed": [run["learned_projection_strict_test_passes_vs_base"] for run in runs],
    }
    return summary


def _markdown_report(results: dict[str, Any], output_dir: Path, code_hash: str) -> str:
    lines = [
        "# ETTh Projection Control Attribution Probe",
        "",
        f"Run directory: `{output_dir}`  ",
        f"Probe script SHA-256: `{code_hash}`",
        "",
        "This is a bounded attribution diagnostic on the already inspected ETTh validation/test segments. "
        "It is not an independent test. No model was trained. All checkpoint inference and projection "
        "ran on CPU with two Torch threads.",
        "",
        "Independent candidates use the fixed alpha grid `[0, 0.1, 0.3, 1, 3, 10]` and the legacy "
        "validation rule: mean three-class accuracy must strictly improve, mean macro-F1 must not fall, "
        "and standardized MSE must be at most 1.01 times base; if none qualify, alpha is zero. "
        "`prior_at_learned_alpha` and `flat_at_learned_alpha` are fixed-strength diagnostics using the "
        "saved learned projection's validation-selected alpha; they are not separately selected candidates.",
        "",
        "Two strict test checks use standardized MSE <= base + 1e-8: raw-sign accuracy must be strictly "
        "above base for the raw-sign check; mean three-class accuracy must be strictly above base for "
        "the three-class check. The combined flag requires both checks. These test flags do not require "
        "macro-F1. The legacy validation gate remains mean three-class accuracy > base, macro-F1 >= base, "
        "and MSE <= 1.01 * base.",
        "",
        "## Per-seed results",
        "",
        "| Dataset | Seed | Variant | Alpha | Val acc | Val macro-F1 | Test acc | Test macro-F1 | "
        "Test MSE (std) | Test MSE (raw) | Raw sign acc | Δ acc vs base | Δ MSE vs base | "
        "Δ acc vs learned | Δ MSE vs learned | Raw+MSE | 3class+MSE | Both |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for dataset, dataset_result in results["datasets"].items():
        for run in dataset_result["runs"]:
            for variant in CONTROL_VARIANTS:
                item = run["variants"][variant]
                val, test = item["validation_at_applied_alpha"], item["test"]
                base_delta = item["test_comparison_to_base"]
                learned_delta = item["test_comparison_to_learned_projection"]
                passes = item["strict_test_passes_vs_base"]
                lines.append(
                    f"| {dataset} | {run['seed']} | {variant} | {item['selected_alpha']:g} | "
                    f"{val['mean_3class_accuracy']:.4f} | {val['mean_macro_f1']:.4f} | "
                    f"{test['mean_3class_accuracy']:.4f} | {test['mean_macro_f1']:.4f} | "
                    f"{test['standardized_mse']:.5f} | {test['original_units_mse']:.5f} | "
                    f"{test['raw_sign_accuracy']:.4f} | "
                    f"{base_delta['mean_3class_accuracy_delta']:+.4f} | "
                    f"{base_delta['standardized_mse_delta']:+.5f} | "
                    f"{learned_delta['mean_3class_accuracy_delta']:+.4f} | "
                    f"{learned_delta['standardized_mse_delta']:+.5f} | "
                    f"{'yes' if passes['raw_sign_accuracy_strictly_above_base_and_mse_within_1e_8'] else 'no'} | "
                    f"{'yes' if passes['mean_3class_accuracy_strictly_above_base_and_mse_within_1e_8'] else 'no'} | "
                    f"{'yes' if passes['both_direction_metrics_strictly_above_base_and_mse_within_1e_8'] else 'no'} |"
                )
    lines.extend(["", "## Aggregate test results", "", "| Dataset | Variant | Alpha mean | Accuracy | Macro-F1 | "
                  "MSE (std) | MSE (raw) | Raw sign accuracy | Δ raw sign (pp) | MSE change (%) | "
                  "Δ 3class (pp) | Raw+MSE / 3 | 3class+MSE / 3 | Both / 3 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for dataset, dataset_result in results["datasets"].items():
        for variant in CONTROL_VARIANTS:
            aggregate = dataset_result["summary"][variant]
            metrics = aggregate["test_metrics"]
            alphas = aggregate["selected_alpha_per_seed"]
            passes = aggregate["strict_test_pass_counts_vs_base"]
            changes = aggregate["mean_change_vs_base"]
            alpha_mean = float(np.mean(alphas))
            lines.append(
                f"| {dataset} | {variant} | {alpha_mean:.3g} | "
                f"{metrics['mean_3class_accuracy']['mean']:.4f} | "
                f"{metrics['mean_macro_f1']['mean']:.4f} | "
                f"{metrics['standardized_mse']['mean']:.5f} | "
                f"{metrics['original_units_mse']['mean']:.5f} | "
                f"{metrics['raw_sign_accuracy']['mean']:.4f} | "
                f"{changes['mean_raw_sign_accuracy_change_percentage_points']:+.3f} | "
                f"{changes['mean_standardized_mse_percent_change']:+.2f} | "
                f"{changes['mean_3class_accuracy_change_percentage_points']:+.3f} | "
                f"{passes['raw_sign_accuracy_strictly_above_base_and_mse_within_1e_8']} / 3 | "
                f"{passes['mean_3class_accuracy_strictly_above_base_and_mse_within_1e_8']} / 3 | "
                f"{passes['both_direction_metrics_strictly_above_base_and_mse_within_1e_8']} / 3 |"
            )
        learned = dataset_result["summary"]["learned_projection_vs_base"]
        learned_metrics = learned["test_metrics"]
        learned_changes = learned["mean_change_vs_base"]
        learned_passes = learned["strict_test_pass_counts"]
        learned_alphas = [run["learned_projection_validation_selected_alpha"]
                          for run in dataset_result["runs"]]
        lines.append(
            f"| {dataset} | learned_projection | {float(np.mean(learned_alphas)):.3g} | "
            f"{learned_metrics['mean_3class_accuracy']['mean']:.4f} | "
            f"{learned_metrics['mean_macro_f1']['mean']:.4f} | "
            f"{learned_metrics['standardized_mse']['mean']:.5f} | "
            f"{learned_metrics['original_units_mse']['mean']:.5f} | "
            f"{learned_metrics['raw_sign_accuracy']['mean']:.4f} | "
            f"{learned_changes['mean_raw_sign_accuracy_change_percentage_points']:+.3f} | "
            f"{learned_changes['mean_standardized_mse_percent_change']:+.2f} | "
            f"{learned_changes['mean_3class_accuracy_change_percentage_points']:+.3f} | "
            f"{learned_passes['raw_sign_accuracy_strictly_above_base_and_mse_within_1e_8']} / 3 | "
            f"{learned_passes['mean_3class_accuracy_strictly_above_base_and_mse_within_1e_8']} / 3 | "
            f"{learned_passes['both_direction_metrics_strictly_above_base_and_mse_within_1e_8']} / 3 |"
        )
    lines.extend([
        "",
        "## Learned projection strict checks",
        "",
        "| Dataset | Seed | Raw-sign + MSE | Three-class + MSE | Both |",
        "|---|---:|:---:|:---:|:---:|",
    ])
    for dataset, dataset_result in results["datasets"].items():
        for run in dataset_result["runs"]:
            passes = run["learned_projection_strict_test_passes_vs_base"]
            lines.append(
                f"| {dataset} | {run['seed']} | "
                f"{'yes' if passes['raw_sign_accuracy_strictly_above_base_and_mse_within_1e_8'] else 'no'} | "
                f"{'yes' if passes['mean_3class_accuracy_strictly_above_base_and_mse_within_1e_8'] else 'no'} | "
                f"{'yes' if passes['both_direction_metrics_strictly_above_base_and_mse_within_1e_8'] else 'no'} |"
            )
    lines.extend([
        "",
        "All per-seed metrics, all independent validation alpha grids, class priors, source/data hashes, "
        "and reconstructed predictions are in `results.json`, `predictions.npz`, and `resolved_protocol.json`.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    torch.set_num_threads(2)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = WORKSPACE / "DTL/trend_lab_runs" / f"projection_controls_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    script_hash = _sha256(Path(__file__).resolve())
    snapshot_hash_files = {name: _sha256(run_dir / "source_sha256.txt") for name, run_dir in RUNS.items()}
    source_tree_hashes = {
        name: _read_hash_file(run_dir / "source_sha256.txt") for name, run_dir in RUNS.items()
    }
    imported_modules = (
        "trend_lab/__init__.py", "trend_lab/data.py", "trend_lab/decoding.py",
        "trend_lab/metrics.py", "trend_lab/models.py", "trend_lab/operators.py",
    )
    for module_path in imported_modules:
        module_hashes = {name: source_tree_hashes[name].get(module_path) for name in RUNS}
        if len(set(module_hashes.values())) != 1 or None in module_hashes.values():
            raise AssertionError(f"ETTh source snapshots differ for imported module {module_path}: {module_hashes}")

    resolved_protocol = {
        "run_id_utc": timestamp,
        "purpose": "attribution-only controls on the existing ETTh pilot validation/test segments; not an independent test",
        "training": "none; checkpoints are loaded weights_only=True with map_location=cpu",
        "runtime": {
            "device": "cpu",
            "torch_num_threads": 2,
            "cuda_visible_devices": "",
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "code": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": script_hash,
            "snapshot_source_sha256_file_hashes": snapshot_hash_files,
            "snapshot_source_tree_hashes": source_tree_hashes,
            "loaded_package_snapshot": str(SNAPSHOT_PACKAGE),
        },
        "frozen_protocol": {
            "datasets": ["ETTh1", "ETTh2"],
            "seeds": SEEDS,
            "scales": SCALES,
            "alpha_grid": ALPHA_GRID,
            "projection_steps": PROJECTION_STEPS,
            "max_rms": MAX_RMS,
            "prior_projection": "train-label frequencies independently for each scale/horizon/channel; reused at all validation/test windows; no target access",
            "flat_projection": "one-hot flat class (class index 1) at every scale/horizon/channel/window",
            "independent_validation_selection": "strict accuracy improvement; macro-F1 >= base; standardized MSE <= 1.01 * base; rank accuracy desc, MSE asc, alpha asc; fallback alpha 0",
            "fixed_strength_diagnostics": {
                "prior_at_learned_alpha": "prior probabilities at that seed's saved learned projection alpha, selected on validation in the original pilot",
                "flat_at_learned_alpha": "flat probabilities at that seed's saved learned projection alpha, selected on validation in the original pilot",
                "candidate_status": "diagnostics only; no independent alpha selection and no test tuning",
            },
            "strict_user_test_pass": {
                "raw_sign": "raw sign accuracy > reference AND standardized MSE <= reference + 1e-8",
                "mean_3class": "mean three-class accuracy > reference AND standardized MSE <= reference + 1e-8",
                "macro_f1": "reported separately; not an extra requirement for either strict user criterion",
            },
            "test_use": "reported for attribution only; the ETTh test segments were already inspected in the pilot",
        },
    }

    all_predictions: dict[str, np.ndarray] = {}
    dataset_results: dict[str, Any] = {}
    for dataset in RUNS:
        print(f"[{dataset}] loading recorded CPU snapshot and recreating splits", flush=True)
        dataset_results[dataset] = _run_dataset(dataset, all_predictions)

    result_bundle = {
        "schema_version": 1,
        "run_id_utc": timestamp,
        "resolved_protocol": resolved_protocol,
        "datasets": dataset_results,
    }
    _write_json(output_dir / "resolved_protocol.json", resolved_protocol)
    _write_json(output_dir / "results.json", result_bundle)
    (output_dir / "probe_projection_controls.py").write_bytes(Path(__file__).resolve().read_bytes())
    np.savez_compressed(output_dir / "predictions.npz", **all_predictions)
    (output_dir / "code_sha256.txt").write_text(
        f"{script_hash}  {Path(__file__).resolve()}\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        _markdown_report(result_bundle, output_dir, script_hash), encoding="utf-8"
    )
    print(f"[complete] output={output_dir}", flush=True)
    print(f"[complete] script_sha256={script_hash}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
