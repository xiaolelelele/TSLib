#!/usr/bin/env python3
"""Independent NumPy audit for validation-first trend-fusion artifacts.

Reads JSON/NPZ artifacts (allow_pickle=False). It does not load checkpoints or
import the fusion evaluator. It independently recomputes the two selection
metrics plus three-class scale accuracies, replays validation selections,
checks selected test mixtures and bootstrap inputs, then exits nonzero on any
mismatch. Prior validation checks saved counts against probabilities; it does
not independently rebuild train labels from the source CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ATOL = 3e-6
STRICT_MSE_TOLERANCE = 1e-8
ALPHA_GRID = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
STANDALONE_ORDER = [
    "base", "difference", "residual", "residual_direction",
    "learned_single", "learned_multi", "prior", "flat",
]
PAIR_MEMBERS = {
    "base_plus_difference": ("base", "difference"),
    "base_plus_residual": ("base", "residual"),
    "base_plus_residual_direction": ("base", "residual_direction"),
    "learned_selected_plus_flat": ("learned_selected", "flat"),
    "learned_selected_plus_prior": ("learned_selected", "prior"),
    "learned_selected_plus_difference": ("learned_selected", "difference"),
}
FUSION_ORDER = list(PAIR_MEMBERS) + ["threeway_mean"]


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


def npz_key(group: str, seed: int, split: str, candidate_id: str, field: str = "path") -> str:
    return f"{group}__seed{seed}__{split}__{candidate_id}__{field}"


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


def labels(history: np.ndarray, path: np.ndarray, scales: list[int], epsilon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(history, dtype=np.float64)
    path = np.asarray(path, dtype=np.float64)
    epsilon = np.asarray(epsilon, dtype=np.float64)
    joined = np.concatenate([history, path], axis=1)
    origin = history.shape[1] - 1
    steps = np.arange(1, path.shape[1] + 1)
    slope_list = [
        (joined[:, origin + steps, :] - joined[:, origin + steps - scale, :]) / scale
        for scale in scales
    ]
    slopes = np.stack(slope_list, axis=1)
    eps = epsilon[None, :, None, :]
    classes = np.where(slopes > eps, 2, np.where(slopes < -eps, 0, 1)).astype(np.int64)
    raw_sign = np.sign(np.diff(np.concatenate([history[:, -1:, :], path], axis=1), axis=1))
    return classes, raw_sign


def score(history: np.ndarray, truth: np.ndarray, path: np.ndarray,
          scales: list[int], epsilon: np.ndarray, std: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    true_classes, true_sign = labels(history, truth, scales, epsilon)
    pred_classes, pred_sign = labels(history, path, scales, epsilon)
    err = path.astype(np.float64) - truth.astype(np.float64)
    scale_accuracy: dict[str, float] = {}
    per_window_scale: dict[str, np.ndarray] = {}
    for i, scale in enumerate(scales):
        key = str(scale)
        per_window_scale[key] = np.mean(true_classes[:, i] == pred_classes[:, i], axis=(1, 2))
        scale_accuracy[key] = float(np.mean(true_classes[:, i] == pred_classes[:, i]))
    raw_window = np.mean(true_sign == pred_sign, axis=(1, 2))
    mse_window = np.mean(err ** 2, axis=(1, 2))
    original_window = np.mean((err * np.asarray(std, dtype=np.float64).reshape(1, 1, -1)) ** 2, axis=(1, 2))
    mean3_window = np.mean(np.stack(list(per_window_scale.values()), axis=1), axis=1)
    metric = {
        "raw_sign_accuracy": float(np.mean(true_sign == pred_sign)),
        "standardized_mse": float(np.mean(err ** 2)),
        "original_units_mse": float(np.mean((err * np.asarray(std, dtype=np.float64).reshape(1, 1, -1)) ** 2)),
        "accuracy_by_scale": scale_accuracy,
        "mean_3class_accuracy": float(np.mean(list(scale_accuracy.values()))),
    }
    window = {
        "raw_sign_accuracy": raw_window,
        "mean_3class_accuracy": mean3_window,
        "standardized_mse": mse_window,
        "original_units_mse": original_window,
        **{f"accuracy_scale_{k}": v for k, v in per_window_scale.items()},
    }
    return metric, window


def check_reported_metrics(reported: dict[str, Any], measured: dict[str, Any], label: str) -> None:
    for key in ("raw_sign_accuracy", "standardized_mse", "original_units_mse", "mean_3class_accuracy"):
        close(reported[key], measured[key], f"{label}.{key}")
    for scale, value in measured["accuracy_by_scale"].items():
        close(reported["accuracy_by_scale"][scale], value, f"{label}.accuracy_by_scale.{scale}")


def check_delta_report(reported: dict[str, Any], candidate: dict[str, np.ndarray],
                       reference: dict[str, np.ndarray], label: str) -> None:
    for metric, candidate_values in candidate.items():
        reference_values = reference[metric]
        expected_delta = float(np.mean(candidate_values - reference_values))
        close(reported[f"{metric}_mean_delta"], expected_delta, f"{label}.{metric}_mean_delta")
        if metric in ("standardized_mse", "original_units_mse"):
            ref_mean = float(np.mean(reference_values))
            cand_mean = float(np.mean(candidate_values))
            expected_pct = 100.0 * (cand_mean / ref_mean - 1.0) if ref_mean != 0 else None
            if expected_pct is None:
                require(reported.get(f"{metric}_percent_change") is None,
                        f"{label}.{metric}_percent_change should be null")
            else:
                close(reported[f"{metric}_percent_change"], expected_pct,
                      f"{label}.{metric}_percent_change")


def eligible(metric: dict[str, Any], base: dict[str, Any]) -> bool:
    return bool(
        metric["raw_sign_accuracy"] > base["raw_sign_accuracy"]
        and metric["standardized_mse"] <= base["standardized_mse"] + STRICT_MSE_TOLERANCE
    )


def best(rows: list[dict[str, Any]], base: dict[str, Any], order: list[str],
         candidate_id_field: str = "candidate_id", tie_field: str | None = None) -> dict[str, Any] | None:
    eligible_rows = [r for r in rows if eligible(r["metrics"], base)]
    rank = {value: i for i, value in enumerate(order)}
    eligible_rows.sort(key=lambda r: (
        -r["metrics"]["raw_sign_accuracy"], r["metrics"]["standardized_mse"],
        rank.get(r.get(candidate_id_field, ""), len(rank)),
        r.get(tie_field, 0.0) if tie_field else 0.0,
    ))
    return eligible_rows[0] if eligible_rows else None


def path_for(archive: Any, group: str, seed: int, split: str, candidate_id: str) -> np.ndarray:
    return get_array(archive, npz_key(group, seed, split, candidate_id, "path"))


def verify_key_and_metrics(archive: Any, row: dict[str, Any], group: str, seed: int,
                           split: str, history: np.ndarray, truth: np.ndarray,
                           scales: list[int], epsilon: np.ndarray, std: np.ndarray,
                           label: str) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    key = row["prediction_key"]
    path = get_array(archive, key)
    candidate_id = row.get("candidate_id")
    if candidate_id:
        require(key == npz_key(group, seed, split, candidate_id, "path"),
                f"{label}: key does not match candidate ID ({key}, {candidate_id})")
    measured, window = score(history, truth, path, scales, epsilon, std)
    check_reported_metrics(row["metrics"], measured, label)
    return path, measured, window


def check_bootstrap(group_key: str, group: dict[str, Any], archive: Any,
                    per_seed_windows: dict[int, dict[str, dict[str, np.ndarray]]]) -> int:
    bootstrap = group["time_block_bootstrap_after_seed_mean"]
    count = 0
    for comparison_id, spec in bootstrap.items():
        candidate = spec.get("candidate")
        reference = spec.get("reference")
        metric_rows = spec.get("metrics", {})
        # Fixed-reference comparisons can be independently rebuilt directly.
        fixed_reference = (
            candidate in per_seed_windows[sorted(per_seed_windows)[0]]
            and reference in per_seed_windows[sorted(per_seed_windows)[0]]
        )
        matched_dynamic = candidate == "fusion_selected" and bool(spec.get("seed_references"))
        if not fixed_reference and not matched_dynamic:
            # Older, incorrectly labeled seed-specific comparisons are deliberately rejected.
            fail(f"{comparison_id}: unsupported or unpaired bootstrap reference")
        for metric, interval in metric_rows.items():
            if fixed_reference:
                for seed in sorted(per_seed_windows):
                    require(metric in per_seed_windows[seed][candidate], f"{comparison_id}: missing candidate window metric {metric}")
                    require(metric in per_seed_windows[seed][reference], f"{comparison_id}: missing reference window metric {metric}")
                seed_diffs = [
                    per_seed_windows[seed][candidate][metric] - per_seed_windows[seed][reference][metric]
                    for seed in sorted(per_seed_windows)
                ]
            else:
                seed_diffs = []
                for seed in spec.get("included_seeds", []):
                    ref = spec["seed_references"].get(str(seed))
                    require(ref is not None, f"{comparison_id}: missing selected member for seed {seed}")
                    require(seed in per_seed_windows, f"{comparison_id}: unknown included seed {seed}")
                    require(metric in per_seed_windows[seed][candidate], f"{comparison_id}: missing candidate window metric {metric}")
                    require(metric in per_seed_windows[seed][ref], f"{comparison_id}: missing reference window metric {metric}")
                    seed_diffs.append(
                        per_seed_windows[seed][candidate][metric] - per_seed_windows[seed][ref][metric]
                    )
            if len({len(row) for row in seed_diffs}) != 1:
                fail(f"{comparison_id}: seed window lengths differ")
            seed_mean = np.stack(seed_diffs, axis=0).mean(axis=0)
            array_key = f"{group_key}__{comparison_id}__seedmean_window_delta__{metric}"
            saved = get_array(archive, array_key)
            close(saved, seed_mean, f"bootstrap input {array_key}", atol=1e-9)
            block = min(int(spec["block_size_windows"]), len(seed_mean))
            blocks_needed = int(np.ceil(len(seed_mean) / block))
            rng = np.random.default_rng(0)
            starts = rng.integers(
                0, len(seed_mean) - block + 1,
                size=(int(spec["repeats"]), blocks_needed),
            )
            indices = (
                starts[:, :, None] + np.arange(block, dtype=np.int64)[None, None, :]
            ).reshape(int(spec["repeats"]), -1)[:, :len(seed_mean)]
            means = seed_mean[indices].mean(axis=1)
            expected = [float(seed_mean.mean()), *np.quantile(means, [0.025, 0.975]).tolist()]
            reported = interval
            close(reported["mean"], expected[0], f"bootstrap {comparison_id}.{metric}.mean")
            close(reported["ci95"], expected[1:], f"bootstrap {comparison_id}.{metric}.ci95")
            count += 1
    return count


def audit_seed(group_key: str, seed: int, run: dict[str, Any], archive: Any, atol: float) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    scales = [int(x) for x in get_array(archive, f"{group_key}__scales")]
    epsilon = get_array(archive, f"{group_key}__epsilon").astype(np.float64)
    std = get_array(archive, f"{group_key}__normalization_std").astype(np.float64)
    validation_history = get_array(archive, npz_key(group_key, seed, "validation", "split", "history"))
    validation_truth = get_array(archive, npz_key(group_key, seed, "validation", "split", "truth"))
    test_history = get_array(archive, npz_key(group_key, seed, "test", "split", "history"))
    test_truth = get_array(archive, npz_key(group_key, seed, "test", "split", "truth"))
    grid = run["validation_grids"]
    selection = run["validation_selection"]
    measured_val: dict[str, dict[str, Any]] = {}
    val_paths: dict[str, np.ndarray] = {}
    checks = 0

    # Independently rescore the baseline.
    base_path = path_for(archive, group_key, seed, "validation", "base")
    base_metric, _ = score(validation_history, validation_truth, base_path, scales, epsilon, std)
    check_reported_metrics(run["validation_metrics"]["base"], base_metric, f"{group_key}/seed{seed}/val/base")
    measured_val["base"] = base_metric
    val_paths["base"] = base_path
    checks += 1

    prior_counts = get_array(archive, f"{group_key}__train_prior_class_counts").astype(np.float64)
    prior_prob = get_array(archive, f"{group_key}__prior_probabilities").astype(np.float64)
    flat_prob = get_array(archive, f"{group_key}__flat_probabilities").astype(np.float64)
    require(prior_counts.shape == prior_prob.shape == flat_prob.shape, f"{group_key}: prior array shapes differ")
    require(np.all(prior_counts.sum(axis=-1) > 0), f"{group_key}: empty prior class histogram")
    close(prior_prob, prior_counts / prior_counts.sum(axis=-1, keepdims=True), f"{group_key}: prior frequency probabilities")
    flat_expected = np.zeros_like(flat_prob)
    flat_expected[..., 1] = 1.0
    close(flat_prob, flat_expected, f"{group_key}: flat soft-projection probabilities")
    checks += 2

    atom_selected: dict[str, str] = {}
    for atom in ("learned_multi", "learned_single", "prior", "flat"):
        rows = grid["atom_alpha_grids"][atom]
        require([float(r["alpha"]) for r in rows] == ALPHA_GRID, f"{group_key}/seed{seed}/{atom}: alpha grid mismatch")
        replay_rows = []
        for row in rows:
            path, metric, _ = verify_key_and_metrics(
                archive, row, group_key, seed, "validation", validation_history,
                validation_truth, scales, epsilon, std, f"{group_key}/seed{seed}/val/{row['candidate_id']}"
            )
            val_paths[row["candidate_id"]] = path
            measured_val[row["candidate_id"]] = metric
            require(row["eligible"] == eligible(metric, base_metric), f"{atom}/{row['candidate_id']}: wrong gate")
            replay_rows.append({**row, "metrics": metric})
            checks += 1
            if float(row["alpha"]) == 0.0:
                require(np.array_equal(path, base_path), f"{atom}: alpha=0 is not an exact base no-op")
                require(bool(row["is_no_op"]), f"{atom}: alpha=0 no-op flag is false")
                require(not row["eligible"], f"{atom}: alpha=0 was counted as a gain")
        chosen = best(replay_rows, base_metric, [r["candidate_id"] for r in replay_rows])
        if chosen is None:
            chosen = next(r for r in rows if float(r["alpha"]) == 0.0)
        picked = selection["atom_selection"][atom]
        require(picked["selected_candidate_id"] == chosen["candidate_id"], f"{atom}: selected alpha differs from strict validation replay")
        close(picked["selected_alpha"], chosen["alpha"], f"{atom}: selected alpha")
        atom_selected[atom] = chosen["candidate_id"]
        val_paths[atom] = val_paths[chosen["candidate_id"]]
        measured_val[atom] = measured_val[chosen["candidate_id"]]
        check_reported_metrics(
            run["validation_metrics"]["selected_atoms"][atom], measured_val[atom],
            f"{group_key}/seed{seed}/selected_atom/{atom}",
        )

    # Learned selector: exact ties prefer single-scale.
    learned_rows = grid["learned_selected_candidates"]
    learned_rows_by_id = {r["candidate_id"]: r for r in learned_rows}
    learned_order = sorted(
        ("learned_single", "learned_multi"),
        key=lambda name: (
            -measured_val[name]["raw_sign_accuracy"],
            measured_val[name]["standardized_mse"],
            0 if name == "learned_single" else 1,
        ),
    )
    learned_name = learned_order[0]
    learned_source = atom_selected[learned_name]
    require(selection["learned_selected"]["selected_member"] == learned_name, f"learned_selected member mismatch")
    require(selection["learned_selected"]["source_candidate_id"] == learned_source, f"learned_selected source mismatch")
    val_paths["learned_selected"] = val_paths[learned_source]
    measured_val["learned_selected"] = measured_val[learned_source]
    check_reported_metrics(
        run["validation_metrics"]["learned_selected"], measured_val["learned_selected"],
        f"{group_key}/seed{seed}/validation_summary/learned_selected",
    )
    learned_alias_key = selection["learned_selected"]["prediction_key"]
    learned_alias = get_array(archive, learned_alias_key)
    close(learned_alias, val_paths["learned_selected"], f"{group_key}/seed{seed}: learned_selected validation alias")
    check_reported_metrics(
        selection["learned_selected"]["validation_metrics"], measured_val["learned_selected"],
        f"{group_key}/seed{seed}/val/learned_selected",
    )
    checks += 1

    # Standalone selector uses only strict joint-gate candidates.
    standalone_rows = grid["standalone_candidates"]
    standalone_replay_rows = []
    for row in standalone_rows:
        source = row["source_candidate_id"]
        path = path_for(archive, group_key, seed, "validation", source)
        metric, _ = score(validation_history, validation_truth, path, scales, epsilon, std)
        check_reported_metrics(row["metrics"], metric, f"{group_key}/seed{seed}/val/standalone/{row['candidate_id']}")
        require(row["eligible"] == eligible(metric, base_metric), f"standalone {row['candidate_id']}: wrong gate")
        val_paths[row["candidate_id"]] = path
        measured_val[row["candidate_id"]] = metric
        standalone_replay_rows.append({**row, "metrics": metric})
        checks += 1
    standalone_choice = best(standalone_replay_rows, base_metric, STANDALONE_ORDER)
    if standalone_choice is None:
        standalone_choice = next(r for r in standalone_replay_rows if r["candidate_id"] == "base")
    standalone_source = standalone_choice["source_candidate_id"]
    require(selection["standalone_selected"]["selected_member"] == standalone_choice["candidate_id"], "standalone member mismatch")
    require(selection["standalone_selected"]["source_candidate_id"] == standalone_source, "standalone source mismatch")
    val_paths["standalone_selected"] = path_for(archive, group_key, seed, "validation", standalone_source)
    measured_val["standalone_selected"], _ = score(
        validation_history, validation_truth, val_paths["standalone_selected"], scales, epsilon, std
    )
    check_reported_metrics(
        run["validation_metrics"]["standalone_selected"], measured_val["standalone_selected"],
        f"{group_key}/seed{seed}/validation_summary/standalone_selected",
    )
    standalone_alias = get_array(archive, selection["standalone_selected"]["prediction_key"])
    close(standalone_alias, val_paths["standalone_selected"], f"{group_key}/seed{seed}: standalone_selected validation alias")
    checks += 1

    # Check all six weight grids and replay each pair's validation choice.
    selected_pair_source: dict[str, str] = {}
    for pair_name, (member_a, member_b) in PAIR_MEMBERS.items():
        rows = grid["pairwise_weight_grids"][pair_name]
        require([float(r["weight"]) for r in rows] == WEIGHT_GRID, f"{pair_name}: weight grid mismatch")
        a, b = val_paths[member_a], val_paths[member_b]
        replay_rows = []
        for row in rows:
            weight = float(row["weight"])
            candidate = (1.0 - weight) * a + weight * b
            saved = get_array(archive, row["prediction_key"])
            close(saved, candidate, f"{pair_name}/{row['candidate_id']} path formula")
            metric, _ = score(validation_history, validation_truth, saved, scales, epsilon, std)
            check_reported_metrics(row["metrics"], metric, f"{group_key}/seed{seed}/val/{row['candidate_id']}")
            require(row["eligible"] == eligible(metric, base_metric), f"{row['candidate_id']}: wrong gate")
            require(bool(row["is_nontrivial_blend"]) == bool(0 < weight < 1 and not np.array_equal(a, b)),
                    f"{row['candidate_id']}: wrong nontrivial-blend identity")
            learned_path_differs = not np.array_equal(val_paths["learned_selected"], val_paths["base"])
            learned_active = float(selection["learned_selected"]["selected_alpha"]) > 0 and learned_path_differs
            includes_learned = bool(
                learned_active and (
                    (member_a == "learned_selected" and (1.0 - weight) > 0)
                    or (member_b == "learned_selected" and weight > 0)
                )
            )
            require(bool(row.get("includes_actual_learned_component", False)) == includes_learned,
                    f"{row['candidate_id']}: wrong actual learned-component flag")
            if "is_true_learned_fusion" in row:
                require(bool(row["is_true_learned_fusion"]) == bool(
                    0 < weight < 1 and not np.array_equal(a, b) and includes_learned
                ), f"{row['candidate_id']}: wrong true-learned-fusion flag")
            val_paths[row["candidate_id"]] = saved
            measured_val[row["candidate_id"]] = metric
            replay_rows.append({**row, "metrics": metric})
            checks += 1
        chosen = best(replay_rows, base_metric, [r["candidate_id"] for r in replay_rows], tie_field="tie_value")
        item = selection["pairwise_fusions"][pair_name]
        if chosen is None:
            require(item["fallback_to_base"] is True and item["selected_candidate_id"] == "base", f"{pair_name}: missing base fallback")
            selected_pair_source[pair_name] = "base"
            val_paths[pair_name] = val_paths["base"]
        else:
            require(item["fallback_to_base"] is False, f"{pair_name}: spurious fallback")
            require(item["selected_candidate_id"] == chosen["candidate_id"], f"{pair_name}: selected weight differs from replay")
            close(item["selected_weight"], chosen["weight"], f"{pair_name}: selected weight")
            selected_pair_source[pair_name] = chosen["candidate_id"]
            val_paths[pair_name] = val_paths[chosen["candidate_id"]]
        row_source = selected_pair_source[pair_name]
        require(item["member_a"] == member_a and item["member_b"] == member_b, f"{pair_name}: member order mismatch")
        if row_source != "base":
            chosen_row = next(r for r in replay_rows if r["candidate_id"] == row_source)
            require(item["is_endpoint"] == (chosen_row["weight"] in (0.0, 1.0)), f"{pair_name}: endpoint flag mismatch")
            require(item["members_different_path"] == (not np.array_equal(a, b)), f"{pair_name}: distinct-member flag mismatch")
            expected_selected_nontrivial = bool(
                0.0 < float(chosen_row["weight"]) < 1.0 and not np.array_equal(a, b)
            )
            require(chosen_row["is_nontrivial_blend"] == expected_selected_nontrivial,
                    f"{pair_name}: selected nontrivial identity mismatch")
            require(item["includes_actual_learned_component"] == chosen_row["includes_actual_learned_component"], f"{pair_name}: learned-component identity mismatch")
            selected_weight = float(chosen_row["weight"])
            selected_learned_active = bool(
                learned_active and (
                    (member_a == "learned_selected" and (1.0 - selected_weight) > 0)
                    or (member_b == "learned_selected" and selected_weight > 0)
                )
            )
            expected_true_learned = bool(
                0 < selected_weight < 1 and not np.array_equal(a, b) and selected_learned_active
            )
            require(item["is_true_learned_fusion"] == expected_true_learned,
                    f"{pair_name}: true learned-fusion identity mismatch")
            chosen_metric = chosen_row["metrics"]
        else:
            require(item["is_endpoint"] is False, f"{pair_name}: fallback marked endpoint")
            require(item["is_true_learned_fusion"] is False, f"{pair_name}: fallback marked true learned fusion")
            chosen_metric = base_metric
        check_reported_metrics(item["validation_metrics"], chosen_metric, f"{group_key}/seed{seed}/selected/{pair_name}")
        measured_val[pair_name], _ = score(validation_history, validation_truth, val_paths[pair_name], scales, epsilon, std)

    # Fixed three-way average and fallback identity.
    triple_id = "fixedmean__learned_selected_flat_difference"
    triple_path = (val_paths["learned_selected"] + val_paths["flat"] + val_paths["difference"]) / 3.0
    triple_saved = path_for(archive, group_key, seed, "validation", triple_id)
    close(triple_saved, triple_path, f"{group_key}/seed{seed}/fixedmean path formula")
    triple_metric, _ = score(validation_history, validation_truth, triple_saved, scales, epsilon, std)
    triple_row = grid["fixed_threeway_mean"]
    check_reported_metrics(triple_row["metrics"], triple_metric, f"{group_key}/seed{seed}/val/fixedmean")
    require(triple_row["eligible"] == eligible(triple_metric, base_metric), "fixedmean gate mismatch")
    triple_paths = [val_paths[name] for name in ("learned_selected", "flat", "difference")]
    triple_all_distinct = all(
        not np.array_equal(triple_paths[i], triple_paths[j])
        for i in range(len(triple_paths)) for j in range(i + 1, len(triple_paths))
    )
    triple_nontrivial = len({path.tobytes() for path in triple_paths}) > 1
    learned_active = (
        float(selection["learned_selected"]["selected_alpha"]) > 0
        and not np.array_equal(val_paths["learned_selected"], val_paths["base"])
    )
    require(bool(triple_row["members_pairwise_different"]) == triple_all_distinct,
            "fixedmean pairwise-distinct flag mismatch")
    require(bool(triple_row["includes_actual_learned_component"]) == learned_active,
            "fixedmean actual-learned flag mismatch")
    if "is_nontrivial_blend" in triple_row:
        require(bool(triple_row["is_nontrivial_blend"]) == triple_nontrivial,
                "fixedmean nontrivial-blend flag mismatch")
    val_paths["threeway_mean"] = triple_saved if triple_row["eligible"] else val_paths["base"]
    measured_val["threeway_mean"], _ = score(validation_history, validation_truth, val_paths["threeway_mean"], scales, epsilon, std)
    checks += 1

    # Overall fusion selector, with protocol-fixed tie order.
    fusion_rows = grid["fusion_selected_candidates"]
    fusion_replay_rows = []
    for row in fusion_rows:
        name = row["candidate_id"]
        source = row["source_candidate_id"]
        expected_metric = measured_val[name] if name in measured_val else measured_val[source]
        check_reported_metrics(row["metrics"], expected_metric, f"fusion candidate {name}")
        require(row["eligible"] == eligible(expected_metric, base_metric), f"fusion candidate {name}: wrong gate")
        if name in PAIR_MEMBERS:
            expected_nontrivial = selection["pairwise_fusions"][name]["is_true_learned_fusion"] or (
                selection["pairwise_fusions"][name]["selected_weight"] is not None
                and 0 < selection["pairwise_fusions"][name]["selected_weight"] < 1
                and selection["pairwise_fusions"][name]["members_different_path"]
            )
            expected_learned = selection["pairwise_fusions"][name]["includes_actual_learned_component"]
        else:
            expected_nontrivial = bool(triple_row["eligible"] and triple_nontrivial)
            expected_learned = bool(triple_row["eligible"] and learned_active)
        require(bool(row["is_nontrivial_blend"]) == bool(expected_nontrivial),
                f"fusion candidate {name}: nontrivial identity mismatch")
        require(bool(row["includes_actual_learned_component"]) == bool(expected_learned),
                f"fusion candidate {name}: actual learned identity mismatch")
        source_path = path_for(archive, group_key, seed, "validation", source)
        source_key = npz_key(group_key, seed, "validation", source, "path")
        require(row["prediction_key"] == source_key, f"fusion candidate {name}: prediction key/source mismatch")
        close(get_array(archive, row["prediction_key"]), source_path,
              f"fusion candidate {name}: declared source path")
        fusion_replay_rows.append({**row, "metrics": expected_metric})
    fusion_choice = best(fusion_replay_rows, base_metric, FUSION_ORDER)
    fusion_sel = selection["fusion_selected"]
    if fusion_choice is None:
        require(fusion_sel["fallback_to_base"] is True and fusion_sel["source_candidate_id"] == "base", "fusion selector missing base fallback")
        fusion_source = "base"
    else:
        require(fusion_sel["fallback_to_base"] is False, "fusion selector spurious fallback")
        require(fusion_sel["selected_scheme"] == fusion_choice["candidate_id"], "fusion scheme differs from validation replay")
        require(fusion_sel["source_candidate_id"] == fusion_choice["source_candidate_id"], "fusion source differs from validation replay")
        fusion_source = fusion_choice["source_candidate_id"]
    require(bool(fusion_sel["includes_actual_learned_component"]) == bool(
        fusion_choice and fusion_choice.get("includes_actual_learned_component", False)
    ), "fusion_selected learned-component identity mismatch")
    require(bool(fusion_sel["is_nontrivial_blend"]) == bool(
        fusion_choice and fusion_choice.get("is_nontrivial_blend", False)
    ), "fusion_selected nontrivial identity mismatch")
    val_paths["fusion_selected"] = val_paths["base"] if fusion_source == "base" else (
        triple_saved if fusion_source == triple_id else val_paths[fusion_source]
    )
    fusion_alias = get_array(archive, selection["fusion_selected"]["prediction_key"])
    close(fusion_alias, val_paths["fusion_selected"], f"{group_key}/seed{seed}: fusion_selected validation alias")

    # Test paths and reported test rows must match the frozen identities/formulas.
    selected_ids = run["test_selected_identity"]["selected_test_candidate_ids"]
    measured_test_windows: dict[str, dict[str, np.ndarray]] = {}
    test_paths: dict[str, np.ndarray] = {}
    for method, row in run["test_metrics"].items():
        path, metric, window = verify_key_and_metrics(
            archive, row, group_key, seed, "test", test_history, test_truth,
            scales, epsilon, std, f"{group_key}/seed{seed}/test/{method}"
        )
        test_paths[method] = path
        measured_test_windows[method] = window
        require(selected_ids[method] == row["candidate_id"], f"{method}: JSON selected test ID mismatch")
        checks += 1

    test_base_metric = run["test_metrics"]["base"]["metrics"]
    for method, row in run["test_metrics"].items():
        metrics = row["metrics"]
        require(metrics["strict_joint_pass"] == eligible(metrics, test_base_metric),
                f"{method}: post-test joint-pass flag mismatch")
        check_delta_report(metrics["relative_to_base"], measured_test_windows[method],
                           measured_test_windows["base"], f"{method}.relative_to_base")
        check_delta_report(metrics["relative_to_validation_selected_standalone"],
                           measured_test_windows[method], measured_test_windows["standalone_selected"],
                           f"{method}.relative_to_standalone")

    for atom in ("learned_multi", "learned_single", "prior", "flat"):
        require(selected_ids[atom] == atom_selected[atom], f"{atom}: test candidate ID is not validation-selected alpha")
    for method in ("base", "difference", "residual", "residual_direction"):
        require(selected_ids[method] == method, f"{method}: test identity changed")
    require(selected_ids["learned_selected"] == learned_source, "learned_selected test ID mismatch")
    require(selected_ids["standalone_selected"] == standalone_source, "standalone_selected test ID mismatch")

    require(np.array_equal(test_paths["learned_selected"], test_paths[learned_name]),
            "learned_selected test path differs from its selected learned member")
    standalone_member = selection["standalone_selected"]["selected_member"]
    require(np.array_equal(test_paths["standalone_selected"], test_paths[standalone_member]),
            "standalone_selected test path differs from its selected standalone member")
    # Validate each pre-selected fusion path against its frozen members and weight.
    for pair_name, (member_a, member_b) in PAIR_MEMBERS.items():
        item = selection["pairwise_fusions"][pair_name]
        if item["fallback_to_base"]:
            expected_path = test_paths["base"]
            expected_id = "base"
        else:
            weight = float(item["selected_weight"])
            expected_path = (1.0 - weight) * test_paths[member_a] + weight * test_paths[member_b]
            expected_id = item["selected_candidate_id"]
        close(test_paths[pair_name], expected_path, f"{pair_name}: test member-mixture identity")
        require(selected_ids[pair_name] == expected_id, f"{pair_name}: selected test candidate ID mismatch")

    expected_triple = (
        (test_paths["learned_selected"] + test_paths["flat"] + test_paths["difference"]) / 3.0
        if triple_row["eligible"] else test_paths["base"]
    )
    close(test_paths["threeway_mean"], expected_triple, "threeway test identity/fallback")
    expected_triple_id = triple_id if triple_row["eligible"] else "base"
    require(selected_ids["threeway_mean"] == expected_triple_id, "threeway test candidate ID mismatch")
    expected_fusion = test_paths["base"] if fusion_source == "base" else (
        test_paths["threeway_mean"] if fusion_source == triple_id else test_paths[
            next(name for name, source in selected_pair_source.items() if source == fusion_source)
        ]
    )
    close(test_paths["fusion_selected"], expected_fusion, "fusion_selected test source identity")
    require(run["test_selected_identity"]["fusion_selected_candidate_id"] == fusion_source,
            "fusion_selected identity does not trace to validation")

    # The exported member deltas must match the selected test path and its actual members.
    member_reports = run["test_comparisons_to_fusion_members"]
    for pair_name, (member_a, member_b) in PAIR_MEMBERS.items():
        spec = member_reports[pair_name]
        for label, member in (("relative_to_member_a", member_a), ("relative_to_member_b", member_b)):
            delta = spec[label]
            check_delta_report(delta, measured_test_windows[pair_name],
                               measured_test_windows[member], f"{pair_name}.{label}")
        expected_interior = bool(
            not selection["pairwise_fusions"][pair_name]["fallback_to_base"]
            and selection["pairwise_fusions"][pair_name]["selected_weight"] not in (0.0, 1.0)
            and selection["pairwise_fusions"][pair_name]["members_different_path"]
        )
        require(spec["is_interior_nonidentical_blend"] == expected_interior,
                f"{pair_name}: test nontrivial identity mismatch")
        expected_learned = bool(
            expected_interior
            and selection["pairwise_fusions"][pair_name]["includes_actual_learned_component"]
        )
        require(spec["is_true_learned_fusion"] == expected_learned,
                f"{pair_name}: test true-learned-fusion identity mismatch")
    triple_report = member_reports["threeway_mean"]
    for member in ("learned_selected", "flat", "difference"):
        check_delta_report(triple_report["relative_to_each_member"][member],
                           measured_test_windows["threeway_mean"], measured_test_windows[member],
                           f"threeway_mean.relative_to_{member}")

    # Every JSON prediction key, including all validation and test candidates, must exist.
    def walk(value: Any) -> None:
        nonlocal checks
        if isinstance(value, dict):
            key = value.get("prediction_key")
            if isinstance(key, str):
                require(key in archive, f"JSON prediction_key is absent from NPZ: {key}")
                checks += 1
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(run)
    return measured_test_windows, checks


def audit_root(root: Path, expected_cohort: str | None, atol: float) -> dict[str, Any]:
    manifest_path = root / "output_manifest.json"
    results_path = root / "results.json"
    predictions_path = root / "predictions.npz"
    for path in (manifest_path, results_path, predictions_path):
        require(path.is_file(), f"missing output artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    require(manifest.get("complete") is True, "output manifest is not complete")
    require(results.get("schema_version") == 1, "unsupported results schema")
    cohort = manifest.get("cohort")
    if expected_cohort:
        require(cohort == expected_cohort, f"expected cohort {expected_cohort}, got {cohort}")
    require(results.get("cohort") == cohort, "manifest/results cohort mismatch")
    groups = results.get("groups", {})
    require(set(groups) == set(manifest.get("completed_groups", [])), "completed groups differ from results")
    require(set(groups), "no groups in output")
    evaluator_snapshot = Path(manifest.get("script_snapshot", root / "evaluate_trend_fusions.py"))
    require(evaluator_snapshot.is_file(), f"missing evaluator snapshot: {evaluator_snapshot}")
    evaluator_sha = sha256_file(evaluator_snapshot)
    require(evaluator_sha == manifest.get("script_sha256"), "evaluator snapshot hash differs from output manifest")
    input_hashes = {
        "output_manifest_sha256": sha256_file(manifest_path),
        "results_sha256": sha256_file(results_path),
        "predictions_sha256": sha256_file(predictions_path),
        "evaluator_snapshot_sha256": evaluator_sha,
    }
    checked = 0
    summary = []
    with np.load(predictions_path, allow_pickle=False) as archive:
        for group_key, group in groups.items():
            require(group["group_key"] == group_key, f"group key mismatch: {group_key}")
            require(group["cohort"] == cohort, f"{group_key}: cohort mismatch")
            require(group.get("protocol", {}).get("test_selection", "").startswith("all alpha"),
                    f"{group_key}: missing validation-first protocol declaration")
            require(set(group["seeds"]) == {"0", "1", "2"}, f"{group_key}: expected seeds 0,1,2")
            source_cell = Path(group["cell_dir"])
            source_config_path = source_cell / "config.json"
            source_results_path = source_cell / "results.json"
            require(source_config_path.is_file() and source_results_path.is_file(),
                    f"{group_key}: source cell metadata is missing")
            source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
            source_results = json.loads(source_results_path.read_text(encoding="utf-8"))
            require(source_config.get("smoke", False) is False, f"{group_key}: source config is smoke")
            require(source_config.get("data", {}).get("smoke", False) is False,
                    f"{group_key}: source data config is smoke")
            require(source_results.get("smoke", False) is False, f"{group_key}: source results are smoke")
            require(source_results.get("checkpoint_reload_verified") is True,
                    f"{group_key}: source checkpoint reload is not verified")
            source_files = {
                "config_sha256": source_cell / "config.json",
                "checkpoint_sha256": source_cell / "checkpoint.pt",
                "source_prediction_npz_sha256": source_cell / "predictions.npz",
            }
            for hash_name, path in source_files.items():
                require(path.is_file(), f"{group_key}: missing source artifact {path}")
                require(sha256_file(path) == group["source"][hash_name],
                        f"{group_key}: source artifact hash mismatch: {hash_name}")
            data_path = Path(group["group_data"]["data_path"])
            require(data_path.is_file(), f"{group_key}: source data file is missing: {data_path}")
            require(sha256_file(data_path) == group["source"]["data_sha256"],
                    f"{group_key}: source data hash mismatch")
            require(group["source"]["script_sha256"] == evaluator_sha,
                    f"{group_key}: per-group evaluator hash differs from snapshot")
            test_windows_by_seed: dict[int, dict[str, dict[str, np.ndarray]]] = {}
            for seed_key, run in group["seeds"].items():
                seed = int(seed_key)
                windows, n_checks = audit_seed(group_key, seed, run, archive, atol)
                test_windows_by_seed[seed] = windows
                checked += n_checks
            # Recompute group summaries from the verified per-seed test metrics.
            seed_ids = sorted(int(s) for s in group["seeds"])
            for method, summary_row in group["test_summary_mean_sample_std"].items():
                method_rows = [group["seeds"][str(seed)]["test_metrics"][method]["metrics"] for seed in seed_ids]
                for key in ("raw_sign_accuracy", "standardized_mse", "original_units_mse", "mean_3class_accuracy"):
                    values = np.asarray([row[key] for row in method_rows], dtype=np.float64)
                    close(summary_row[key]["mean"], values.mean(), f"{group_key}/{method}/{key} summary mean")
                    close(summary_row[key]["sample_std"], values.std(ddof=1), f"{group_key}/{method}/{key} summary std")
                    close(summary_row[key]["per_seed"], values, f"{group_key}/{method}/{key} summary per-seed")
                for scale in summary_row["accuracy_by_scale"]:
                    values = np.asarray([row["accuracy_by_scale"][scale] for row in method_rows], dtype=np.float64)
                    close(summary_row["accuracy_by_scale"][scale]["mean"], values.mean(),
                          f"{group_key}/{method}/accuracy_scale_{scale} summary mean")
                    close(summary_row["accuracy_by_scale"][scale]["sample_std"], values.std(ddof=1),
                          f"{group_key}/{method}/accuracy_scale_{scale} summary std")
                    checked += 2
                checked += 12
            bootstrap_checks = check_bootstrap(group_key, group, archive, test_windows_by_seed)
            checked += bootstrap_checks
            summary.append({"group": group_key, "seeds": 3, "bootstrap_metric_intervals_checked": bootstrap_checks})
    return {
        "audit": "passed", "cohort": cohort, "groups": summary,
        "assertions_checked": checked, "input_hashes": input_hashes,
        "prior_audit_scope": "saved train-label counts were checked against saved frequency probabilities; train labels were not independently rebuilt from source CSV",
        "bootstrap_interpretation": "paired windows are resampled after averaging fitted seeds; intervals condition on the observed dataset and trained seeds, which are not treated as independent datasets",
    }


def main() -> int:
    global ATOL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cohort", choices=("exploratory", "additional_validation"))
    parser.add_argument("--atol", type=float, default=ATOL)
    parser.add_argument("--report", type=Path, help="also save the successful audit JSON here")
    parser.add_argument("--snapshot-to", type=Path, help="save an exact audit-script copy here")
    args = parser.parse_args()
    ATOL = float(args.atol)
    result = audit_root(args.output_dir.expanduser().resolve(), args.cohort, ATOL)
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    result["audit_script_sha256"] = script_sha
    result["audit_script"] = str(script_path)
    if args.snapshot_to:
        snapshot = args.snapshot_to.expanduser().resolve()
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(script_path.read_bytes())
        require(sha256_file(snapshot) == script_sha, "saved audit snapshot hash mismatch")
        result["audit_script_snapshot"] = str(snapshot)
    if args.report:
        report = args.report.expanduser().resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
