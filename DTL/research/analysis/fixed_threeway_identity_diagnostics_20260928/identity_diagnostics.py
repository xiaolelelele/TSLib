#!/usr/bin/env python3
"""Descriptive identity checks for the frozen fixed-threeway fresh evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = REPO / "DTL/trend_lab_runs/fixed_threeway_additional_validation_electricity_traffic_20260928T102453Z"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def path_key(seed, split, method):
    return "seed{}__{}__{}__path".format(seed, split, method)


def unique_count(paths):
    unique = []
    for path in paths:
        if not any(np.array_equal(path, prior) for prior in unique):
            unique.append(path)
    return len(unique)


def exact_equal(npz, seed, split, left, right):
    return bool(np.array_equal(npz[path_key(seed, split, left)], npz[path_key(seed, split, right)]))


def test_truth_adjacency(npz, seed):
    history = npz["seed{}__test__split__history".format(seed)]
    truth = npz["seed{}__test__split__truth".format(seed)]
    if history.ndim != 3 or truth.ndim != 3 or history.shape[0] != truth.shape[0]:
        raise ValueError("unexpected test history/truth shapes: {} / {}".format(history.shape, truth.shape))
    if history.shape[2] != truth.shape[2] or truth.shape[1] < 1:
        raise ValueError("incompatible test history/truth channel or horizon dimensions")
    first = truth[:, :1, :] == history[:, -1:, :]
    later = truth[:, 1:, :] == truth[:, :-1, :]
    equal_count = int(first.sum() + later.sum())
    transition_count = int(truth.shape[0] * truth.shape[1] * truth.shape[2])
    return {
        "test_windows": int(truth.shape[0]),
        "forecast_horizon": int(truth.shape[1]),
        "channels": int(truth.shape[2]),
        "adjacent_value_comparisons": transition_count,
        "exact_zero_changes": equal_count,
        "exact_zero_change_fraction": equal_count / transition_count,
        "first_step_includes_last_history_value": True,
        "history_dtype": str(history.dtype),
        "truth_dtype": str(truth.dtype),
    }


def metric_comparison(seed_rows, candidate_name, reference_name):
    per_seed = []
    cand_mse, ref_mse = [], []
    direction_delta, mse_delta, joint = [], [], []
    for row in seed_rows:
        methods = row["seed_record"]["test"]["methods"]
        candidate = methods[candidate_name]["metrics"]
        reference = methods[reference_name]["metrics"]
        sign_delta = candidate["raw_sign_accuracy"] - reference["raw_sign_accuracy"]
        candidate_mse = candidate["standardized_mse"]
        reference_mse = reference["standardized_mse"]
        mse_diff = candidate_mse - reference_mse
        is_joint = sign_delta > 0 and candidate_mse <= reference_mse + 1e-8
        per_seed.append({
            "seed": row["seed"],
            "candidate_raw_sign_accuracy": candidate["raw_sign_accuracy"],
            "reference_raw_sign_accuracy": reference["raw_sign_accuracy"],
            "raw_sign_delta_percentage_points": sign_delta * 100.0,
            "candidate_standardized_mse": candidate_mse,
            "reference_standardized_mse": reference_mse,
            "standardized_mse_delta": mse_diff,
            "joint_raw_sign_up_and_mse_nonincrease": bool(is_joint),
        })
        cand_mse.append(candidate_mse)
        ref_mse.append(reference_mse)
        direction_delta.append(sign_delta * 100.0)
        mse_delta.append(mse_diff)
        joint.append(bool(is_joint))
    mean_candidate_mse = float(np.mean(cand_mse))
    mean_reference_mse = float(np.mean(ref_mse))
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "direction_delta_percentage_points_seed_mean": float(np.mean(direction_delta)),
        "standardized_mse_candidate_seed_mean": mean_candidate_mse,
        "standardized_mse_reference_seed_mean": mean_reference_mse,
        "standardized_mse_percent_change_of_seed_means": (
            (mean_candidate_mse / mean_reference_mse - 1.0) * 100.0
        ),
        "mean_seed_standardized_mse_delta": float(np.mean(mse_delta)),
        "joint_seed_count": int(sum(joint)),
        "seed_count": len(seed_rows),
        "joint_rule": "candidate raw_sign_accuracy > reference and candidate standardized_mse <= reference + 1e-8",
        "per_seed": per_seed,
    }


def run(input_root, output_json):
    input_root = Path(input_root).resolve()
    output_json = Path(output_json).resolve()
    audit_path = input_root / "independent_audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed"):
        raise RuntimeError("fresh full-root independent audit is not passed")

    cells = []
    for result_path in sorted((input_root / "cell_artifacts").glob("*/results.json")):
        cell_dir = result_path.parent
        npz_path = cell_dir / "predictions.npz"
        result = json.loads(result_path.read_text())
        group = result["group_key"]
        if result.get("cohort") != "additional_validation":
            raise ValueError("unexpected cohort in {}".format(group))
        seed_rows = []
        identity_rows = []
        with np.load(npz_path, allow_pickle=False) as npz:
            for seed_key, seed_record in sorted(result["seeds"].items(), key=lambda pair: int(pair[0])):
                seed = int(seed_key)
                split_results = {}
                for split in ("validation", "test"):
                    lfd_paths = [
                        npz[path_key(seed, split, name)]
                        for name in ("learned_selected", "flat", "difference")
                    ]
                    bfd_paths = [
                        npz[path_key(seed, split, name)]
                        for name in ("base", "flat", "difference")
                    ]
                    fd_equals_bd = {
                        "raw": exact_equal(npz, seed, split, "flat_difference_mean_raw", "base_difference_mean_raw"),
                        "deployed": exact_equal(npz, seed, split, "flat_difference_mean_deployed", "base_difference_mean_deployed"),
                    }
                    split_results[split] = {
                        "flat_equals_base": exact_equal(npz, seed, split, "flat", "base"),
                        "learned_selected_equals_base": exact_equal(npz, seed, split, "learned_selected", "base"),
                        "learned_threeway_raw_equals_deployed": exact_equal(
                            npz, seed, split, "learned_threeway_raw", "learned_threeway_deployed"
                        ),
                        "learned_threeway_member_unique_path_count": unique_count(lfd_paths),
                        "nohead_threeway_member_unique_path_count": unique_count(bfd_paths),
                        "flat_difference_equals_base_difference": fd_equals_bd,
                    }
                truth_adjacency = test_truth_adjacency(npz, seed)
                identity_rows.append({"seed": seed, "splits": split_results, "test_truth_adjacency": truth_adjacency})
                seed_rows.append({"seed": seed, "seed_record": seed_record})

        comparisons = [
            metric_comparison(seed_rows, "learned_multi", "learned_single"),
            metric_comparison(seed_rows, "learned_selected", "learned_threeway_raw"),
            metric_comparison(seed_rows, "learned_selected", "flat"),
        ]
        n_test_windows = [row["test_truth_adjacency"]["test_windows"] for row in identity_rows]
        if len(set(n_test_windows)) != 1:
            raise ValueError("test window counts differ across seeds in {}".format(group))
        cells.append({
            "group_key": group,
            "dataset": result["dataset"],
            "backbone": result["backbone"],
            "seeds": [row["seed"] for row in identity_rows],
            "identity_checks": identity_rows,
            "test_metric_comparisons": comparisons,
            "test_windows_per_seed": n_test_windows[0],
            "input_artifacts": {
                "results_json": str(result_path),
                "results_json_sha256": sha256(result_path),
                "predictions_npz": str(npz_path),
                "predictions_npz_sha256": sha256(npz_path),
            },
        })

    if len(cells) != 4:
        raise RuntimeError("expected four fresh cells, found {}".format(len(cells)))
    output = {
        "schema_version": "fixed_threeway_identity_diagnostics_v1",
        "status": "descriptive_only_no_selection_no_training_no_bootstrap",
        "input_root": str(input_root),
        "independent_audit": {
            "path": str(audit_path),
            "sha256": sha256(audit_path),
            "passed": True,
            "checks": audit.get("checks"),
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "definitions": {
            "path_identity": "exact numpy.array_equal on saved paths",
            "learned_threeway_members": ["learned_selected", "flat", "difference"],
            "nohead_threeway_members": ["base", "flat", "difference"],
            "FD_vs_BD": "flat_difference_mean vs base_difference_mean, both raw and deployed identities",
            "adjacent_truth_zero": "exact equality from history[-1] to truth[0], then consecutive truth steps, across every test window/channel",
            "direction_delta": "candidate minus reference raw_sign_accuracy, converted to percentage points and averaged across seeds",
            "MSE_percent": "100 * (mean candidate standardized MSE / mean reference standardized MSE - 1), averaging seeds first",
            "joint_seed_rule": "raw_sign_accuracy strictly increases and standardized MSE is at most reference + 1e-8",
        },
        "cells": cells,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(str(output_json))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=Path(__file__).resolve().parent / "results.json")
    args = parser.parse_args()
    run(args.input_root, args.output_json)


if __name__ == "__main__":
    main()
