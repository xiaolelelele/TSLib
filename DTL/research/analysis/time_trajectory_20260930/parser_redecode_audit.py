"""Locked NEM decisions under nearest decimal parsing, without re-selection.

Frozen train statistics, learned p/mu/b and validation-selected lambda remain
fixed. Rebuild the shared grid and re-decode posterior and mean-only controls.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def scoring(x, y, path):
    def classes(z):
        prev = np.concatenate((x[:, -1:, :], z[:, :-1]), axis=1)
        return np.where(z < prev, 0, np.where(z > prev, 2, 1))
    true, pred = classes(y), classes(path)
    return {"mse": float(np.mean((path-y)**2)), "raw_accuracy": float(np.mean(true == pred)),
            "predicted_exact_zero_count": int((pred == 1).sum()), "true_exact_zero_count": int((true == 1).sum())}, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--frozen-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frozen = Path(args.frozen_source)
    sys.path.insert(0, str(frozen))
    from trend_lab.trajectory_decision import make_common_grids, decode_trajectory
    root = Path(args.results) / "nem"
    meta = json.loads((root / "dataset.json").read_text())
    with open(meta["csv"], newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        raw = np.array([[float(v) for v in row[1:]] for row in reader], dtype=np.float64)
    normalized = (raw-np.array(meta["mean"]))/np.array(meta["std"])
    cases, checks, failures = [], [], []

    def check(name, value):
        checks.append(name)
        if not bool(value):
            failures.append(name)

    for prediction_file in sorted(root.glob("*_seed*/predictions.npz")):
        if prediction_file.parent.name.startswith("shared_") or not (prediction_file.parent / "result.json").exists():
            continue
        lock = json.loads((prediction_file.parent / "validation_lock.json").read_text())
        with np.load(prediction_file) as archive:
            x, y, origins = archive["test__x"], archive["test__y"], archive["test__origins"]
            b, mu, p = archive["test__cosdir"], archive["test__mean"], archive["test__probabilities"]
            L, H = x.shape[1], y.shape[1]
            new_x = np.stack([normalized[i-L+1:i+1] for i in origins])
            new_y = np.stack([normalized[i+1:i+H+1] for i in origins])
            check(str(prediction_file.parent) + " float32 x unchanged", np.array_equal(x.astype(np.float32), new_x.astype(np.float32)))
            check(str(prediction_file.parent) + " float32 y unchanged", np.array_equal(y.astype(np.float32), new_y.astype(np.float32)))
            arguments = dict(mode=meta["spec"]["grid_mode"], step=.1, padding=.25,
                             normalizer_mean=meta["mean"], normalizer_std=meta["std"],
                             native_step=1., native_origin=0., lower=None, max_states=257, coarsen=True)
            old_grids, old_grid_info = make_common_grids(x, b, mu, **arguments)
            new_grids, new_grid_info = make_common_grids(new_x, b, mu, **arguments)
            unequal_grids = sum(not np.array_equal(a, c) for ra, rc in zip(old_grids, new_grids) for a, c in zip(ra, rc))
            unequal_sizes = sum(len(a) != len(c) for ra, rc in zip(old_grids, new_grids) for a, c in zip(ra, rc))
            row = {"case": str(prediction_file.parent), "predictions_sha256": hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
                   "validation_lock_sha256": hashlib.sha256((prediction_file.parent / "validation_lock.json").read_bytes()).hexdigest(),
                   "grid_nonidentical_cells": unequal_grids, "grid_size_changed_cells": unequal_sizes,
                   "old_grid_metadata": old_grid_info, "new_grid_metadata": new_grid_info, "methods": {}}
            for name in ("posterior_selected", "mean_only_grid"):
                selected = lock["choices"]["posterior"] if name == "posterior_selected" else "mean_only_grid"
                strength = (float(selected.rsplit("_", 1)[1]) if name == "posterior_selected" and selected else 0.)
                if name == "posterior_selected" and selected is None:
                    old_path, new_path, old_diagnostic, new_diagnostic = b.copy(), b.copy(), {}, {}
                else:
                    old_path, old_diagnostic = decode_trajectory(x, b, mu, p, strength, old_grids)
                    new_path, new_diagnostic = decode_trajectory(new_x, b, mu, p, strength, new_grids)
                saved = archive["test__" + name]
                check(str(prediction_file.parent) + " replay saved " + name, np.array_equal(old_path, saved))
                old_score, old_class = scoring(x, y, old_path)
                new_score, new_class = scoring(new_x, new_y, new_path)
                changes = old_class != new_class
                row["methods"][name] = {
                    "locked_choice": selected, "strength": strength,
                    "old_score": old_score, "nearest_parser_redecoded_score": new_score,
                    "raw_accuracy_change_pp": 100*(new_score["raw_accuracy"]-old_score["raw_accuracy"]),
                    "mse_change": new_score["mse"]-old_score["mse"],
                    "path_nonidentical_elements": int((old_path != new_path).sum()),
                    "path_max_absolute_change": float(np.max(np.abs(old_path-new_path))),
                    "predicted_class_changed_elements": int(changes.sum()),
                    "changed_after_first_horizon": int(changes[:, 1:].sum()),
                    "selected_base_changed_samples": int(np.sum(np.asarray(old_diagnostic.get("selected_base", [])) != np.asarray(new_diagnostic.get("selected_base", []))))}
            cases.append(row)
            print("audited", prediction_file.parent.name, "grids", unequal_grids,
                  {k: (v["predicted_class_changed_elements"], v["raw_accuracy_change_pp"], v["mse_change"]) for k, v in row["methods"].items()}, flush=True)
    effects = [m for row in cases for m in row["methods"].values()]
    result = {"completed_NEM_cases_at_audit": len(cases), "checks": len(checks), "failures": failures,
              "check_names": checks, "max_redecoded_raw_accuracy_change_pp": max((abs(r["raw_accuracy_change_pp"]) for r in effects), default=0),
              "max_redecoded_mse_absolute_change": max((abs(r["mse_change"]) for r in effects), default=0),
              "max_redecoded_predicted_class_changed_elements": max((r["predicted_class_changed_elements"] for r in effects), default=0),
              "policy": "no training/reselection/lambda expansion; frozen mean/std, p,mu,b; rebuild grid under nearest-parse x, re-run locked posterior and mean_only_grid",
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [frozen / "trend_lab/trajectory_decision.py", Path(meta["csv"])]},
              "cases": cases}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "check_names", "source_sha256")}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
