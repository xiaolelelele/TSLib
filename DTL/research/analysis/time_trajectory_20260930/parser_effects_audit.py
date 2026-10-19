"""Assess parser precision with frozen normalization and held predictions.

Does not retrain or change the experiment. Prediction-label checks use held
paths, so exact-anchor rules need regeneration to simulate another pipeline.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def labels(anchor, path):
    previous = np.concatenate((anchor[:, None, :], path[:, :-1]), axis=1)
    return np.where(path < previous, 0, np.where(path > previous, 2, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results)
    rows, checks, failures = [], 0, []
    for dataset_file in sorted(root.glob("*/dataset.json")):
        metadata = json.loads(dataset_file.read_text())
        with open(metadata["csv"], newline="") as stream:
            reader = csv.reader(stream)
            next(reader)
            raw = np.array([[float(v) if v else np.nan for v in row[1:]] for row in reader], dtype=np.float64)
        values = (raw - np.array(metadata["mean"])) / np.array(metadata["std"])
        for prediction_file in sorted(dataset_file.parent.glob("*_seed*/predictions.npz")):
            # Shared OCE/CaReTS files have no anchors and are audited via their
            # copies in each completed backbone case; skip standalone shared.
            if prediction_file.parent.name.startswith("shared_"):
                continue
            with np.load(prediction_file) as archive:
                origins = archive["test__origins"]
                original_x, original_y = archive["test__x"], archive["test__y"]
                L, H = original_x.shape[1], original_y.shape[1]
                history = values.copy()
                if metadata["history_policy"] == "causal_ffill":
                    for channel in range(raw.shape[1]):
                        last = np.nan
                        for i in range(len(raw)):
                            if np.isfinite(history[i, channel]):
                                last = history[i, channel]
                            else:
                                history[i, channel] = last
                alternate_x = np.stack([history[i-L+1:i+1] for i in origins])
                alternate_y = np.stack([values[i+1:i+H+1] for i in origins])
                input_equal = np.array_equal(original_x.astype(np.float32), alternate_x.astype(np.float32))
                target_equal = np.array_equal(original_y.astype(np.float32), alternate_y.astype(np.float32))
                true_old, true_new = labels(original_x[:, -1], original_y), labels(alternate_x[:, -1], alternate_y)
                row = {"case": str(prediction_file.parent), "predictions_sha256": hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
                       "float32_input_bit_identical": bool(input_equal), "float32_target_bit_identical": bool(target_equal),
                       "true_class_changed_elements": int((true_old != true_new).sum()),
                       "float64_y_max_difference": float(np.max(np.abs(original_y-alternate_y))), "methods": {}}
                checks += 3
                if not input_equal or not target_equal or np.any(true_old != true_new):
                    failures.append(str(prediction_file.parent) + " input/target/true-label discrepancy")
                for key in archive.files:
                    if not key.startswith("test__") or key in ("test__x", "test__y", "test__origins", "test__probabilities"):
                        continue
                    path = archive[key]
                    if path.shape != original_y.shape:
                        continue
                    mse_old = float(np.mean((path-original_y)**2))
                    mse_new = float(np.mean((path-alternate_y)**2))
                    prediction_old = labels(original_x[:, -1], path)
                    prediction_new = labels(alternate_x[:, -1], path)
                    changed_mask = prediction_old != prediction_new
                    changed = int(changed_mask.sum())
                    accuracy_old = float(np.mean(prediction_old == true_old))
                    accuracy_new = float(np.mean(prediction_new == true_new))
                    row["methods"][key] = {"mse_old": mse_old, "mse_nearest_parse": mse_new,
                                           "mse_absolute_difference": abs(mse_new-mse_old),
                                           "held_path_raw_accuracy_difference_pp": 100 * (accuracy_new-accuracy_old),
                                           "held_path_predicted_class_changed_elements": changed,
                                           "changed_after_first_horizon": int(changed_mask[:, 1:].sum()),
                                           "held_path_note": "alternate parser changes anchor last bit; paths are held fixed, not regenerated" if changed else None}
                    checks += 1
                    if abs(mse_new-mse_old) > 1e-10 * max(1, abs(mse_old)):
                        failures.append(str(prediction_file.parent) + " " + key + " nontrivial MSE change")
                rows.append(row)
    effects = [m for row in rows for m in row["methods"].values()]
    result = {"completed_cases_at_audit": len(rows), "checks": checks, "failures": failures,
              "max_method_mse_absolute_difference": max((m["mse_absolute_difference"] for m in effects), default=0),
              "held_paths_with_predicted_class_difference": sum(m["held_path_predicted_class_changed_elements"] > 0 for m in effects),
              "max_held_path_predicted_class_changed_elements": max((m["held_path_predicted_class_changed_elements"] for m in effects), default=0),
              "max_held_path_raw_accuracy_absolute_difference_pp": max((abs(m["held_path_raw_accuracy_difference_pp"]) for m in effects), default=0),
              "normalization": "original dataset.json train statistics held fixed; no normalization refit or retraining",
              "interpretation": "true labels/float32 tensors can be equivalent while held exact-anchor paths are not invariant to a decimal-parser last-bit perturbation; this is reported explicitly",
              "cases": rows}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
