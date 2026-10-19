"""Independent controls audit: nearest integer rule, raw classes, source locks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

METHODS = ("mean_raw", "residual_capacity", "residual_direction_capacity", "mse_base", "cosdir", "tdalign")


def raw_class(x, z):
    prev = np.concatenate((x[:, -1:], z[:, :-1]), axis=1)
    return np.sign(z-prev).astype(np.int64)+1


def independent_round(a):
    low = np.floor(a)
    fraction = a-low
    high = low+1
    result = np.where(fraction < .5, low, np.where(fraction > .5, high, np.where(low % 2 == 0, low, high)))
    return np.maximum(result, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls", required=True)
    parser.add_argument("--frozen-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.frozen_source)
    from trend_lab.trajectory_decision import make_common_grids
    root, output = Path(args.controls), Path(args.output)
    checks, failures, rows, cases = [], [], [], []

    def check(name, value):
        checks.append(name)
        if not bool(value):
            failures.append(name)

    for report_path in sorted(root.glob("*/*/result.json")):
        report = json.loads(report_path.read_text())
        source = Path(report["source_case"])
        meta = report["metadata"]
        mean, std = np.array(meta["mean"]), np.array(meta["std"])
        for path, key in [(source/"predictions.npz", "source_predictions_sha256"),
                          (source/"validation_lock.json", "source_validation_lock_sha256"),
                          (source/"result.json", "source_primary_result_sha256"),
                          (report_path.parent/"predictions.npz", "controls_predictions_sha256")]:
            check(str(report_path.parent)+" hash "+key, hashlib.sha256(path.read_bytes()).hexdigest() == report[key])
        check(str(report_path.parent)+" no training/selection", report["posthoc_diagnostic"] and report["no_training"] and report["no_selection"])
        with np.load(source/"predictions.npz") as original, np.load(report_path.parent/"predictions.npz") as controlled:
            for split in ("validation", "test"):
                x, y = original[split+"__x"], original[split+"__y"]
                for name in ("x", "y", "origins"):
                    check(str(report_path.parent)+" "+split+" source "+name, np.array_equal(original[split+"__"+name], controlled[split+"__"+name]))
                truth = raw_class(x, y)
                for method in METHODS:
                    name = method+"_native_round"
                    path = controlled[split+"__"+name]
                    raw = original[split+"__"+method]*std+mean
                    rounded = independent_round(raw)
                    expected = (rounded-mean)/std
                    check(str(report_path.parent)+" "+split+" independent "+name, np.array_equal(path, expected))
                    check(str(report_path.parent)+" "+split+" native grid "+name, np.all(rounded >= 0) and np.all(rounded == np.floor(rounded)))
                b, mu, p = original[split+"__cosdir"], original[split+"__mean"], original[split+"__probabilities"]
                grids, _ = make_common_grids(x, b, mu, mode="native", step=.1, padding=.25,
                    normalizer_mean=mean, normalizer_std=std, native_step=1., native_origin=0.,
                    lower=-mean/std, max_states=257, coarsen=True)
                for method in ("residual_capacity", "residual_direction_capacity"):
                    center = original[split+"__"+method]
                    nearest = np.empty_like(center, dtype=np.float64)
                    # Enumerate actual grid distances independently of DP.
                    for sample in range(len(x)):
                        for channel in range(x.shape[-1]):
                            grid = grids[sample][channel]
                            distances = np.abs(center[sample, :, channel, None]-grid[None, :])
                            nearest[sample, :, channel] = grid[np.argmin(distances, axis=1)]
                    nearest_cost = ((nearest-center)**2).mean((1, 2))
                    b_cost = ((b-center)**2).mean((1, 2))
                    path = controlled[split+"__"+method+"_shared_grid"]
                    actual_cost = ((path-center)**2).mean((1, 2))
                    check(str(report_path.parent)+" "+split+" shared-grid point optimum "+method,
                          np.allclose(actual_cost, np.minimum(nearest_cost, b_cost), rtol=1e-12, atol=1e-12))
                for method, stored in report["stages"][split]["metrics"].items():
                    path = controlled[split+"__"+method]
                    predicted = raw_class(x, path)
                    confusion = np.bincount((3*truth+predicted).ravel(), minlength=9).reshape(3, 3)
                    mse = float(np.mean((path-y)**2))
                    accuracy = float(np.mean(predicted == truth))
                    check(str(report_path.parent)+" "+split+" metrics "+method,
                          abs(mse-stored["mse"]) <= 1e-12 and accuracy == stored["raw_accuracy"] and confusion.tolist() == stored["confusion"])
                    rows.append({"domain": report_path.parent.parent.name, "case": source.name, "split": split,
                                 "method": method, "mse": mse, "raw_accuracy": accuracy,
                                 "macro_f1": stored["macro_f1"], "predicted_zero_fraction": stored["predicted_zero_fraction"],
                                 "vs_original_accuracy_pp": 100*(accuracy-report["stages"][split]["reference_metrics"][method[:-len("_native_round")]]["raw_accuracy"]) if method.endswith("_native_round") else None})
        cases.append(str(report_path.parent))
    result = {"cases": len(cases), "checks": len(checks), "failures": failures, "check_names": checks,
              "case_paths": cases, "posthoc": True, "no_selection": True,
              "independence": "manual floor/fraction/tie-even round; independent signs/confusion; grid-distance enumeration instead of decoder; source hashes unchanged"}
    output.mkdir(parents=True, exist_ok=True)
    (output/"precision_controls_audit.json").write_text(json.dumps(result, indent=2))
    if rows:
        with (output/"precision_controls_per_seed.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: v for k, v in result.items() if k not in ("check_names", "case_paths")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
