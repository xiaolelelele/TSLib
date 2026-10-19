"""Post-hoc fixed daily seasonal-naive diagnostic for completed hourly H24 runs.

This standalone NumPy script neither imports the runner nor changes its protocol.
The sole added predictor is history[:, -24:, :]. It is not fitted or selected,
and its identical path must not be counted as three independent seed fits.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ("base", "learned_selected", "posterior_candidates")
METRICS = ("raw_sign_accuracy", "standardized_mse", "original_units_mse")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_metrics(history, truth, prediction, std):
    """Independent float64 path score; raw sign uses the last observed anchor."""
    x, y, z, scale = (np.asarray(a, dtype=np.float64) for a in (history, truth, prediction, std))
    if x.ndim != 3 or y.ndim != 3 or z.shape != y.shape or x.shape[0] != len(y):
        raise ValueError("incompatible path shapes")
    if x.shape[-1] != y.shape[-1] or scale.shape != (y.shape[-1],):
        raise ValueError("incompatible channel scales")
    if not y.size or not all(np.isfinite(a).all() for a in (x, y, z, scale)) or np.any(scale <= 0):
        raise ValueError("metric inputs must be nonempty, finite, with positive scales")
    true_changes = np.diff(np.concatenate([x[:, -1:], y], axis=1), axis=1)
    pred_changes = np.diff(np.concatenate([x[:, -1:], z], axis=1), axis=1)
    error = z - y
    return {"raw_sign_accuracy": float(np.mean(np.sign(true_changes) == np.sign(pred_changes))),
            "standardized_mse": float(np.mean(error ** 2)),
            "original_units_mse": float(np.mean((error * scale) ** 2))}


def source_check(result, history, truth, origins):
    """Check 24 rows really mean yesterday, and arrays match the saved CSV split."""
    cfg, meta = result["config"]["data"], result["data"]
    if cfg["kind"] != "csv":
        raise ValueError("this fixed daily diagnostic requires an hourly CSV")
    source = Path(cfg["csv_path"]).resolve()
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    times = np.asarray([r[cfg["time_column"]] for r in rows], dtype="datetime64[s]")
    if np.isnat(times).any() or not np.all(np.diff(times) == np.timedelta64(1, "h")):
        raise ValueError("daily seasonal naive requires exactly regular hourly timestamps")
    raw = np.asarray([[float(row[c]) for c in cfg["value_columns"]] for row in rows], dtype=np.float64)
    if len(raw) != meta["length"] or not np.isfinite(raw).all():
        raise ValueError("CSV length/finite values differ from the completed full run")
    train = raw[:meta["split_bounds"][1]]
    mean, std = train.mean(0), train.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    if not (np.array_equal(mean.astype(np.float32), np.asarray(result["normalization"]["mean"], dtype=np.float32))
            and np.array_equal(std.astype(np.float32), np.asarray(result["normalization"]["std"], dtype=np.float32))):
        raise AssertionError("saved normalization differs from train-only CSV normalization")
    standardized = ((raw - mean) / std).astype(np.float32)
    for i, origin in enumerate(origins):
        if not np.array_equal(history[i], standardized[origin-history.shape[1]+1:origin+1]):
            raise AssertionError("saved history differs from CSV")
        if not np.array_equal(truth[i], standardized[origin+1:origin+25]):
            raise AssertionError("saved truth differs from CSV")
        if times[origin+1] - times[origin-23] != np.timedelta64(24, "h"):
            raise AssertionError("seasonal source is not exactly 24 hours before the target")
    raw_errors = np.stack([raw[o-23:o+1] - raw[o+1:o+25] for o in origins])
    return {"csv_path": str(source), "csv_sha256": sha256(source),
            "hourly_cadence_verified": True, "arrays_exactly_match_csv": True,
            "columns": cfg["value_columns"],
            "test_first_target_timestamp": str(times[origins[0]+1]),
            "test_last_target_timestamp": str(times[origins[-1]+24]),
            "seasonal_naive_direct_csv_original_units_mse": float(np.mean(raw_errors ** 2))}


def evaluate(directory):
    directory = directory.resolve()
    files = {name: {"path": str(directory / name), "sha256": sha256(directory / name)}
             for name in ("results.json", "predictions.npz", "config.json")}
    result = json.loads((directory / "results.json").read_text())
    cfg, meta = result["config"], result["data"]
    if result.get("complete") is not True or result.get("schema_version") != 2 or cfg.get("protocol_version") != 2:
        raise ValueError("requires a completed v2 run: " + str(directory))
    if result.get("smoke") or cfg.get("smoke") or meta.get("smoke_truncated"):
        raise ValueError("diagnostic requires the complete non-smoke test split")
    if cfg != json.loads((directory / "config.json").read_text()):
        raise AssertionError("config.json differs from completed results")
    if meta["horizon"] != 24 or meta["lookback"] < 24:
        raise ValueError("this fixed baseline is defined only for H=24 and L>=24")
    if [r["seed"] for r in result["runs"]] != cfg["seeds"]:
        raise ValueError("all configured seed results are required")
    with np.load(directory / "predictions.npz", allow_pickle=False) as arrays:
        x, y, origins = (arrays["test__" + key] for key in ("history", "truth", "origins"))
        if y.shape[1] != 24 or x.shape[1] != meta["lookback"] or y.shape[-1] != meta["channels"]:
            raise AssertionError("array shapes differ from configured L/H/channels")
        if origins.shape != (len(y),) or not np.issubdtype(origins.dtype, np.integer) or np.any(np.diff(origins) <= 0):
            raise ValueError("test origins must be chronological integer row indices")
        source = source_check(result, x, y, origins)
        baseline = x[:, -24:, :].copy()
        std = result["normalization"]["std"]
        seasonal = path_metrics(x, y, baseline, std)
        if not np.isclose(seasonal["original_units_mse"], source["seasonal_naive_direct_csv_original_units_mse"], rtol=1e-6, atol=1e-12):
            raise AssertionError("de-standardized baseline MSE differs materially from direct raw CSV MSE")
        rows, summary = [], []
        for run in result["runs"]:
            seed = run["seed"]
            for method in METHODS:
                metrics = path_metrics(x, y, arrays["seed%s__test__%s" % (seed, method)], std)
                saved = run["methods"][method]
                expected = {"standardized_mse": saved["mse"], "raw_sign_accuracy": saved["raw_sign_accuracy"],
                            "original_units_mse": saved["original_units_mse"]}
                if not all(np.isclose(metrics[k], expected[k], atol=2e-10, rtol=2e-10) for k in METRICS):
                    raise AssertionError("independent NumPy scores disagree with saved model metrics")
                rows.append({"seed": seed, "method": method, **metrics,
                             "raw_gain_vs_seasonal_pp": 100*(metrics["raw_sign_accuracy"]-seasonal["raw_sign_accuracy"]),
                             "standardized_mse_delta_vs_seasonal": metrics["standardized_mse"]-seasonal["standardized_mse"],
                             "original_units_mse_delta_vs_seasonal": metrics["original_units_mse"]-seasonal["original_units_mse"]})
        for method in METHODS:
            members = [r for r in rows if r["method"] == method]
            fields = [k for k in members[0] if k not in ("seed", "method")]
            summary.append({"method": method, "model_seed_fits": len(members),
                            **{k + "_mean": float(np.mean([r[k] for r in members])) for k in fields},
                            **{k + "_seed_std_ddof0": float(np.std([r[k] for r in members], ddof=0)) for k in fields}})
    for item in files.values():
        if item["sha256"] != sha256(Path(item["path"])):
            raise RuntimeError("source run changed during diagnostic")
    digest = hashlib.sha256()
    for value in (x, y, origins):
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return {"group": cfg.get("group", directory.name), "dataset": cfg.get("dataset"),
            "backbone": cfg.get("backbone"), "input_files": files, "source": source,
            "shared_test_input_id": digest.hexdigest(), "test_windows": len(y),
            "scored_points": int(y.size), "lookback": x.shape[1], "horizon": 24,
            "seasonal_naive": {"fitted_models": 0, "distinct_prediction_paths": 1, "metrics": seasonal},
            "model_per_seed": rows, "model_seed_summary": summary,
            "independent_scores_match_saved_metrics": True}, baseline, origins


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, predictions = [], {}
    for directory in args.runs:
        result, baseline, origins = evaluate(directory)
        if Path(result["group"]).name != result["group"]:
            raise ValueError("group must be a plain filename")
        if result["group"] in predictions:
            raise ValueError("duplicate group")
        records.append(result)
        predictions[result["group"]] = (baseline, origins)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"analysis_status": "post-hoc diagnostic added after inspecting partial experiment results",
                "predictor": "test history[:, -24:, :] copied unchanged as the H24 prediction",
                "period_rows": 24, "period_search": False, "additional_variants": False,
                "training": False, "validation_selection": False,
                "changes_frozen_runner_or_success_criterion": False,
                "raw_sign": "equal signs of float64 consecutive changes, anchored at the final history point; zero is a separate exact sign",
                "mse": "uniform mean over test windows, 24 horizons and retained channels",
                "original_units": "standardized error multiplied by saved train-only channel std; direct CSV baseline MSE also recorded as a rounding cross-check",
                "seed_interpretation": "model seed variation is descriptive; the seasonal path is scored once per group and is never three independent fits",
                "shared_inputs": "groups with equal shared_test_input_id reuse exactly the same seasonal prediction and test data",
                "limitations": "one daily naive predictor; no weekly/period search; no statistical significance or causal mechanism claims; this is not a new method selection or a replacement for the frozen primary comparisons"}
    cross_backbone = []
    by_group = {record["group"]: record for record in records}
    if "repair_traffic_linear" in by_group and "repair_traffic_dlinear" in by_group:
        linear, dlinear = by_group["repair_traffic_linear"], by_group["repair_traffic_dlinear"]
        if linear["shared_test_input_id"] != dlinear["shared_test_input_id"]:
            raise AssertionError("cross-backbone comparison requires identical saved test inputs")
        left = next(row for row in linear["model_seed_summary"] if row["method"] == "learned_selected")
        right = next(row for row in dlinear["model_seed_summary"] if row["method"] == "base")
        cross_backbone.append({
            "analysis_status": "post-hoc descriptive comparison of seed means, not an independent mechanism test",
            "left": "repair_traffic_linear / learned_selected",
            "right": "repair_traffic_dlinear / base",
            "identical_test_inputs": True,
            "left_seed_mean_metrics": {key: left[key + "_mean"] for key in METRICS},
            "right_seed_mean_metrics": {key: right[key + "_mean"] for key in METRICS},
            "raw_sign_difference_left_minus_right_pp": 100*(left["raw_sign_accuracy_mean"]-right["raw_sign_accuracy_mean"]),
            "standardized_mse_difference_left_minus_right": left["standardized_mse_mean"]-right["standardized_mse_mean"],
            "original_units_mse_difference_left_minus_right": left["original_units_mse_mean"]-right["original_units_mse_mean"],
        })
    payload = {"protocol": protocol, "groups": records, "posthoc_cross_backbone_mean_comparison": cross_backbone}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "seasonal_once_per_group.csv", [
        {"group": r["group"], "shared_test_input_id": r["shared_test_input_id"],
         "fitted_models": 0, "distinct_prediction_paths": 1, **r["seasonal_naive"]["metrics"]} for r in records])
    write_csv(args.output / "models_per_seed.csv", [
        {"group": r["group"], **row} for r in records for row in r["model_per_seed"]])
    write_csv(args.output / "models_seed_summary.csv", [
        {"group": r["group"], **row} for r in records for row in r["model_seed_summary"]])
    for group, (baseline, origins) in predictions.items():
        np.savez_compressed(args.output / (group + "_seasonal_predictions.npz"), prediction=baseline, origins=origins)
    (args.output / "evaluate_repaired_seasonal_naive.py").write_bytes(Path(__file__).read_bytes())
    lines = ["# Fixed daily seasonal naive: post-hoc diagnostic", "",
             "Added after partial experiment results were seen. The only added predictor copies the last 24 hourly history values into the H24 future. No fitting, period search, validation selection, new variants, or change to the frozen runner and success criterion occurs.", "",
             "Raw sign is scored independently in NumPy, including the first step relative to the final observed history value. MSE uses a uniform mean over windows, horizons, and retained channels. Original-unit MSE de-standardizes each channel with its saved train-only standard deviation; `metrics.json` also records the direct raw-CSV naive MSE as a precision check.", "",
             "The deterministic baseline is scored once per group. Model means and population standard deviations (ddof=0) describe actual training seeds on the same test windows; they are not independent data replications. Equal `shared_test_input_id` values identify reused data and the same seasonal baseline across backbones.", "",
             "All shown model metrics were independently recomputed and checked against the frozen results. This checks whether a natural daily cycle baseline already suffices on these retained channels and test windows; it does not prove a causal mechanism, general superiority, or statistical significance.", "",
             "| Group | Method | Raw sign % | Standardized MSE | Original-unit MSE |", "|---|---|---:|---:|---:|"]
    for record in records:
        metric = record["seasonal_naive"]["metrics"]
        lines.append("| %s | seasonal naive (one deterministic path) | %.4f | %.8g | %.8g |" % (
            record["group"], 100*metric["raw_sign_accuracy"], metric["standardized_mse"], metric["original_units_mse"]))
        for row in record["model_seed_summary"]:
            lines.append("| %s | %s (seed mean) | %.4f | %.8g | %.8g |" % (
                record["group"], row["method"], 100*row["raw_sign_accuracy_mean"],
                row["standardized_mse_mean"], row["original_units_mse_mean"]))
    if cross_backbone:
        pair = cross_backbone[0]
        a, b = pair["left_seed_mean_metrics"], pair["right_seed_mean_metrics"]
        lines.extend(["", "A further post-hoc comparison uses identical Traffic columns 0–7 and saved test inputs: linear `learned_selected` averages %.4f%% raw sign and %.8g standardized MSE, while the uncorrected DLinear `base` averages %.4f%% and %.8g. These are training-seed means, not a significance claim or a substitute for the frozen comparisons within each backbone. DLinear corrections also exceed the fixed daily naive predictor in both direction and MSE on the Traffic groups shown here; daily repetition alone does not match those corrected results." % (
            100*a["raw_sign_accuracy"], a["standardized_mse"], 100*b["raw_sign_accuracy"], b["standardized_mse"])])
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    manifest = {"analysis_status": protocol["analysis_status"], "files": {
        path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "groups": [r["group"] for r in records],
                      "distinct_test_inputs": len({r["shared_test_input_id"] for r in records})}))


if __name__ == "__main__":
    main()
