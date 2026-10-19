"""Read-only NumPy audit for trend_lab.repaired schema-v2 experiment artifacts.

No experiment module or PyTorch is imported. Labels are computed by converting
saved endpoint values to float64 BEFORE subtraction. Output is a new audit file;
the source run directories and historical results are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def labels(history, future, scales, epsilon):
    x, y = np.asarray(history, dtype=np.float64), np.asarray(future, dtype=np.float64)
    joined = np.concatenate([x, y], axis=1)
    end = np.arange(x.shape[1], joined.shape[1])
    slopes = np.stack([(joined[:, end] - joined[:, end - k]) / k for k in scales], axis=1)
    threshold = np.asarray(epsilon, dtype=np.float64)[None, :, None, :]
    return np.where(slopes > threshold, 2, np.where(slopes < -threshold, 0, 1))


def path_metrics(history, truth, path, scales, epsilon):
    x, y, z = (np.asarray(v, dtype=np.float64) for v in (history, truth, path))
    if x.ndim != 3 or y.ndim != 3 or z.shape != y.shape or x.shape[0] != len(y) or x.shape[2] != y.shape[2]:
        raise ValueError("incompatible history/truth/path shapes")
    if not y.size or not all(np.isfinite(v).all() for v in (x, y, z)):
        raise ValueError("empty or nonfinite metric inputs")
    raw_truth = np.sign(np.diff(np.concatenate([x[:, -1:], y], axis=1), axis=1))
    raw_path = np.sign(np.diff(np.concatenate([x[:, -1:], z], axis=1), axis=1))
    true_labels, path_labels = labels(x, y, scales, epsilon), labels(x, z, scales, epsilon)
    rows = {}
    for i, k in enumerate(scales):
        matrix = np.bincount((3 * true_labels[:, i] + path_labels[:, i]).ravel(), minlength=9).reshape(3, 3)
        tp = np.diag(matrix)
        denom = matrix.sum(0) + matrix.sum(1)
        f1 = np.divide(2 * tp, denom, out=np.zeros(3, dtype=np.float64), where=denom != 0)
        rows[str(k)] = {"accuracy": float(np.mean(true_labels[:, i] == path_labels[:, i])),
                        "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist()}
    return {"mse": float(np.mean((z - y) ** 2)), "mae": float(np.mean(np.abs(z - y))),
            "raw_sign_accuracy": float(np.mean(raw_truth == raw_path)),
            "mean_3class_accuracy": float(np.mean(true_labels == path_labels)),
            "mse_by_horizon": np.mean((z - y) ** 2, axis=(0, 2)).tolist(),
            "scales": rows}


def posterior_indices(history, base, candidates, probabilities, scales, epsilon, beta):
    probs = np.asarray(probabilities, dtype=np.float64)
    scores = []
    for candidate in candidates:
        classes = labels(history, candidate, scales, epsilon)
        scores.append(np.take_along_axis(probs, classes[..., None], axis=-1)[..., 0].mean(axis=(1, 2, 3)))
    paths = np.asarray(candidates, dtype=np.float64)
    cost = ((paths - np.asarray(base, dtype=np.float64)[None]) ** 2).mean(axis=(2, 3))
    utility = np.stack(scores) - float(beta) * cost
    index = np.argmax(utility, axis=0)  # NumPy selects the first exact tie, as the runner does.
    return index, utility


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []
        self.max_numeric_error = 0.0

    def truth(self, condition, name, detail=None):
        self.count += 1
        if not condition:
            self.failures.append({"check": name, "detail": detail})

    def close(self, actual, expected, name, atol=2e-10, rtol=2e-10):
        a, b = np.asarray(actual), np.asarray(expected)
        equal_shape = a.shape == b.shape
        error = float(np.max(np.abs(a - b))) if equal_shape and a.size else 0.0
        if np.isfinite(error):
            self.max_numeric_error = max(self.max_numeric_error, error)
        self.truth(equal_shape and np.allclose(a, b, atol=atol, rtol=rtol), name,
                   {"max_abs_error": error if np.isfinite(error) else None,
                    "actual_shape": list(a.shape), "expected_shape": list(b.shape)})

    def exact(self, actual, expected, name):
        self.truth(np.array_equal(actual, expected), name)


def _origins(start, end, lookback, horizon, stride):
    result = np.arange(start - 1, end - horizon, stride, dtype=np.int64)
    return result[result >= lookback - 1]


def _audit_splits(result, arrays, check):
    meta, cfg = result["data"], result["config"]
    bounds = np.asarray(meta["split_bounds"], dtype=np.int64)
    recomputed_bounds = np.floor(np.r_[0, np.cumsum(cfg["data"]["split_ratios"])] * meta["length"]).astype(np.int64)
    recomputed_bounds[-1] = meta["length"]
    check.exact(bounds, recomputed_bounds, "split bounds from ratios")
    info = result["earlystop_calibration_split"]
    start, end = int(bounds[1]), int(bounds[2])
    fraction = float(cfg.get("earlystop_fraction", 0.5))
    cut = start + int(np.floor((end - start) * fraction))
    check.truth(info["cut_index"] == cut, "earlystop calibration cut")
    check.exact(info["source_bounds"], [start, end], "earlystop calibration source bounds")
    check.exact(info["earlystop_bounds"], [start, cut], "earlystop target bounds metadata")
    check.exact(info["calibration_bounds"], [cut, end], "calibration target bounds metadata")
    check.truth(info["earlystop_fraction"] == fraction, "earlystop fraction metadata")
    check.truth(info["target_intervals_disjoint"] is True, "target disjointness claim metadata")
    lookback, horizon, stride = meta["lookback"], meta["horizon"], meta["eval_stride"]
    source_origins = _origins(start, end, lookback, horizon, stride)
    left = source_origins + horizon + 1 <= cut
    right = source_origins + 1 >= cut
    expected = {"earlystop": source_origins[left], "calibration": source_origins[right],
                "validation": _origins(bounds[2], bounds[3], lookback, horizon, stride),
                "test": _origins(bounds[3], bounds[4], lookback, horizon, stride)}
    intervals = {"earlystop": [start, cut], "calibration": [cut, end],
                 "validation": [int(bounds[2]), int(bounds[3])], "test": [int(bounds[3]), int(bounds[4])]}
    check.truth(info["earlystop_windows"] == int(left.sum()), "full earlystop count")
    check.truth(info["calibration_windows"] == int(right.sum()), "full calibration count")
    check.exact(info["discarded_origins"], source_origins[~(left | right)], "discard boundary-crossing windows")
    for name, origins in expected.items():
        if result["smoke"]:
            origins = origins[:8]
        saved = arrays[name + "__origins"]
        check.exact(saved, origins, name + " origins")
        check.truth(len(saved) == result["split_windows"][name], name + " saved window count")
        check.truth(len(saved) > 0, name + " nonempty")
        check.truth(np.issubdtype(saved.dtype, np.integer), name + " integer origins")
        check.truth(np.all(saved + 1 >= intervals[name][0]) and np.all(saved + horizon + 1 <= intervals[name][1]),
                    name + " targets stay in interval")
        check.truth(arrays[name + "__history"].shape == (len(saved), lookback, meta["channels"]), name + " history shape")
        check.truth(arrays[name + "__truth"].shape == (len(saved), horizon, meta["channels"]), name + " truth shape")
    names = list(expected)
    for a, b in zip(names, names[1:]):
        check.truth(arrays[a + "__origins"].max() + horizon < arrays[b + "__origins"].min() + 1,
                    a + " / " + b + " target disjointness")
    return {"target_intervals": intervals, "cut_index": cut,
            "histories_can_overlap": True, "statistical_independence_claimed": False}


def _audit_csv_source(result, arrays, check):
    cfg, meta = result["config"]["data"], result["data"]
    if cfg["kind"] != "csv":
        return {"checked": False, "reason": "synthetic generator not duplicated; saved arrays audited"}
    source = Path(cfg["csv_path"])
    if not source.exists():
        raise FileNotFoundError("source CSV unavailable for independent audit: " + str(source))
    with source.open(newline="", encoding="utf-8-sig") as stream:
        records = csv.DictReader(stream)
        raw = np.asarray([[float(row[c]) for c in cfg["value_columns"]] for row in records], dtype=np.float64)
    if not cfg.get("smoke", False):
        check.truth(len(raw) == meta["length"], "full CSV row count without data truncation")
    raw = raw[:meta["length"]]
    check.truth(raw.shape == (meta["length"], meta["channels"]), "CSV retained shape")
    check.truth(np.isfinite(raw).all(), "CSV finite")
    train_end = meta["split_bounds"][1]
    mean, std = raw[:train_end].mean(0), raw[:train_end].std(0)
    std = np.where(std < 1e-8, 1.0, std)
    check.exact(np.asarray(result["normalization"]["mean"], dtype=np.float32), mean.astype(np.float32), "train-only mean")
    check.exact(np.asarray(result["normalization"]["std"], dtype=np.float32), std.astype(np.float32), "train-only std")
    standardized = ((raw - mean) / std).astype(np.float32)
    lookback, horizon = meta["lookback"], meta["horizon"]
    for split in ("earlystop", "calibration", "validation", "test"):
        origins = arrays[split + "__origins"]
        expected_x = np.stack([standardized[o-lookback+1:o+1] for o in origins])
        expected_y = np.stack([standardized[o+1:o+horizon+1] for o in origins])
        check.exact(arrays[split + "__history"], expected_x, split + " history reconstructed from CSV")
        check.exact(arrays[split + "__truth"], expected_y, split + " truth reconstructed from CSV")
    train_origins = _origins(0, train_end, lookback, horizon, meta["train_stride"])
    if result["smoke"]:
        train_origins = train_origins[:64]
    check.truth(len(train_origins) == result["split_windows"]["train"], "train window count from CSV and stride")
    tx = np.stack([standardized[o-lookback+1:o+1] for o in train_origins]).astype(np.float64)
    ty = np.stack([standardized[o+1:o+horizon+1] for o in train_origins]).astype(np.float64)
    scales = result["config"].get("scales", [1, 2, 4, 8])
    joined = np.concatenate([tx, ty], axis=1)
    endpoint = np.arange(lookback, lookback + horizon)
    quantile = result["config"].get("epsilon_quantile", 0.2)
    thresholds, loss_scales = [], []
    for k in scales:
        slopes = np.abs((joined[:, endpoint] - joined[:, endpoint-k]) / k)
        thresholds.append(np.maximum(np.quantile(slopes, quantile, axis=(0, 1)), 1e-6))
        loss_scales.append(np.maximum(np.quantile(slopes, .5, axis=(0, 1)), result["config"].get("slope_scale_floor", .02)))
    thresholds = np.asarray(thresholds, dtype=np.float32)
    check.close(result["epsilon"], thresholds, "train-only float64 threshold quantile", atol=1e-8, rtol=1e-7)
    check.close(result["slope_scale"], np.asarray(loss_scales, dtype=np.float32),
                "train-only independent loss scale", atol=1e-8, rtol=1e-7)
    train_labels = labels(tx, ty, scales, result["epsilon"])
    counts = np.stack([(train_labels == c).sum(axis=(0, 2)) for c in range(3)], axis=-1)
    check.exact(result["prior_counts"], counts, "train-only prior counts")
    return {"checked": True, "path": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def _audit_diagnostics(diagnostics, count, config, check, prefix):
    """Check saved diagnostic consistency, without claiming gradient recomputation."""
    for key, diag in diagnostics.items():
        if "projected_gradient_rms" not in diag:
            continue
        name = prefix + " " + key
        grad = np.asarray(diag["projected_gradient_rms"])
        iterations = np.asarray(diag["iterations_per_path"])
        perturbation = np.asarray(diag["perturbation_rms"])
        check.truth(grad.shape == iterations.shape == perturbation.shape == (count,), name + " shapes")
        check.truth(np.isfinite(grad).all() and np.isfinite(perturbation).all()
                    and (grad >= 0).all() and (perturbation >= 0).all(), name + " finite nonnegative norms")
        check.truth((iterations >= 0).all() and (iterations <= diag["max_steps"]).all(), name + " iteration bounds")
        check.truth(diag["max_steps"] == config.get("projection_steps", 400), name + " configured step budget")
        check.truth(diag["convergence_tol"] == config.get("projection_tolerance", 1e-5), name + " configured tolerance")
        cap = config.get("max_rms", .5)
        check.truth(np.all(perturbation <= cap + 10*np.finfo(np.float32).eps), name + " RMS cap")
        check.exact(diag["cap_binding_per_path"], perturbation >= cap - 10*np.finfo(np.float32).eps,
                    name + " cap-binding flags")
        if diag["convergence_tol"] is not None:
            expected = grad <= diag["convergence_tol"]
            check.exact(diag["converged_per_path"], expected, name + " convergence flags")
            check.truth(diag["all_converged"] == bool(expected.all()), name + " all-converged flag")


def audit_run(directory):
    root = Path(directory).resolve()
    result = json.loads((root / "results.json").read_text())
    if result.get("schema_version") != 2 or result.get("config", {}).get("protocol_version") != 2:
        raise ValueError("refusing non-v2 or historical results: " + str(root))
    arrays = np.load(root / "predictions.npz", allow_pickle=False)
    check = Checks()
    check.truth(result.get("complete") is True, "run complete")
    check.truth(result["config"] == json.loads((root / "config.json").read_text()), "config consistency")
    for key in arrays.files:
        check.truth(np.isfinite(arrays[key]).all(), "finite array " + key)
        if key.endswith("__probs"):
            check.truth(np.all(arrays[key] >= 0), "nonnegative probabilities " + key)
            check.close(arrays[key].sum(-1), np.ones(arrays[key].shape[:-1]), "probability sums " + key,
                        atol=1e-5, rtol=1e-5)
    split_report = _audit_splits(result, arrays, check)
    source_report = _audit_csv_source(result, arrays, check)
    cfg, scales, eps = result["config"], result["config"].get("scales", [1, 2, 4, 8]), result["epsilon"]
    seeds = [r["seed"] for r in result["runs"]]
    check.exact(seeds, cfg.get("seeds", [0, 1, 2]), "all configured seeds completed in order")
    details = []
    for run in result["runs"]:
        seed, val = run["seed"], run["validation"]
        _audit_diagnostics(val["convergence"], len(arrays["validation__truth"]), cfg, check,
                           "validation diagnostics %s" % seed)
        _audit_diagnostics(run["test_convergence"], len(arrays["test__truth"]), cfg, check,
                           "test diagnostics %s" % seed)
        check.truth(run.get("checkpoint_reload_verified") is True, "recorded checkpoint reload " + str(seed))
        check.truth(val == json.loads((root / ("seed%s_validation_lock.json" % seed)).read_text()),
                    "validation lock matches final record " + str(seed))
        vx, vy = arrays["validation__history"], arrays["validation__truth"]
        prefix = "seed%s__validation__" % seed
        paths = {key[len(prefix):]: arrays[key] for key in arrays.files
                 if key.startswith(prefix) and not key.endswith(("__probs", "__indices"))}
        computed = {name: path_metrics(vx, vy, path, scales, eps) for name, path in paths.items()}
        check.truth(set(paths) == set(val["all_model_metrics"]), "all validation paths recorded " + str(seed))
        for name, metrics in val["all_model_metrics"].items():
            for metric, value in metrics.items():
                check.close(value, computed[name][metric], "validation %s %s %s" % (seed, name, metric))
        for metric, value in val["base"].items():
            check.close(value, computed["base"][metric], "validation base %s %s" % (seed, metric))
        chosen = {}
        for family, rows in val["grids"].items():
            feasible = []
            for order, row in enumerate(rows):
                name = row["key"]
                check.truth(order == row["order"], "grid order %s %s %s" % (seed, family, name))
                for metric, value in row["metrics"].items():
                    check.close(value, computed[name][metric], "grid %s %s %s %s" % (seed, family, name, metric))
                if (computed[name]["raw_sign_accuracy"] > computed["base"]["raw_sign_accuracy"]
                        and computed[name]["mse"] <= computed["base"]["mse"] + 1e-8):
                    feasible.append((name, order))
            expected = min(feasible, key=lambda item: (-computed[item[0]]["raw_sign_accuracy"],
                           computed[item[0]]["mse"], item[1]))[0] if feasible else "base"
            chosen[family] = expected
            check.truth(val["choices"][family] == expected, "chosen candidate %s %s" % (seed, family),
                        {"expected": expected, "saved": val["choices"][family]})
            check.exact(paths[family], paths[expected], "validation selected path identity %s %s" % (seed, family))
        alphas = cfg.get("alpha_grid", [0, .1, .3, 1, 3, 10])
        library = ["base"] + ["projection_alpha_%g" % a for a in alphas[1:]] + ["projection_single_alpha_%g" % a for a in alphas[1:]]
        probabilities = arrays[prefix + "direction_head__probs"]
        candidate_arrays = np.stack([paths[k] for k in library])
        posterior_checks = []
        for beta in cfg.get("posterior_betas", [0, .1, 1, 10]):
            name = "posterior_beta_%g" % beta
            expected, utility = posterior_indices(vx, paths["base"], candidate_arrays, probabilities, scales, eps, beta)
            saved = arrays[prefix + name + "__indices"]
            check.exact(saved, expected, "posterior indices %s %s" % (seed, name))
            rebuilt = candidate_arrays[expected, np.arange(len(vx))]
            check.exact(paths[name], rebuilt, "posterior selected candidate path %s %s" % (seed, name))
            ordered = np.sort(utility, axis=0)
            margin = ordered[-1] - ordered[-2] if len(ordered) > 1 else np.zeros(len(vx))
            posterior_checks.append({"candidate": name, "windows": len(saved),
                                     "indices_equal": bool(np.array_equal(saved, expected)),
                                     "exact_utility_ties": int((margin == 0).sum()),
                                     "candidate_counts": np.bincount(expected, minlength=len(library)).tolist()})
        tx, ty = arrays["test__history"], arrays["test__truth"]
        test_rows, model_rows = {}, {}
        entries = [(name, metric, "test", test_rows) for name, metric in run["methods"].items()]
        entries += [(name, metric, "test_model", model_rows) for name, metric in run.get("test_models", {}).items()]
        for name, saved_metric, namespace, table in entries:
            path_key = "seed%s__%s__%s" % (seed, namespace, name)
            # The direction-head row deliberately reports the unchanged base path.
            path = arrays[path_key] if path_key in arrays else arrays["seed%s__test__base" % seed]
            check.truth(path_key in arrays or name == "direction_head", "saved test path %s %s" % (seed, name))
            metrics = path_metrics(tx, ty, path, scales, eps)
            table[name] = metrics
            for key in ("mse", "mae", "raw_sign_accuracy", "mse_by_horizon"):
                check.close(saved_metric[key], metrics[key], "test %s %s %s" % (seed, name, key))
            for k, values in metrics["scales"].items():
                for metric, value in values.items():
                    check.close(saved_metric["scales"][k][metric], value, "test %s %s scale%s %s" % (seed, name, k, metric))
            if "original_units_mse" in saved_metric:
                raw_mse = float(np.mean(((path.astype(np.float64) - ty.astype(np.float64)) * np.asarray(result["normalization"]["std"])) ** 2))
                check.close(saved_metric["original_units_mse"], raw_mse, "original units MSE %s %s" % (seed, name))
            target = chosen.get(name)
            target_key = "seed%s__test__%s" % (seed, target)
            if target_key not in arrays:
                target_key = "seed%s__test_model__%s" % (seed, target)
            if target and target_key in arrays:
                check.exact(path, arrays[target_key], "test selected alias identity %s %s" % (seed, name))
        details.append({"seed": seed, "validation_choices_recomputed": chosen,
                        "posterior_library": library, "posterior_validation": posterior_checks,
                        "test_metrics_recomputed": test_rows, "test_model_metrics_recomputed": model_rows})
    arrays.close()
    return {"run_directory": str(root), "passed": not check.failures, "checks": check.count,
            "failures": check.failures, "max_numeric_error": check.max_numeric_error,
            "smoke": result["smoke"], "split_audit": split_report, "source_audit": source_report,
            "artifact_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ("results.json", "predictions.npz", "config.json")},
            "runs": details, "limits": ["No independent neural checkpoint inference or optimizer audit.",
                "All saved validation candidates and beta indices were audited; unsaved test candidate libraries were not reconstructed.",
                "Probability calibration quality/temperature selection and events were not independently recomputed.",
                "Saved projection norm/iteration/flag consistency was checked; objective gradients were not independently recomputed.",
                "A matching validation lock does not independently establish wall-clock ordering of writes."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite audit " + str(output))
    audits = [audit_run(path) for path in args.runs]
    payload = {"passed": all(item["passed"] for item in audits), "checks": sum(item["checks"] for item in audits),
               "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "groups": audits}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": payload["passed"], "checks": payload["checks"], "output": str(output),
                      "failures": sum(len(item["failures"]) for item in audits)}))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
