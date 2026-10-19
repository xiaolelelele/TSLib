"""Independently audit saved pilot artifacts with NumPy; never fit/select models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def labels(history, future, scales, epsilon):
    history, future = history.astype(np.float64), future.astype(np.float64)
    joined = np.concatenate((history, future), axis=1)
    steps = np.arange(future.shape[1]) + history.shape[1]
    slopes = np.stack([(joined[:, steps] - joined[:, steps - k]) / k for k in scales], axis=1)
    eps = np.asarray(epsilon)[None, :, None, :]
    return np.where(slopes > eps, 2, np.where(slopes < -eps, 0, 1))


def macro_f1(truth, pred):
    matrix = np.bincount((truth * 3 + pred).ravel(), minlength=9).reshape(3, 3)
    denom = matrix.sum(0) + matrix.sum(1)
    return np.divide(2 * np.diag(matrix), denom, out=np.zeros(3), where=denom != 0).mean()


def interval(values, block=8, repeats=2000):
    rng = np.random.default_rng(20260927)
    n = len(values)
    block = min(block, n)
    starts = rng.integers(0, n - block + 1, size=(repeats, int(np.ceil(n / block))))
    indices = (starts[..., None] + np.arange(block)).reshape(repeats, -1)[:, :n]
    return np.quantile(values[indices].mean(1), [.025, .975]).tolist()


def audit(directory):
    root = Path(directory).resolve()
    result = json.loads((root / "results.json").read_text())
    cfg = json.loads((root / "config.json").read_text())
    assert result["checkpoint_reload_verified"] is True
    assert not result["smoke"] and not result["data"]["smoke_truncated"]
    assert [r["seed"] for r in result["runs"]] == cfg["seeds"] == result["seeds"]
    scales, epsilon = result["scales"], result["epsilon"]
    dc = cfg["data"]
    source = Path(dc["csv_path"])
    if not source.is_absolute():
        source = Path(__file__).resolve().parents[1] / source
    with source.open(newline="", encoding="utf-8-sig") as stream:
        raw = np.array([[float(row[c]) for c in dc["value_columns"]] for row in csv.DictReader(stream)])
    bounds = result["data"]["split_bounds"]
    train = raw[:bounds[1]]
    mean, std = train.mean(0), train.std(0)
    std = np.where(std < 1e-8, 1., std)
    np.testing.assert_allclose(mean, result["normalization"]["mean"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(std, result["normalization"]["std"], rtol=1e-6, atol=1e-6)
    standardized = ((raw - mean) / std).astype(np.float32)
    horizon, lookback = dc["horizon"], dc["lookback"]
    origins = np.arange(bounds[3] - 1, bounds[4] - horizon, dc["eval_stride"])
    expected_y = np.stack([standardized[o+1:o+1+horizon] for o in origins])
    expected_x = np.stack([standardized[o-lookback+1:o+1] for o in origins])
    truth_labels = labels(expected_x, expected_y, scales, epsilon)
    train_origins = np.arange(-1, bounds[1] - horizon, dc["train_stride"])
    train_origins = train_origins[train_origins >= lookback - 1]
    train_steps = train_origins[:, None] + np.arange(1, horizon + 1)
    slopes = np.stack([np.abs((standardized[train_steps] - standardized[train_steps-k]) / k)
                       for k in scales])
    fitted_epsilon = np.maximum(np.quantile(slopes, cfg.get("epsilon_quantile", .2), axis=(1, 2)), 1e-6)
    np.testing.assert_allclose(epsilon, fitted_epsilon, rtol=2e-5, atol=2e-6)
    rows, scores, mses, alphas = {}, {}, {}, []
    with np.load(root / "predictions.npz") as archive:
        np.testing.assert_array_equal(archive["origins"], origins)
        for key in archive.files:
            assert np.isfinite(archive[key]).all(), key
            assert len(archive[key]) == len(origins), key
        for run in result["runs"]:
            seed = run["seed"]
            np.testing.assert_array_equal(archive[f"seed{seed}_truth"], expected_y)
            np.testing.assert_array_equal(archive[f"seed{seed}_history"], expected_x)
            val = run["validation"]
            base = val["base"]
            base_acc = np.mean([m["accuracy"] for m in base["scales"].values()])
            base_f1 = np.mean([m["macro_f1"] for m in base["scales"].values()])
            for method, grid_key in [("projection", "alpha_grid"), ("projection_single", "projection_single_alpha_grid")]:
                feasible = [g for g in val[grid_key] if g["accuracy"] > base_acc
                            and g["macro_f1"] >= base_f1 - 1e-12 and g["mse"] <= base["mse"] * 1.01]
                chosen = sorted(feasible, key=lambda g: (-g["accuracy"], g["mse"], g["alpha"]))[0]["alpha"] if feasible else 0
                assert chosen == val["selected_alpha"][method]
            alphas.append({"seed": seed, "temperature": val["temperature"], **val["selected_alpha"]})
            for method, rec in run["methods"].items():
                if "metrics" not in rec:
                    probs = archive[f"seed{seed}_{method}_probs"]
                    score = (probs.argmax(-1) == truth_labels).mean()
                    np.testing.assert_allclose(score, rec["direction_metrics"]["overall"]["accuracy"])
                    f1 = macro_f1(truth_labels, probs.argmax(-1))
                    np.testing.assert_allclose(f1, rec["direction_metrics"]["overall"]["macro_f1"])
                    scale_f1 = np.mean([macro_f1(truth_labels[:, i], probs[:, i].argmax(-1)) for i in range(len(scales))])
                    rows.setdefault(method, []).append({"head_accuracy": float(score), "head_macro_f1": float(f1),
                                                       "head_macro_f1_scale_mean": float(scale_f1)})
                    continue
                pred = archive[f"seed{seed}_{method}"]
                pred_labels = labels(expected_x, pred, scales, epsilon)
                per_window = (pred_labels == truth_labels).mean((1, 2, 3))
                mse = ((pred.astype(np.float64) - expected_y) ** 2).mean((1, 2))
                metrics = rec["metrics"]
                np.testing.assert_allclose(mse.mean(), metrics["mse"], rtol=1e-10)
                for i, scale in enumerate(scales):
                    reported = metrics["scales"][str(scale)]
                    np.testing.assert_allclose((pred_labels[:, i] == truth_labels[:, i]).mean(), reported["accuracy"])
                    np.testing.assert_allclose(macro_f1(truth_labels[:, i], pred_labels[:, i]), reported["macro_f1"])
                row = {"accuracy": float(per_window.mean()), "macro_f1": float(np.mean([m["macro_f1"] for m in metrics["scales"].values()])),
                       "mse": float(mse.mean()), "raw_sign_accuracy": metrics["raw_sign_accuracy"],
                       "mse_ratio": rec["test_mse_ratio_to_base"]}
                for i, scale in enumerate(scales):
                    row[f"accuracy_scale_{scale}"] = metrics["scales"][str(scale)]["accuracy"]
                    row[f"predicted_flat_scale_{scale}"] = float((pred_labels[:, i] == 1).mean())
                if "probability" in metrics:
                    probs = archive[f"seed{seed}_{method}_probs"]
                    row["head_accuracy"] = float((probs.argmax(-1) == truth_labels).mean())
                    np.testing.assert_allclose(row["head_accuracy"], metrics["probability"]["accuracy"])
                    row["head_macro_f1"] = float(macro_f1(truth_labels, probs.argmax(-1)))
                    np.testing.assert_allclose(row["head_macro_f1"], metrics["probability"]["macro_f1"])
                    row["head_macro_f1_scale_mean"] = float(np.mean([
                        macro_f1(truth_labels[:, i], probs[:, i].argmax(-1)) for i in range(len(scales))]))
                    for i, scale in enumerate(scales):
                        row[f"head_flat_scale_{scale}"] = float((probs[:, i].argmax(-1) == 1).mean())
                rows.setdefault(method, []).append(row)
                scores.setdefault(method, []).append(per_window)
                mses.setdefault(method, []).append(mse)
    summary = {method: {key: {"mean": float(np.mean([r[key] for r in records])),
                                  "std": float(np.std([r[key] for r in records], ddof=1)) if len(records) > 1 else 0}
                        for key in records[0]} for method, records in rows.items()}
    comparisons = {}
    for method in ["projection", "projection_single", "residual", "residual_direction", "difference"]:
        for ref in (["base", "projection_single", "residual", "residual_direction"] if method == "projection" else ["base"]):
            if method not in scores or ref not in scores:
                continue
            delta = np.array(scores[method]) - np.array(scores[ref])
            mse_delta = np.array(mses[method]) - np.array(mses[ref])
            comparisons[f"{method}_vs_{ref}"] = {
                "accuracy_delta_per_seed": delta.mean(1).tolist(), "accuracy_delta": float(delta.mean()),
                "accuracy_ci95": interval(delta.mean(0)), "mse_delta": float(mse_delta.mean()),
                "mse_delta_ci95": interval(mse_delta.mean(0))}
    report = {"run": str(root), "source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "verified": True, "checks": ["train-only normalization and thresholds", "test windows match source CSV",
              "finite aligned arrays", "saved MSE/direction accuracy/macro-F1 recomputed", "validation-only alpha selection", "runner checkpoint reload flag"],
              "seeds": result["seeds"], "test_windows": len(origins), "split_bounds": bounds,
              "true_class_fractions_by_scale": {str(k): {name: float((truth_labels[:, i] == c).mean())
                  for c, name in enumerate(["down", "flat", "up"])} for i, k in enumerate(scales)},
              "summary": summary, "alphas": alphas, "comparisons": comparisons,
              "ci_note": "Moving-block bootstrap (8 chronological windows, 2000 replicates), on per-window differences averaged over fitted seeds; conditional on this dataset and these seeds, not independent dataset repetitions."}
    destination = root / "independent_audit.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory)
