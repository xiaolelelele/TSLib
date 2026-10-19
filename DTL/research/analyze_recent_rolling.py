"""Summaries, paired time-block intervals and declared physical-range diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sign(x, y):
    extended = np.concatenate((x[:, -1:].astype(float), y.astype(float)), 1)
    return np.sign(extended[:, 1:] - extended[:, :-1])


def stats(x, y, z):
    t, p = sign(x, y), sign(x, z)
    return {"raw_accuracy": np.mean(t==p), "mse": np.mean((z.astype(float)-y.astype(float))**2),
            "raw_nonzero_accuracy": np.mean((t==p)[t!=0]), "raw_zero_accuracy": np.mean((t==p)[t==0]),
            "zero_fraction": np.mean(t==0)}


def time_interval(per_origin, origins, seed=9030):
    blocks = (origins-origins.min())//(24*7)
    ids = np.unique(blocks)
    values = np.asarray(per_origin)
    totals = np.stack([values[blocks==b].sum(0) for b in ids])
    counts = np.array([(blocks==b).sum() for b in ids])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(ids), size=(2000, len(ids)))
    sampled = totals[samples].sum(1)/counts[samples].sum(1)[:, None]
    lower, upper = np.quantile(sampled, [.025, .975], axis=0)
    return lower, upper, len(ids)


def physical_clip(z, metadata, dataset):
    """Post hoc fixed domain ranges, never tuned against any labels.

    This diagnostic is separate from the preregistered method matrix. Bounds
    are specified by the variable meaning, and truth violations are reported.
    """
    mean, std = np.array(metadata["mean"]), np.array(metadata["std"])
    original = z.astype(float)*std+mean
    lower = np.full(len(mean), -np.inf)
    upper = np.full(len(mean), np.inf)
    if dataset == "smard_de":
        lower[:] = 0
    else:
        for i, c in enumerate(metadata["columns"]):
            if c.endswith("solar_radiation"):
                lower[i] = 0
            elif c.endswith("relative_humidity"):
                lower[i], upper[i] = 0, 100
    # Use the same standardized float32 representation as ground-truth windows.
    # A float64 inverse/forward round trip can introduce a tiny nonzero first
    # increment even when both physical observations are exactly zero.
    normalized_low = ((lower-mean)/std).astype(np.float32)
    normalized_high = ((upper-mean)/std).astype(np.float32)
    clipped = np.clip(z, normalized_low, normalized_high)
    return clipped, lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    seed_rows, pair_rows, phase_rows, gate_rows, training_rows, range_rows, gate_increments, clipped_increments = [], [], [], [], [], [], [], []
    grouped = {}
    for filename in sorted((root/"results_v2").glob("*/*/*_seed*/result.json")):
        case = filename.parent
        if not (case/"validation_lock.json").is_file():
            continue
        result = json.loads(filename.read_text())
        lock = json.loads((case/"validation_lock.json").read_text())
        dataset, year = case.parents[1].name, int(case.parent.name)
        context = {"dataset": dataset, "year": year, "backbone": result["backbone"], "seed": result["seed"]}
        with np.load(case/"predictions.npz") as ar:
            x, y, b = ar["test__x"], ar["test__y"], ar["test__base"]
            reference = stats(x, y, b)
            truth, base_labels = sign(x, y), sign(x, b)
            block = {}
            for method, scores in result["test"].items():
                z = ar["test__"+method]
                pair = np.c_[((sign(x, z)==truth).astype(float)-(base_labels==truth)).mean((1, 2)),
                             ((z.astype(float)-y)**2-(b.astype(float)-y)**2).mean((1, 2))]
                block[method] = pair
                seed_rows.append({**context, "method": method, **scores,
                                  "raw_gain_pp": 100*(scores["raw_accuracy"]-reference["raw_accuracy"]),
                                  "mse_change_pct": 100*(scores["mse"]/reference["mse"]-1),
                                  "joint_win": scores["raw_accuracy"]>reference["raw_accuracy"] and scores["mse"]<=reference["mse"]+1e-8})
                if method in ("raw_selected", "paired_gate", "alpha_1.0"):
                    alternative = sign(x, z)
                    fix = (alternative==truth)&(base_labels!=truth)
                    harm = (alternative!=truth)&(base_labels==truth)
                    for hour in [0, 6, 12, 18]:
                        m = ar["test__origins"]%24 == hour
                        phase_rows.append({**context, "method": method, "origin_hour_utc": hour,
                                           "windows": int(m.sum()), "fix_rate": float(fix[m].mean()),
                                           "harm_rate": float(harm[m].mean()), "gain_pp": float(100*pair[m, 0].mean()),
                                           "mse_delta": float(pair[m, 1].mean())})
            grouped.setdefault((dataset, year, result["backbone"]), []).append((block, ar["test__origins"].copy()))
            base_clipped, lo, hi = physical_clip(b, result["metadata"], dataset)
            physical_truth = pd.read_csv(result["metadata"]["csv"]).iloc[:, 1:].to_numpy(float)
            physical_truth = np.stack([physical_truth[t+1:t+25] for t in ar["test__origins"]])
            violations = int(np.sum((physical_truth < lo)|(physical_truth > hi)))
            clip_reference = stats(x, y, base_clipped)
            clipped_paths = {"base_clipped": base_clipped}
            for method in ["raw_selected", "paired_gate"]:
                clipped_paths[method+"_clipped"] = physical_clip(ar["test__"+method], result["metadata"], dataset)[0]
            np.savez_compressed(out/f"{dataset}_{year}_{result['backbone']}_seed{result['seed']}_ranges.npz", **clipped_paths)
            for method, clipped in clipped_paths.items():
                clip_scores = stats(x, y, clipped)
                range_rows.append({**context, "method": method, **clip_scores, "truth_range_violations": violations,
                                   "raw_gain_pp": 100*(clip_scores["raw_accuracy"]-reference["raw_accuracy"]),
                                   "mse_change_pct": 100*(clip_scores["mse"]/reference["mse"]-1),
                                   "label_changed_fraction": float(np.mean(sign(x, clipped)!=base_labels)),
                                   "declared_posthoc": True})
                if method != "base_clipped":
                    clipped_increments.append({**context, "method": method,
                                               "raw_gain_pp": 100*(clip_scores["raw_accuracy"]-clip_reference["raw_accuracy"]),
                                               "mse_change_pct": 100*(clip_scores["mse"]/clip_reference["mse"]-1),
                                               "joint_win": clip_scores["raw_accuracy"]>clip_reference["raw_accuracy"] and clip_scores["mse"]<=clip_reference["mse"]+1e-8})
            fixed_candidate = result["test"]["alpha_1.0"]
            gated = result["test"]["paired_gate"]
            gate_increments.append({**context, "raw_gain_pp": 100*(gated["raw_accuracy"]-fixed_candidate["raw_accuracy"]),
                                    "mse_change_pct": 100*(gated["mse"]/fixed_candidate["mse"]-1),
                                    "joint_win": gated["raw_accuracy"]>fixed_candidate["raw_accuracy"] and gated["mse"]<=fixed_candidate["mse"]+1e-8})
            raw_change = ar["test__alpha_1.0__paired_targets"]
            accepted = ar["test__gate_mask"]
            valid_choice = lock["gate_selection"]["chosen"]
            gate_rows.append({**context, "selected_alpha": lock["alpha"]["chosen"], "selected_gate": valid_choice,
                              "test_coverage": float(accepted.mean()), "accepted_windows": int(accepted.sum()),
                              "harmful_mse_fraction_of_accepted": float(np.mean(raw_change[accepted, 1]>0)) if accepted.any() else None,
                              "harmful_direction_fraction_of_accepted": float(np.mean(raw_change[accepted, 0]<0)) if accepted.any() else None,
                              "accepted_raw_gain_pp": float(100*raw_change[accepted, 0].mean()) if accepted.any() else None,
                              "accepted_mse_delta": float(raw_change[accepted, 1].mean()) if accepted.any() else None})
        training_rows.extend([{**context, **row} for row in result["training"]])
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(out/"per_seed.csv", index=False)
    summary = seed_frame.groupby(["dataset", "year", "backbone", "method"], sort=True).agg(
        raw_accuracy=("raw_accuracy", "mean"), mse=("mse", "mean"), raw_gain_pp=("raw_gain_pp", "mean"),
        mse_change_pct=("mse_change_pct", "mean"), joint_wins=("joint_win", "sum"),
        raw_nonzero_accuracy=("raw_nonzero_accuracy", "mean"), raw_zero_accuracy=("raw_zero_accuracy", "mean"),
        zero_fraction=("true_zero_fraction", "mean")).reset_index()
    summary.to_csv(out/"summary.csv", index=False)
    for (dataset, year, backbone), fitted in grouped.items():
        origins = fitted[0][1]
        if not all(np.array_equal(origins, v[1]) for v in fitted):
            raise ValueError("different paired origins across seeds")
        for method in fitted[0][0]:
            average = np.mean([v[0][method] for v in fitted], axis=0)
            lower, upper, blocks = time_interval(average, origins)
            pair_rows.append({"dataset": dataset, "year": year, "backbone": backbone, "method": method,
                              "seeds": len(fitted), "time_blocks": blocks, "gain_pp": 100*average[:, 0].mean(),
                              "gain_lower_pp": 100*lower[0], "gain_upper_pp": 100*upper[0], "mse_delta": average[:, 1].mean(),
                              "mse_lower": lower[1], "mse_upper": upper[1]})
    for name, rows in [("paired_intervals", pair_rows), ("origin_phases", phase_rows), ("gate_diagnostics", gate_rows),
                       ("training", training_rows), ("physical_ranges_posthoc", range_rows),
                       ("gate_increment_vs_alpha1", gate_increments), ("clipped_increment_vs_clipped_base", clipped_increments)]:
        pd.DataFrame(rows).to_csv(out/(name+".csv"), index=False)
    # A compact generated evidence table for direct document verification.
    lines = ["# 固定矩阵结果", "", "方向增益为百分点，MSE 为每 seed 相对其同训练 base 的百分比变化再取均值。", "",
             "| 数据/年/骨干 | 方法 | raw 增益 pp | MSE变化 % | 双指标 seed |", "|---|---|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        if row["method"] in ["cosdir", "tdalign", "residual", "raw_selected", "paired_gate", "carets3_cnn", "carets4_cnn"]:
            lines.append(f"| {row['dataset']}/{row['year']}/{row['backbone']} | {row['method']} | {row['raw_gain_pp']:+.4f} | {row['mse_change_pct']:+.4f} | {row['joint_wins']}/3 |")
    (out/"evidence.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"cases": len(seed_frame[seed_frame.method=="base"]), "summary_rows": len(summary), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
