"""Post-hoc descriptive diagnostics; never selects or retrains a method.

Uses saved v2 predictions only. Seed replications share the same test windows;
counts of improved channels/horizons are descriptive, not significance tests.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_repaired_experiment import labels


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(directory, include_time_blocks=False):
    result = json.loads((directory / "results.json").read_text())
    if result.get("complete") is not True or result.get("schema_version") != 2:
        raise ValueError("requires a completed v2 run: " + str(directory))
    cfg = result["config"]
    group = cfg.get("group", directory.name)
    scales, epsilon = cfg["scales"], result["epsilon"]
    arrays = np.load(directory / "predictions.npz", allow_pickle=False)
    x, y = arrays["test__history"].astype(float), arrays["test__truth"].astype(float)
    raw = lambda z: np.sign(np.diff(np.concatenate((x[:, -1:], z), axis=1), axis=1))
    true_raw, true_labels = raw(y), labels(x, y, scales, epsilon)
    values, channels, horizons, decisions, time_blocks = [], [], [], [], []
    block_ids = []
    if include_time_blocks:
        origins = np.asarray(arrays["test__origins"])
        if len(origins) != len(x) or len(x) < 4 or np.any(np.diff(origins) <= 0):
            raise ValueError("four time blocks require at least four chronologically ordered test windows")
        block_ids = np.array_split(np.arange(len(x)), 4)
    cols = cfg["data"].get("value_columns", list(map(str, range(y.shape[-1]))))
    for run in result["runs"]:
        seed = run["seed"]
        base = arrays["seed%s__test__base" % seed].astype(float)
        base_correct = raw(base) == true_raw
        base_errors = (base-y)**2
        probs = arrays["seed%s__test__direction_head__probs" % seed].astype(float)
        utility = lambda z: np.take_along_axis(probs, labels(x, z, scales, epsilon)[..., None], -1)[..., 0]
        base_utility = utility(base).mean()
        learned = arrays["seed%s__test__learned_selected" % seed].astype(float)
        learned_utility = utility(learned).mean()
        for method in run["methods"]:
            key = "seed%s__test__%s" % (seed, method)
            if key not in arrays.files:
                continue  # Classification-head row has no independent numeric path.
            z = arrays[key].astype(float)
            correct = raw(z) == true_raw
            fixes = (~base_correct) & correct
            harms = base_correct & (~correct)
            raw_gain = correct.mean() - base_correct.mean()
            if not np.isclose(raw_gain, fixes.mean()-harms.mean(), atol=1e-12):
                raise AssertionError("fix-minus-harm identity failed")
            eps_labels = labels(x, z, scales, epsilon)
            score = utility(z)
            costs = ((z-base)**2).mean(axis=(1, 2))
            values.append({"group": group, "seed": seed, "method": method,
                           "raw_gain_pp": 100*raw_gain,
                           "fixed_points_pct": 100*fixes.mean(),
                           "harmed_points_pct": 100*harms.mean(),
                           "sign_changed_pct": 100*(raw(z) != raw(base)).mean(),
                           "path_changed_windows_pct": 100*np.any(z != base, axis=(1, 2)).mean(),
                           "mean_squared_displacement": costs.mean(),
                           "predicted_class_utility": score.mean(),
                           "empirical_class_accuracy": (eps_labels == true_labels).mean(),
                           "predicted_utility_gain_vs_base_pp": 100*(score.mean()-base_utility),
                           "predicted_utility_gain_vs_learned_pp": 100*(score.mean()-learned_utility),
                           "marginal_mode_utility_gap_pp": 100*(probs.max(-1).mean()-score.mean()),
                           "class_score_minus_empirical_accuracy_pp": 100*(score.mean()-(eps_labels == true_labels).mean())})
            err = (z-y)**2
            for c, column in enumerate(cols):
                bm, cm = base_errors[:, :, c].mean(), err[:, :, c].mean()
                channels.append({"group": group, "seed": seed, "method": method, "column": column,
                                 "raw_gain_pp": 100*(correct[:, :, c].mean()-base_correct[:, :, c].mean()),
                                 "mse_delta": cm-bm, "mse_change_pct": 100*(cm/bm-1) if bm else None})
            for h in range(y.shape[1]):
                bm, cm = base_errors[:, h].mean(), err[:, h].mean()
                horizons.append({"group": group, "seed": seed, "method": method, "horizon": h+1,
                                 "raw_gain_pp": 100*(correct[:, h].mean()-base_correct[:, h].mean()),
                                 "mse_delta": cm-bm, "mse_change_pct": 100*(cm/bm-1) if bm else None})
            for block, ids in enumerate(block_ids, start=1):
                bm, cm = base_errors[ids].mean(), err[ids].mean()
                base_acc, method_acc = base_correct[ids].mean(), correct[ids].mean()
                time_blocks.append({
                    "group": group, "seed": seed, "method": method, "time_block": block,
                    "windows": len(ids), "origin_start": int(origins[ids[0]]),
                    "origin_end": int(origins[ids[-1]]),
                    "target_start": int(origins[ids[0]] + 1),
                    "target_end_inclusive": int(origins[ids[-1]] + y.shape[1]),
                    "base_raw_accuracy_pct": 100*base_acc,
                    "method_raw_accuracy_pct": 100*method_acc,
                    "raw_gain_pp": 100*(method_acc-base_acc),
                    "base_mse": bm, "method_mse": cm, "mse_delta": cm-bm,
                    "mse_ratio": cm/bm if bm > 0 else None,
                })
        choice = run["validation"]["choices"]["posterior_candidates"]
        diag = run["test_convergence"].get(choice, {})
        if choice == "base":
            counts = {"base": len(x)}
        else:
            library = ["base"] + ["projection_alpha_%g" % a for a in cfg["alpha_grid"][1:]] + [
                "projection_single_alpha_%g" % a for a in cfg["alpha_grid"][1:]]
            indices = np.asarray(diag["chosen_candidate_indices"], dtype=int)
            counts = {name: int((indices == i).sum()) for i, name in enumerate(library)}
        decisions.append({"group": group, "seed": seed, "posterior_validation_choice": choice,
                          "test_candidate_counts": counts,
                          "learned_validation_choice": run["validation"]["choices"]["learned_selected"]})
    original = values, channels, horizons, decisions
    return original + (time_blocks,) if include_time_blocks else original


def aggregate(rows, key_fields, metric_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in key_fields)].append(row)
    return [dict(zip(key_fields, key), **{
        metric: float(np.mean([r[metric] for r in members])) for metric in metric_fields})
        for key, members in groups.items()]


def summarize_time_blocks(rows):
    """Average seeds within each fixed block; do not treat blocks as replicates."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["group"], row["method"], row["time_block"])].append(row)
    summary = []
    bounds = ("windows", "origin_start", "origin_end", "target_start", "target_end_inclusive")
    metrics = ("base_raw_accuracy_pct", "method_raw_accuracy_pct", "raw_gain_pp",
               "base_mse", "method_mse", "mse_delta")
    for (group, method, block), members in groups.items():
        if any(len({r[key] for r in members}) != 1 for key in bounds):
            raise ValueError("seed time blocks must share window counts and target intervals")
        row = {"group": group, "method": method, "time_block": block,
               **{key: members[0][key] for key in bounds}, "seed_replicates": len(members),
               **{key: float(np.mean([r[key] for r in members])) for key in metrics}}
        ratios = [r["mse_ratio"] for r in members if r["mse_ratio"] is not None]
        row["mean_seed_mse_ratio"] = float(np.mean(ratios)) if ratios else None
        row["mse_ratio_defined_seeds"] = len(ratios)
        row["ratio_of_seed_mean_mse"] = row["method_mse"] / row["base_mse"] if row["base_mse"] > 0 else None
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    values, channels, horizons, decisions, time_blocks = [], [], [], [], []
    for run in args.runs:
        a, b, c, d, blocks = analyze(run, include_time_blocks=True)
        values.extend(a); channels.extend(b); horizons.extend(c); decisions.extend(d)
        time_blocks.extend(blocks)
    write_csv(args.output / "per_seed.csv", values)
    write_csv(args.output / "per_channel.csv", channels)
    write_csv(args.output / "per_horizon.csv", horizons)
    write_csv(args.output / "per_time_block.csv", time_blocks)
    write_csv(args.output / "time_block_summary.csv", summarize_time_blocks(time_blocks))
    summary = aggregate(values, ("group", "method"), list(values[0])[3:])
    channel_mean = aggregate(channels, ("group", "method", "column"), ("raw_gain_pp", "mse_delta"))
    horizon_mean = aggregate(horizons, ("group", "method", "horizon"), ("raw_gain_pp", "mse_delta"))
    for row in summary:
        for prefix, source in (("channels", channel_mean), ("horizons", horizon_mean)):
            members = [r for r in source if all(r[k] == row[k] for k in ("group", "method"))]
            row[prefix+"_raw_improved"] = sum(r["raw_gain_pp"] > 0 for r in members)
            row[prefix+"_joint_improved"] = sum(r["raw_gain_pp"] > 0 and r["mse_delta"] <= 1e-8 for r in members)
            row[prefix+"_total"] = len(members)
    write_csv(args.output / "summary.csv", summary)
    (args.output / "decisions.json").write_text(json.dumps(decisions, indent=2)+"\n")
    (args.output / "README.md").write_text(
        "# Post-hoc descriptive mechanism analysis\n\n"
        "No method selection, retraining, or test tuning. Each method is the frozen validation-selected path. "
        "Channel/horizon counts average seeds on shared test windows and are not significance tests. "
        "Fix/harm percentages use all forecast points as denominator. Candidate class utilities use "
        "the learned marginal probabilities, so their gap to realized accuracy diagnoses score quality, "
        "not a bound on true accuracy. The marginal-mode utility bound is only for the learned additive "
        "score; it is not an empirical accuracy ceiling. This script does not estimate MSE risk guarantees.\n\n"
        "Time blocks are four fixed consecutive groups from np.array_split over test-window indices, "
        "not blocks selected after inspecting their gains. per_time_block.csv records each seed and "
        "method, window count, global origin endpoints, and inclusive forecast-target range. Its "
        "accuracy and MSE average all window/horizon/channel points inside that block. MSE is in "
        "training-standardized units. Block counts can differ by one window: reconstruct overall "
        "results with window-count weights rather than an unweighted block average. Target ranges "
        "can overlap if the evaluation stride is shorter than the forecast horizon.\n\n"
        "time_block_summary.csv averages seeds within the same block. mean_seed_mse_ratio averages "
        "defined per-seed ratios; ratio_of_seed_mean_mse divides the seed-mean method MSE by the "
        "seed-mean base MSE. A zero base MSE gives an undefined ratio, recorded as an empty CSV cell; "
        "mse_ratio_defined_seeds reports how many seed ratios were available. Four blocks and their "
        "shared-window seed repetitions are descriptive strata, not independent experimental "
        "replicates or significance tests. These summaries do not change validation selections, "
        "establish the cause of validation/test differences, or attribute failure to origin phase.\n")
    print("Saved descriptive mechanism diagnostics to", args.output)


if __name__ == "__main__":
    main()
