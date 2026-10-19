"""Independent CSV/time/window audit of the TIME trajectory pilot (no training)."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np


def spec_for(root, name):
    shared = dict(grid_mode="native", lookback=96, horizon=16, train_stride=4,
                  evaluation_stride=2, frequency="15min", name=name)
    if name == "nem":
        shared.update(csv=str(root / "nem_5min.csv"), grid_mode="continuous", lookback=288,
                      horizon=24, train_stride=12, evaluation_stride=6, frequency="5min",
                      boundaries=["2025-01-01", "2025-04-01", "2025-04-15", "2025-05-01", "2025-05-15", "2025-06-01"])
    elif name == "carpark":
        shared.update(csv=str(root / "sg_carpark_15min.csv"), history_policy="causal_ffill",
                      boundaries=["2025-01-01", "2025-04-01", "2025-04-15", "2025-05-01", "2025-05-15", "2025-06-01"])
    else:
        shared.update(csv=str(root / "finland_traffic_15min.csv"),
                      boundaries=["2024-01-01", "2024-10-01", "2024-10-15", "2024-11-01", "2024-12-01", "2025-01-01"])
    return shared


def independent_csv(path):
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        rows = list(reader)
    times = [datetime.fromisoformat(row[0]) for row in rows]
    raw = np.array([[float(v) if v else np.nan for v in row[1:]] for row in rows], dtype=np.float64)
    return times, raw, columns[1:]


def amend_suffix(path, destination, start, missing_index=None):
    """Copy exact prefix bytes; do not reserialize training rows during tamper."""
    lines = Path(path).read_text().splitlines(keepends=True)
    if missing_index is not None:
        row = next(csv.reader([lines[missing_index + 1]]))
        lines[missing_index + 1] = row[0] + "," * (len(row) - 1) + "\n"
    else:
        for i in range(start + 1, len(lines)):
            row = next(csv.reader([lines[i]]))
            for j in range(1, len(row)):
                if row[j]:
                    row[j] = format(float(row[j]) + 10000 + 3 * (i % 17) + j, ".17g")
            buffer = io.StringIO()
            csv.writer(buffer, lineterminator="\n").writerow(row)
            lines[i] = buffer.getvalue()
    Path(destination).write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/lgc/TSLib")
    parser.add_argument("--frozen-source", default="/home/lgc/TSLib/DTL/trend_lab_runs/time_trajectory_20260930/source/v2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_root)
    frozen_source = Path(args.frozen_source)
    sys.path.insert(0, str(frozen_source))
    from trend_lab.trajectory_data import load_trajectory_data
    from trend_lab.trajectory_decision import raw_classes
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checks, failures, reports, bitwise_checks = [], [], {}, []

    def check(name, passed):
        checks.append(name)
        if not bool(passed):
            failures.append(name)

    def numerical_check(name, actual, expected):
        identical = bool(np.array_equal(actual, expected))
        difference = np.abs(actual - expected)
        bitwise_checks.append({"name": name, "bitwise_identical": identical,
                               "differing_elements": int((actual != expected).sum()),
                               "max_absolute_difference": float(difference.max(initial=0))})
        # Separate reported decimal-parser last-bit differences from semantic
        # failures. This is explicit numerical equivalence, never bit identity.
        limit = 32 * np.finfo(np.float64).eps * np.maximum(1, np.abs(expected))
        check(name + " numerical equivalence (32 float64 eps)", np.all(difference <= limit))

    data_root = repo / "DTL/research/data/time_validation_20260930/prepared"
    for name in ("nem", "carpark", "finland"):
        spec = spec_for(data_root, name)
        splits, meta, _ = load_trajectory_data(spec)
        times, raw, columns = independent_csv(spec["csv"])
        boundaries = [datetime.fromisoformat(v) for v in spec["boundaries"]]
        cuts = [sum(t < cut for t in times) for cut in boundaries]
        check(name + " independent cuts", cuts == meta["cuts"])
        check(name + " columns", columns == meta["columns"])
        train = raw[:cuts[1]]
        mean = np.array([math.fsum(v[np.isfinite(v)].tolist()) / np.isfinite(v).sum() for v in train.T])
        std = np.array([math.sqrt(math.fsum(((v[np.isfinite(v)] - m) ** 2).tolist()) / np.isfinite(v).sum())
                        for v, m in zip(train.T, mean)])
        check(name + " independent train means", np.allclose(mean, meta["mean"], rtol=1e-12, atol=1e-12))
        check(name + " independent train stds", np.allclose(std, meta["std"], rtol=1e-12, atol=1e-12))
        # Use the loader's train statistics only to compare its normalized array.
        normalized = (raw - np.array(meta["mean"])) / np.array(meta["std"])
        historical = normalized.copy()
        if spec.get("history_policy") == "causal_ffill":
            for channel in range(raw.shape[1]):
                previous = np.nan
                for index in range(len(raw)):
                    if np.isfinite(historical[index, channel]):
                        previous = historical[index, channel]
                    else:
                        historical[index, channel] = previous
        rows = {}
        target_sets = []
        L, H = spec["lookback"], spec["horizon"]
        for split_name, start, end in zip(splits, cuts[:-1], cuts[1:]):
            split = splits[split_name]
            stride = spec["train_stride"] if split_name == "train" else spec["evaluation_stride"]
            eligible = [i for i in range(0, len(raw), stride) if i >= L - 1 and i + 1 >= start and i + H + 1 <= end]
            expected = [i for i in eligible if np.isfinite(raw[i:i+H+1]).all() and
                        np.isfinite(historical[i-L+1:i+1]).all()]
            check(name + "/" + split_name + " independent kept origins", np.array_equal(split.origins, expected))
            check(name + "/" + split_name + " float64 arrays", split.x.dtype == split.y.dtype == np.float64)
            check(name + "/" + split_name + " finite x y", np.isfinite(split.x).all() and np.isfinite(split.y).all())
            check(name + "/" + split_name + " shapes", split.x.shape == (len(expected), L, raw.shape[1]) and split.y.shape == (len(expected), H, raw.shape[1]))
            expected_x = np.stack([historical[i-L+1:i+1] for i in expected])
            expected_y = np.stack([normalized[i+1:i+H+1] for i in expected])
            numerical_check(name + "/" + split_name + " independent historical fill", split.x, expected_x)
            numerical_check(name + "/" + split_name + " targets never filled", split.y, expected_y)
            numerical_check(name + "/" + split_name + " real last-history anchor", split.x[:, -1], normalized[expected])
            check(name + "/" + split_name + " float32 historical input bit identity", np.array_equal(split.x.astype(np.float32), expected_x.astype(np.float32)))
            check(name + "/" + split_name + " float32 target bit identity", np.array_equal(split.y.astype(np.float32), expected_y.astype(np.float32)))
            target_indices = set(i + h for i in expected for h in range(1, H + 1))
            target_sets.append(target_indices)
            check(name + "/" + split_name + " target index boundaries", min(target_indices) >= start and max(target_indices) < end)
            check(name + "/" + split_name + " target calendar boundaries", all(boundaries[list(splits).index(split_name)] <= times[i] < boundaries[list(splits).index(split_name)+1] for i in target_indices))
            true_classes = np.stack([np.where(raw[i+1:i+H+1] < raw[i:i+H], 0,
                                             np.where(raw[i+1:i+H+1] > raw[i:i+H], 2, 1)) for i in expected])
            check(name + "/" + split_name + " raw-sign and exact-zero labels preserved", np.array_equal(raw_classes(split.x, split.y), true_classes))
            expected_fill = np.mean(np.stack([~np.isfinite(raw[i-L+1:i+1]) for i in expected]))
            check(name + "/" + split_name + " fill ratio", expected_fill == meta["splits"][split_name]["filled_history_fraction"])
            rows[split_name] = {"eligible": len(eligible), "kept": len(expected),
                                "discarded": len(eligible)-len(expected), "history_filled_fraction": float(expected_fill),
                                "first_target": str(times[min(target_indices)]), "last_target": str(times[max(target_indices)]),
                                "origin_hour_counts": {str(h): int(sum(times[i].hour == h for i in expected)) for h in range(24)},
                                "origin_minute_values": sorted(set(times[i].minute for i in expected)),
                                "true_exact_zero_count": int((true_classes == 1).sum()), "true_direction_elements": int(true_classes.size)}
        check(name + " all five target sets disjoint", all(not (a & b) for i, a in enumerate(target_sets) for b in target_sets[i+1:]))
        with tempfile.TemporaryDirectory(prefix="trajectory_loader_audit_") as temporary:
            altered = Path(temporary) / "altered.csv"
            for start, unaffected in [(cuts[1], ["train"]), (cuts[-2], list(splits)[:-1])]:
                amend_suffix(spec["csv"], altered, start)
                edited = dict(spec, csv=str(altered))
                altered_splits, altered_meta, _ = load_trajectory_data(edited)
                for key in ("mean", "std", "train_min", "train_max", "native_integer_training"):
                    check(name + f" tamper from {start} unchanged " + key, altered_meta[key] == meta[key])
                for split_name in unaffected:
                    check(name + f" tamper from {start} unchanged " + split_name, all(np.array_equal(getattr(altered_splits[split_name], field), getattr(splits[split_name], field)) for field in ("origins", "x", "y")))
            if name == "carpark":
                strict = dict(spec, history_policy="complete")
                try:
                    load_trajectory_data(strict)
                    rejected = False
                except ValueError as error:
                    rejected = str(error) == "train insufficient complete windows: 0"
                check(name + " original all-complete rule rejects zero windows", rejected)
            lines = Path(spec["csv"]).read_text().splitlines(keepends=True)
            altered.write_text("".join(lines[:11] + lines[12:]))
            try:
                load_trajectory_data(dict(spec, csv=str(altered)))
                rejected = False
            except ValueError as error:
                rejected = "do not compress gaps" in str(error)
            check(name + " time compression rejected", rejected)
        reports[name] = {"spec": spec, "cuts": cuts, "splits": rows,
                         "raw_missing_by_channel": np.isnan(raw).sum(0).tolist()}
        if name == "carpark":
            valid = np.isfinite(raw).all(1)
            changes = np.diff(np.r_[False, valid, False].astype(int))
            starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
            reports[name]["longest_raw_complete_run"] = int(np.max(ends-starts))
            reports[name]["missing_time_counts"] = {time: sum((t.strftime("%H:%M") == time) and not good for t, good in zip(times, valid)) for time in sorted(set(t.strftime("%H:%M") for t, good in zip(times, valid) if not good))}
            reports[name]["coverage_limitation"] = "daily 00:00 missing targets exclude forecast windows crossing midnight; history ffill is past-only, not target imputation"
    sources = [frozen_source / "trend_lab/trajectory_data.py", frozen_source / "trend_lab/trajectory_decision.py", frozen_source / "protocol.md"]
    result = {"checks": len(checks), "failures": failures, "check_names": checks, "datasets": reports,
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "audit_method": "independent csv.reader/datetime/full-window enumeration/manual past-only fill/math.fsum stats, plus suffix tampering",
              "frozen_source": str(frozen_source.resolve()),
              "bitwise_comparisons": bitwise_checks,
              "bitwise_nonidentity_count": int(sum(not r['bitwise_identical'] for r in bitwise_checks)),
              "bitwise_limitation": "NEM pandas C-parser rounding is numerically equivalent but does not preserve every parsed decimal's final binary bit; original failed strict comparisons are retained",
              "no_training": True, "no_model_scores": True}
    (output / "data_loader_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"checks": len(checks), "failures": failures,
                      "bitwise_comparisons": len(bitwise_checks), "bitwise_nonidentity_count": result['bitwise_nonidentity_count'],
                      "counts": {n: {k: v["kept"] for k, v in r["splits"].items()} for n, r in reports.items()}}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
