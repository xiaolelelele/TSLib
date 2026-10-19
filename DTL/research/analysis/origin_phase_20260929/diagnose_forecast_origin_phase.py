"""Describe forecast-origin sampling phases from saved splits and CSV dates.

Reads timestamps and saved origins, not target values. This is a post-hoc
protocol diagnostic; it neither changes the experiment nor identifies a cause
of any performance difference.
"""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


def analyze(root):
    result = json.loads((root / "results.json").read_text())
    cfg = result["config"]["data"]
    with Path(cfg["csv_path"]).open(newline="", encoding="utf-8-sig") as stream:
        timestamps = [datetime.fromisoformat(r[cfg["time_column"]]) for r in csv.DictReader(stream)]
    gaps = Counter((b-a).total_seconds() for a, b in zip(timestamps, timestamps[1:]))
    arrays = np.load(root / "predictions.npz", allow_pickle=False)
    horizon = cfg["horizon"]
    rows = []
    for split in ("earlystop", "calibration", "validation", "test"):
        origins = arrays[split + "__origins"].astype(int)
        origin_dates = [timestamps[i] for i in origins]
        target_dates = [timestamps[i+h] for i in origins for h in range(1, horizon+1)]
        rows.append({"split": split, "windows": len(origins),
                     "first_origin_index": int(origins[0]), "last_origin_index": int(origins[-1]),
                     "first_origin_time": origin_dates[0].isoformat(),
                     "last_origin_time": origin_dates[-1].isoformat(),
                     "origin_hour_counts": dict(sorted(Counter(t.hour for t in origin_dates).items())),
                     "first_future_hour_counts": dict(sorted(Counter(timestamps[i+1].hour for i in origins).items())),
                     "all_forecast_target_hour_counts": dict(sorted(Counter(t.hour for t in target_dates).items()))})
    return {"group": result["config"].get("group", root.name), "source": cfg["csv_path"],
            "consecutive_timestamp_gap_seconds": dict(sorted(gaps.items())),
            "eval_stride": cfg["eval_stride"], "horizon": horizon, "splits": rows,
            "interpretation": "Fixed-hour origins can differ across splits even when H=24 covers all "
                              "target hours. This restricts origin-phase generalization and may affect "
                              "selection transfer, but is not evidence of leakage or causal attribution."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump([analyze(root) for root in args.runs], stream, indent=2)
        stream.write("\n")
