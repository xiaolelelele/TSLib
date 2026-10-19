"""Independent stdlib CSV audit of the locked acquisition (no model metrics)."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def read_rows(path, columns=None, inspect_values=True):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        time_column = header.index("timestamp" if "timestamp" in header else "date")
        chosen = columns if columns is not None else [c for i, c in enumerate(header) if i != time_column]
        indices = [header.index(c) for c in chosen]
        times, values, missing = [], [], []
        for row in reader:
            ts = datetime.fromisoformat(row[time_column])
            if ts.tzinfo is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            times.append(ts)
            tokens = [row[i].strip() for i in indices]
            missing.append([v.lower() in ("", "nan", "na", "null") for v in tokens])
            if inspect_values:
                values.append([np.nan if m else float(v) for v, m in zip(tokens, missing[-1])])
    return times, np.asarray(values, dtype=np.float64), np.asarray(missing, dtype=bool), chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    metadata = json.loads((root / "metadata.json").read_text())
    manifest = json.loads((root / "download_manifest.json").read_text())
    checks, failures, domains = [], [], {}

    def check(name, condition):
        checks.append(name)
        if not bool(condition):
            failures.append(name)

    for record in manifest:
        data = Path(record["file"]).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        check("hash " + record["relative_file"], digest == record["sha256"])
        check("size " + record["relative_file"], len(data) == record["bytes"])
        if record["expected_lfs_sha256"]:
            check("LFS " + record["relative_file"], digest == record["expected_lfs_sha256"])
    check("download budget", sum(r["bytes"] for r in manifest) <= metadata["max_download_bytes"])
    for name, entry in metadata["datasets"].items():
        core = name not in ("sg_pm25_hourly", "oil_price_business")
        times, original, missing, cols = read_rows(entry["source_csv"], entry["selected_channels"], core)
        prepared_times, prepared, prepared_missing, prepared_cols = read_rows(entry["prepared_csv"], inspect_values=core)
        check(name + " column order", cols == prepared_cols == entry["selected_channels"])
        check(name + " time order", all(a < b for a, b in zip(times, times[1:])))
        check(name + " unique time axis", len(set(times)) == len(times))
        check(name + " source dates retained", times == prepared_times)
        check(name + " missing mask exact", np.array_equal(missing, prepared_missing))
        check(name + " row counts", len(times) == entry["source_rows"] == entry["prepared_rows"])
        if entry["frequency"] == "B":
            cursor, grid = times[0], []
            while cursor <= times[-1]:
                if cursor.weekday() < 5:
                    grid.append(cursor)
                cursor += timedelta(days=1)
        else:
            seconds = {"5min": 300, "15min": 900, "h": 3600}[entry["frequency"]]
            grid = [times[0] + timedelta(seconds=seconds * i) for i in range(len(times))]
        check(name + " full regular axis", prepared_times == grid)
        check(name + " prepared hash", hashlib.sha256(Path(entry["prepared_csv"]).read_bytes()).hexdigest() == entry["prepared_sha256"])
        details = {"rows": len(times), "columns": cols, "missing": missing.sum(axis=0).tolist(),
                   "original_values_opened": core, "model_statistics_computed": False}
        if core:
            valid = ~missing
            difference = np.abs(original[valid] - prepared[valid])
            tolerance = 4 * np.finfo(np.float64).eps * np.maximum(1, np.abs(original[valid]))
            check(name + " original numeric values preserved", np.all(difference <= tolerance))
            check(name + " adjacent signs preserved", np.array_equal(
                np.sign(original[1:] - original[:-1])[valid[1:] & valid[:-1]],
                np.sign(prepared[1:] - prepared[:-1])[valid[1:] & valid[:-1]]))
            check(name + " exact flats preserved", np.array_equal(
                (original[1:] == original[:-1])[valid[1:] & valid[:-1]],
                (prepared[1:] == prepared[:-1])[valid[1:] & valid[:-1]]))
            details["max_absolute_export_difference"] = float(difference.max(initial=0))
            if name in ("sg_carpark_15min", "finland_traffic_15min"):
                check(name + " count integers preserved", np.all(original[valid] == np.rint(original[valid])))
            if name == "sg_carpark_15min":
                _, _, _, all_columns = read_rows(entry["source_csv"], inspect_values=False)
                check(name + " deterministic first eight columns", cols == all_columns[:8])
        domains[name] = details
    result = {"checks": len(checks), "failures": failures, "datasets": domains,
              "check_names": checks, "method": "independent csv.reader + datetime axis, no acquisition-code imports",
              "staged_holdout_policy": "PM25/Oil structural timestamps/missing mask only; no numeric values or target distributions inspected"}
    (root / "independent_acquisition_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
