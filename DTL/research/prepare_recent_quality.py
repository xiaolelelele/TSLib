"""Pre-training coverage amendment; retain the originally acquired files."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent / "data/recent_hourly_20260930"
    original = root / "uscrn_two_station_2023_2025.csv"
    frame = pd.read_csv(original)
    columns = [c for c in frame if c.startswith("TN_Crossville_7_NW_")]
    if len(columns) != 4:
        raise ValueError("expected exactly four preselected Crossville variables")
    path = root / "uscrn_crossville_2023_2025.csv"
    frame[["date"] + columns].to_csv(path, index=False, float_format="%.9g")
    times = pd.to_datetime(frame.date, utc=True)
    diagnostics = {}
    for label, cols in [("two_station", list(frame.columns[1:])), ("crossville", columns)]:
        valid = np.isfinite(frame[cols].to_numpy(float)).all(1)
        prefix = np.r_[0, np.cumsum(~valid)]
        years = {}
        for year in [2024, 2025]:
            boundaries = ["2023-01-01", f"{year}-01-01", f"{year}-03-01", f"{year}-04-01",
                          f"{year}-07-01", f"{year}-10-01", f"{year+1}-01-01"]
            cuts = [times.searchsorted(pd.Timestamp(v, tz="UTC")) for v in boundaries]
            origins = np.flatnonzero(times.dt.hour.to_numpy()%6 == 0)
            rows = {}
            for name, start, end in zip(["train", "earlystop", "calibration", "riskfit", "validation", "test"], cuts[:-1], cuts[1:]):
                eligible = origins[(origins >= 95) & (origins+1 >= start) & (origins+25 <= end)]
                kept = prefix[eligible+25] == prefix[eligible-95]
                rows[name] = {"eligible": len(eligible), "kept": int(kept.sum()), "discarded": int((~kept).sum())}
            years[year] = rows
        diagnostics[label] = years
    record = {"reason": "13 complete calibration windows for joint two-station 2024 March before any training",
              "parent": str(original), "parent_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
              "derived": str(path), "derived_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "columns": columns, "coverage": diagnostics}
    (root / "quality_amendment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
