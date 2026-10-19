"""Chronological explicit-date windows without deleting missing raw hours."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .data import Split


def load_recent(csv_path, year, lookback=96, horizon=24, stride=6):
    frame = pd.read_csv(csv_path)
    times = pd.to_datetime(frame.pop("date"), utc=True)
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("hourly dates must be strictly increasing")
    if not (times.diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("missing hours must be rows with NaN, not a compressed time axis")
    if times.iloc[0] != pd.Timestamp("2023-01-01", tz="UTC") or times.iloc[-1] != pd.Timestamp("2025-12-31T23:00:00Z"):
        raise ValueError("unexpected frozen data coverage")
    values = frame.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("infinite observations")
    boundaries = ["2023-01-01", f"{year}-01-01", f"{year}-03-01", f"{year}-04-01",
                  f"{year}-07-01", f"{year}-10-01", f"{year+1}-01-01"]
    cuts = [int(np.searchsorted(times.to_numpy(), pd.Timestamp(t, tz="UTC"))) for t in boundaries]
    training = values[:cuts[1]]
    mean, std = np.nanmean(training, axis=0), np.nanstd(training, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 1e-8):
        raise ValueError("training channel has no variation or valid data")
    normalized = ((values - mean) / std).astype(np.float32)
    valid = np.isfinite(normalized).all(axis=1)
    invalid_prefix = np.concatenate(([0], np.cumsum(~valid)))
    origins = np.flatnonzero((times.dt.hour.to_numpy() % stride) == 0)
    splits, split_metadata = {}, {}
    names = ["train", "earlystop", "calibration", "riskfit", "validation", "test"]
    for name, start, end in zip(names, cuts[:-1], cuts[1:]):
        eligible = origins[(origins >= lookback - 1) & (origins + 1 >= start) & (origins + horizon + 1 <= end)]
        keep = invalid_prefix[eligible + horizon + 1] == invalid_prefix[eligible - lookback + 1]
        chosen = eligible[keep]
        if len(chosen) < (300 if name == "train" else 50):
            raise ValueError(f"{name} has too few complete windows: {len(chosen)}")
        x = np.stack([normalized[t - lookback + 1:t + 1] for t in chosen])
        y = np.stack([normalized[t + 1:t + horizon + 1] for t in chosen])
        splits[name] = Split(x, y, chosen)
        unique, count = np.unique(times.iloc[chosen].dt.hour, return_counts=True)
        split_metadata[name] = {"target_bounds": [start, end], "target_dates": [boundaries[names.index(name)], boundaries[names.index(name)+1]],
                                "eligible": len(eligible), "kept": len(chosen), "discarded": int((~keep).sum()),
                                "origin_hours_utc": {str(h): int(n) for h, n in zip(unique, count)},
                                "first_origin": times.iloc[chosen[0]].isoformat(), "last_origin": times.iloc[chosen[-1]].isoformat()}
    metadata = {"csv": str(Path(csv_path).resolve()), "csv_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
                "year": year, "columns": frame.columns.tolist(), "mean": mean.tolist(), "std": std.tolist(),
                "cuts": cuts, "boundaries": boundaries, "splits": split_metadata,
                "lookback": lookback, "horizon": horizon, "stride": stride,
                "normalization": "per-channel finite training observations, float64 stats; float32 windows"}
    return splits, metadata
