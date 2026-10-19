"""Local and synthetic datasets; CSV smoke mode parses only a bounded prefix."""

import csv
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
import math
from pathlib import Path

import numpy as np

@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray
    origins: np.ndarray


@dataclass
class DatasetBundle:
    train: Split
    calibration: Split
    validation: Split
    test: Split
    mean: np.ndarray
    std: np.ndarray
    metadata: dict


def split_earlystop_calibration(bundle, earlystop_fraction=0.5):
    """Partition the old calibration interval into disjoint target intervals.

    Existing windows are retained verbatim. A window whose forecast targets
    cross the new boundary is discarded, even when the evaluation stride is
    smaller than the horizon. Observed histories may cross the boundary; the
    guarantee concerns target overlap, not statistical independence of a time
    series. The input bundle and the legacy four-way loader are unchanged.

    Returns ``(earlystop_split, calibration_split, metadata)``. Both splits must
    be nonempty: callers must provide sufficient data instead of silently
    reverting to shared early-stopping/calibration targets.
    """
    fraction = float(earlystop_fraction)
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("earlystop_fraction must lie strictly between 0 and 1")
    source = bundle.calibration
    start, end = map(int, bundle.metadata["split_bounds"][1:3])
    cut = start + int(math.floor((end - start) * fraction))
    horizon = source.y.shape[1]
    target_start = source.origins + 1
    target_end_exclusive = source.origins + horizon + 1
    if np.any(target_start < start) or np.any(target_end_exclusive > end):
        raise ValueError("calibration targets must lie inside their source interval")
    left = target_end_exclusive <= cut
    right = target_start >= cut
    if not left.any() or not right.any():
        raise ValueError("calibration segment is too short for disjoint earlystop/calibration targets")

    def select(mask):
        return Split(source.x[mask].copy(), source.y[mask].copy(), source.origins[mask].copy())

    metadata = {
        "source_bounds": [start, end], "cut_index": cut,
        "earlystop_bounds": [start, cut], "calibration_bounds": [cut, end],
        "earlystop_fraction": fraction,
        "earlystop_windows": int(left.sum()), "calibration_windows": int(right.sum()),
        "discarded_origins": source.origins[~(left | right)].tolist(),
        "target_intervals_disjoint": True,
    }
    return select(left), select(right), metadata


SCENARIOS = ("persistent_trend", "periodic_turn", "noisy_multiscale", "random_walk", "skewed_increments")


def _synthetic(config):
    length, channels = int(config["length"]), int(config["channels"])
    rng = np.random.default_rng(int(config["seed"]))
    noise = float(config.get("noise", 0.1))
    t = np.arange(length, dtype=np.float64)
    data = np.empty((length, channels), dtype=np.float64)
    phase = np.arange(channels) * 0.7
    kind = config["scenario"]
    for c in range(channels):
        if kind == "persistent_trend":
            drift = np.where((np.floor((t + phase[c]) / 96).astype(int) % 2) == 0, 0.12, -0.08)
            values = np.cumsum(drift + rng.normal(0, noise, length))
        elif kind == "periodic_turn":
            period = float(config.get("period", 96))
            if period <= 0:
                raise ValueError("period must be positive")
            values = np.sin(2 * np.pi * (t + phase[c]) / period) + rng.normal(0, noise, length)
        elif kind == "noisy_multiscale":
            slow_period = float(config.get("slow_period", 96))
            if slow_period <= 0:
                raise ValueError("slow_period must be positive")
            values = 0.018 * t + 0.8 * np.sin(2 * np.pi * (t + phase[c]) / slow_period)
            values += 0.18 * np.sin(2 * np.pi * t / 5 + phase[c]) + rng.normal(0, noise, length)
        elif kind == "random_walk":
            increments = rng.choice(np.array([-1.0, 1.0]), size=length)
            values = np.cumsum(increments + rng.normal(0, noise, length))
        elif kind == "skewed_increments":
            increments = np.where(rng.random(length) < 0.7, 1.0, -4.0)
            values = np.cumsum(increments + rng.normal(0, noise, length))
        else:
            raise ValueError(f"unknown synthetic scenario: {kind}")
        data[:, c] = values
    return data


def _read_csv(config, max_rows=None):
    path = Path(config["csv_path"])
    columns = config.get("value_columns")
    if not columns:
        raise ValueError("CSV input requires explicit value_columns")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        missing = [name for name in columns if name not in headers]
        time_column = config.get("time_column")
        if time_column and time_column not in headers:
            missing.append(time_column)
        if missing:
            raise ValueError(f"CSV columns not found: {missing}")
        rows = list(islice(reader, max_rows)) if max_rows is not None else list(reader)
        truncated = max_rows is not None and len(rows) == max_rows and next(reader, None) is not None
    if not rows:
        raise ValueError("CSV contains no data rows")
    if time_column:
        times = [row[time_column] for row in rows]
        try:
            parsed = [datetime.fromisoformat(value) for value in times]
        except (TypeError, ValueError):
            try:
                parsed = [float(value) for value in times]
            except (TypeError, ValueError) as exc:
                raise ValueError("time_column must contain ISO timestamps or numeric values") from exc
        if isinstance(parsed[0], (int, float)) and not np.isfinite(parsed).all():
            raise ValueError("numeric time_column values must be finite")
        if any(right <= left for left, right in zip(parsed, parsed[1:])):
            raise ValueError("time_column must be strictly increasing")
    try:
        values = np.asarray([[float(row[name]) for name in columns] for row in rows], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV value columns must contain only numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("CSV value columns contain missing or non-finite values")
    return values, {"truncated": bool(truncated), "row_limit": max_rows,
                    "source_rows": None if truncated else len(rows)}


def _windows(series, segment_start, segment_end, lookback, horizon, stride):
    origins = np.arange(segment_start - 1, segment_end - horizon, stride, dtype=np.int64)
    origins = origins[origins >= lookback - 1]
    if not len(origins):
        empty = np.empty((0, lookback, series.shape[1]), dtype=np.float32)
        return Split(empty, np.empty((0, horizon, series.shape[1]), dtype=np.float32), origins)
    x = np.stack([series[o - lookback + 1:o + 1] for o in origins])
    y = np.stack([series[o + 1:o + horizon + 1] for o in origins])
    return Split(x.astype(np.float32), y.astype(np.float32), origins)


def load_data(config):
    """Load data; CSV smoke mode reads a bounded prefix and records truncation."""
    required = ("kind", "channels", "lookback", "horizon", "split_ratios")
    absent = [key for key in required if key not in config]
    if absent:
        raise ValueError(f"missing data config keys: {absent}")
    kind = config["kind"]
    lookback, horizon = int(config["lookback"]), int(config["horizon"])
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    ratios = np.asarray(config["split_ratios"], dtype=np.float64)
    if ratios.shape != (4,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("split_ratios must contain four positive values summing to 1")
    train_stride = int(config.get("train_stride", 1))
    eval_stride = int(config.get("eval_stride", horizon))
    if train_stride < 1 or eval_stride < 1:
        raise ValueError("strides must be positive")
    read_metadata = {"truncated": False, "row_limit": None, "source_rows": None}
    if kind == "synthetic":
        absent = [key for key in ("length", "seed") if key not in config]
        if absent:
            raise ValueError(f"synthetic data requires config keys: {absent}")
        if config.get("scenario") not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        raw = _synthetic(config)
        read_metadata["source_rows"] = len(raw)
    elif kind == "csv":
        max_rows = None
        if config.get("smoke", False):
            split_rows = math.ceil((horizon + 1) / float(np.min(ratios[1:]))) + 4
            train_rows = lookback + horizon + train_stride - 1
            train_length = math.ceil(train_rows / float(ratios[0]))
            while math.floor(float(ratios[0]) * train_length) < train_rows:
                train_length += 1
            max_rows = max(180, split_rows, train_length)
        raw, read_metadata = _read_csv(config, max_rows=max_rows)
    else:
        raise ValueError("kind must be 'synthetic' or 'csv'")
    n, channels = raw.shape
    if channels != int(config["channels"]):
        raise ValueError("channels must equal the number of selected CSV value columns")
    if lookback < 1 or horizon < 1 or n <= lookback + horizon:
        raise ValueError("series must be longer than lookback + horizon")
    bounds = np.floor(np.r_[0, np.cumsum(ratios)] * n).astype(int)
    bounds[-1] = n
    mean = raw[bounds[0]:bounds[1]].mean(axis=0)
    std = raw[bounds[0]:bounds[1]].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    series = ((raw - mean) / std).astype(np.float32)
    splits = []
    names = ("train", "calibration", "validation", "test")
    for i, stride in enumerate((train_stride, eval_stride, eval_stride, eval_stride)):
        split = _windows(series, int(bounds[i]), int(bounds[i + 1]), lookback, horizon, stride)
        if not len(split.origins):
            raise ValueError(f"{names[i]} segment cannot contain a complete lookback/horizon window")
        splits.append(split)
    metadata = {
        "kind": kind, "scenario": config.get("scenario"), "length": int(n), "channels": int(channels),
        "lookback": lookback, "horizon": horizon, "split_bounds": bounds.tolist(),
        "split_ratios": ratios.tolist(),
        "train_stride": train_stride, "eval_stride": eval_stride,
        "smoke_truncated": read_metadata["truncated"], "smoke_row_limit": read_metadata["row_limit"],
        "source_rows": read_metadata["source_rows"], "rows_retained": int(n),
        "slow_period": float(config.get("slow_period", 96)) if config.get("scenario") == "noisy_multiscale" else None,
    }
    return DatasetBundle(*splits, mean.astype(np.float32), std.astype(np.float32), metadata)
