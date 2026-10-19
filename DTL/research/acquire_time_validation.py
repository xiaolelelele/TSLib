"""Acquire a bounded, named TIME subset and retain unchanged source bytes.

The first tranche is selected from filenames/column IDs before any training.
No interpolation, row deletion, or target filling is performed. The HF mirror
is only a transport fallback; LFS SHA256 verifies every CSV against the locked
repository's tree metadata. Originals, cards, and source configuration remain.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import requests

PROCESSED_REPO = "Real-TSF/TIME-ProcessedCSV"
PROCESSED_REV = "683bb21b470f5a26586e1ab2295f0a2a1529ef0f"
TIME_REPO = "Real-TSF/TIME"
TIME_REV = "83e3d0b3be28d11c7182bffcc1892d19b36c4da1"
CODE_REV = "3ca5c41d71c76f3c70632c0121142d325350ee32"
DATASETS = {
    "nem_5min": ("OpenElectricity_NEM/5T/item0.csv", "5min"),
    "finland_traffic_15min": ("Finland_Traffic/15T/1.csv", "15min"),
    "sg_carpark_15min": ("SG_Carpark/15T/item0.csv", "15min"),
    "sg_pm25_hourly": ("SG_PM25/H/item0.csv", "h"),
    "oil_price_business": ("Oil_Price/B/item0.csv", "B"),
}


class Fetcher:
    def __init__(self, root, max_bytes):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.bytes = 0
        self.records = []
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "TSLib bounded academic validation"

    def get(self, official_url, relative, expected_sha=None, transport_url=None):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        url = transport_url or official_url
        was_cached = path.exists()
        if not was_cached:
            for attempt in range(3):
                try:
                    response = self.session.get(url, stream=True, timeout=(15, 45))
                    response.raise_for_status()
                    expected_size = int(response.headers.get("Content-Length", 0))
                    if expected_size and self.bytes + expected_size > self.max_bytes:
                        raise ValueError("bounded total download budget would be exceeded")
                    parts, file_bytes = [], 0
                    for part in response.iter_content(chunk_size=1024 * 1024):
                        file_bytes += len(part)
                        if self.bytes + file_bytes > self.max_bytes:
                            raise ValueError("bounded total download budget exceeded")
                        parts.append(part)
                    data = b"".join(parts)
                    digest = hashlib.sha256(data).hexdigest()
                    if expected_sha and digest != expected_sha:
                        raise ValueError("download does not match locked LFS SHA256")
                    path.write_bytes(data)
                    self.bytes += len(data)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        else:
            data = path.read_bytes()
            self.bytes += len(data)
            if self.bytes > self.max_bytes:
                raise ValueError("cached inputs exceed bounded total budget")
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha and digest != expected_sha:
            raise ValueError("cached file does not match locked LFS SHA256")
        stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        self.records.append({"official_url": official_url, "transport_url": url,
                             "file": str(path.resolve()), "relative_file": relative,
                             "sha256": digest, "bytes": len(data), "fetched_utc": stamp,
                             "cached": was_cached, "expected_lfs_sha256": expected_sha})
        print("saved", relative, len(data), digest, flush=True)
        return data


def hf_url(repo, rev, path, mode="resolve", mirror=False):
    host = "hf-mirror.com" if mirror else "huggingface.co"
    return f"https://{host}/datasets/{repo}/{mode}/{rev}/{path}"


def channel_profile(frame):
    profile = {}
    for name in frame:
        v = frame[name].to_numpy(dtype=np.float64)
        finite = np.isfinite(v)
        paired = finite[1:] & finite[:-1]
        differences = v[1:][paired] - v[:-1][paired]
        profile[str(name)] = {
            "missing": int((~finite).sum()), "finite": int(finite.sum()),
            "min": float(v[finite].min()) if finite.any() else None,
            "max": float(v[finite].max()) if finite.any() else None,
            "negative_count": int((v[finite] < 0).sum()),
            "noninteger_finite_count": int((v[finite] != np.rint(v[finite])).sum()),
            "valid_adjacent_pairs": int(paired.sum()),
            "adjacent_exact_zero_count": int((differences == 0).sum()),
            "adjacent_exact_zero_fraction": float((differences == 0).mean()) if len(differences) else None,
            "observed_min_nonzero_adjacent_change": float(np.abs(differences[differences != 0]).min())
                if (differences != 0).any() else None,
        }
    return profile


def prepare(name, csv_path, freq, root):
    raw = pd.read_csv(csv_path)
    timestamp_column = next((c for c in raw if str(c).lower() in ("timestamp", "date", "datetime", "time")), None)
    if timestamp_column is None:
        raise ValueError(f"no recognized timestamp column in {csv_path}: {raw.columns.tolist()}")
    raw_times = raw[timestamp_column].astype(str)
    parsed = pd.DatetimeIndex(pd.to_datetime(raw_times, errors="raise"))
    if parsed.tz is not None:
        # Preserve absolute instants if the CSV explicitly supplies a zone.
        parsed = parsed.tz_convert("UTC").tz_localize(None)
        time_note = "original timezone-aware timestamps converted to UTC then stored naive UTC"
    else:
        time_note = "source CSV naive timestamps retained verbatim; source timezone unverified"
    if parsed.has_duplicates:
        raise ValueError(f"duplicate timestamps in {name}; no automatic collapse permitted")
    if not parsed.is_monotonic_increasing:
        raise ValueError(f"timestamps not increasing in {name}; no implicit sorting permitted")
    values = raw.drop(columns=[timestamp_column]).apply(pd.to_numeric, errors="raise")
    if np.isinf(values.to_numpy(dtype=np.float64)).any():
        raise ValueError(f"infinite observations in {name}")
    original_columns = [str(c) for c in values]
    if name == "sg_carpark_15min":
        # A single wide file: the deterministic rule is numeric column-ID order.
        def numeric_key(column):
            match = re.fullmatch(r"(?:item)?(\d+)", str(column))
            return int(match.group(1)) if match else None
        keys = [numeric_key(c) for c in values]
        if all(k is not None for k in keys):
            selected = sorted(values.columns, key=numeric_key)[:8]
            selection_rule = "first eight numeric column IDs in ascending order"
        else:
            selected = list(values.columns)[:8]
            selection_rule = "first eight columns in original source-file order (IDs are not all numeric)"
        values = values.loc[:, selected]
    else:
        selection_rule = "all source-file channels; no performance filtering"
    values.index = parsed
    grid = pd.date_range(parsed[0], parsed[-1], freq=freq, name="date")
    off_grid = parsed.difference(grid)
    if len(off_grid):
        raise ValueError(f"{len(off_grid)} off-grid timestamps in {name}; no rounding permitted")
    # Retain every regular time slot, inserting NaN where source rows are absent.
    prepared = values.reindex(grid)
    output = root / "prepared" / f"{name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, date_format="%Y-%m-%d %H:%M:%S", float_format="%.17g")
    deltas = pd.Series(parsed[1:] - parsed[:-1]).value_counts()
    if name in ("sg_pm25_hourly", "oil_price_business"):
        # Staged domains: inspect structural missingness only before advancement.
        original_profile = {str(c): {"missing": int(values[c].isna().sum())} for c in values}
        profile = {str(c): {"missing": int(prepared[c].isna().sum())} for c in prepared}
    else:
        original_profile = channel_profile(pd.DataFrame(values.to_numpy(), index=parsed, columns=values.columns))
        profile = channel_profile(prepared)
    return {
        "source_csv": str(csv_path.resolve()), "prepared_csv": str(output.resolve()),
        "prepared_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_rows": len(raw), "prepared_rows": len(prepared), "frequency": freq,
        "first_timestamp": str(parsed[0]), "last_timestamp": str(parsed[-1]),
        "original_first_timestamp_string": raw_times.iloc[0],
        "original_last_timestamp_string": raw_times.iloc[-1],
        "timestamp_column": str(timestamp_column), "timezone_handling": time_note,
        "source_all_channels": original_columns, "selected_channels": [str(c) for c in prepared],
        "selection_rule": selection_rule, "inserted_missing_time_slots": len(grid.difference(parsed)),
        "original_timestamp_delta_counts": {str(k): int(v) for k, v in deltas.items()},
        "channel_quality": profile, "source_selected_channel_quality": original_profile,
        "missing_mask_available": True, "official_cleaning_modified_mask_available": False,
        "zero_semantics": "exact equal adjacent exported numeric values; may mix real plateaus and upstream cleaning",
        "no_imputation": True, "no_time_compression": True,
        "business_day_note": "weekends are not inserted for B frequency; missing weekday holidays remain NaN" if freq == "B" else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-mb", type=float, default=80)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    fetch = Fetcher(root / "raw", int(args.max_mb * 1_000_000))
    for repo, rev, label in [(PROCESSED_REPO, PROCESSED_REV, "processed"), (TIME_REPO, TIME_REV, "time")]:
        official = f"https://huggingface.co/api/datasets/{repo}/revision/{rev}"
        mirrored = official.replace("huggingface.co", "hf-mirror.com")
        fetch.get(official, f"provenance/{label}_repository_metadata.json", transport_url=mirrored)
        fetch.get(hf_url(repo, rev, "README.md"), f"provenance/{label}_README.md",
                  transport_url=hf_url(repo, rev, "README.md", mirror=True))
    for path in ["README.md", "docs/PREPROCESS.md", "docs/FEATURES.md", "src/timebench/config/datasets.yaml", "src/timebench/preprocess.py"]:
        url = f"https://raw.githubusercontent.com/zqiao11/TIME/{CODE_REV}/{path}"
        fetch.get(url, "provenance/TIME_source/" + path)
    datasets = {}
    for name, (path, freq) in DATASETS.items():
        directory = path.rsplit("/", 1)[0]
        tree_official = f"https://huggingface.co/api/datasets/{PROCESSED_REPO}/tree/{PROCESSED_REV}/{directory}"
        tree = json.loads(fetch.get(tree_official, f"provenance/tree_{name}.json",
                                    transport_url=tree_official.replace("huggingface.co", "hf-mirror.com")))
        record = next(v for v in tree if v.get("path") == path)
        digest = record.get("lfs", {}).get("oid")
        if not digest:
            raise ValueError(f"no expected SHA256 in locked official tree for {path}")
        fetch.get(hf_url(PROCESSED_REPO, PROCESSED_REV, path), "data/" + path,
                  expected_sha=digest, transport_url=hf_url(PROCESSED_REPO, PROCESSED_REV, path, mirror=True))
        datasets[name] = prepare(name, root / "raw" / "data" / path, freq, root)
        (root / "metadata.json").write_text(json.dumps({"datasets": datasets}, indent=2), encoding="utf-8")
        (root / "download_manifest.json").write_text(json.dumps(fetch.records, indent=2), encoding="utf-8")
        print("prepared", name, json.dumps(datasets[name]), flush=True)
    metadata = {
        "acquisition_client_date": "2026-09-30", "processed_repository": PROCESSED_REPO,
        "processed_revision": PROCESSED_REV, "time_revision": TIME_REV,
        "official_code_revision": CODE_REV, "max_download_bytes": fetch.max_bytes,
        "actual_download_bytes": fetch.bytes, "datasets": datasets,
        "transport": "official huggingface.co local TLS resets; public hf-mirror.com fallback with locked LFS SHA256",
        "license": {
            "TIME_card": "CC BY-NC 4.0", "ProcessedCSV_card": "Apache 2.0",
            "policy": "do not override more restrictive upstream and TIME terms with ProcessedCSV label; academic non-commercial use",
            "sources": ["https://huggingface.co/datasets/Real-TSF/TIME", "https://huggingface.co/datasets/Real-TSF/TIME-ProcessedCSV"],
        },
        "retrospective_availability": "exported historical observations, not historical vintage or publication-delay replay",
        "cleaning_mask": "not included in selected official CSVs; upstream replacements may generate flats",
        "selection_policy": "named five domains before training; all NEM/Finland/PM25/Oil channels, first 8 Carpark IDs by file/column order",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (root / "download_manifest.json").write_text(json.dumps(fetch.records, indent=2), encoding="utf-8")
    print("COMPLETE", fetch.bytes, len(fetch.records), flush=True)


if __name__ == "__main__":
    main()
