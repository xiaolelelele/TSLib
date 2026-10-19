"""Bounded acquisition of preselected official recent hourly observations.

The data rules are fixed in recent_data_rolling_protocol_2026-09-30.md.
Raw downloads are cached verbatim and checksummed; missing hours remain NaN.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import requests

START = pd.Timestamp("2023-01-01", tz="UTC")
END = pd.Timestamp("2026-01-01", tz="UTC")
SMARD = {410: "load", 4068: "solar", 4067: "wind_onshore", 1225: "wind_offshore",
         1226: "hydro", 4066: "biomass", 4071: "gas", 1223: "lignite"}
STATIONS = ("CA_Bodega_6_WSW", "TN_Crossville_7_NW")
# Zero-based field indices; flags are only present for the raw quantities.
WEATHER = {"air_temperature": (9, None), "solar_radiation": (13, 14),
           "relative_humidity": (26, 27), "surface_temperature": (20, 21)}


class Fetcher:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records, self.lock, self.local = [], threading.Lock(), threading.local()
        self.bytes = 0

    def get(self, url, name):
        path = self.root / name
        if path.exists():
            data = path.read_bytes()
            downloaded = False
            stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        else:
            if not hasattr(self.local, "session"):
                self.local.session = requests.Session()
                self.local.session.headers["User-Agent"] = "TSLib bounded academic data acquisition"
            for attempt in range(3):
                try:
                    response = self.local.session.get(url, timeout=40)
                    response.raise_for_status()
                    data = response.content
                    if len(data) > 5_000_000:
                        raise ValueError("one-file download exceeds 5 MB budget")
                    with self.lock:
                        if self.bytes + len(data) > 60_000_000:
                            raise ValueError("download budget exceeded")
                        self.bytes += len(data)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)
            downloaded, stamp = True, datetime.now(timezone.utc).isoformat()
            time.sleep(.05)
        record = {"url": url, "file": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                  "bytes": len(data), "downloaded_now": downloaded, "fetched_utc": stamp}
        with self.lock:
            self.records.append(record)
        return data


def smard(fetcher):
    frames = []
    for filter_id, name in SMARD.items():
        prefix = f"https://www.smard.de/app/chart_data/{filter_id}/DE/"
        stamps = json.loads(fetcher.get(prefix + "index_hour.json", f"smard/{filter_id}_index.json"))["timestamps"]
        stamps = sorted(set(stamps))
        # Select intersecting chunks including the preceding hour interval.
        lower, upper = int((START - pd.Timedelta(hours=1)).timestamp() * 1000), int(END.timestamp() * 1000)
        selected = [s for i, s in enumerate(stamps) if s < upper and
                    (i + 1 == len(stamps) or stamps[i + 1] > lower)]

        def chunk(stamp):
            filename = f"{filter_id}_DE_hour_{stamp}.json"
            value = json.loads(fetcher.get(prefix + filename, "smard/" + filename))
            return value["series"]

        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = [row for values in pool.map(chunk, selected) for row in values]
        frame = pd.DataFrame(rows, columns=["timestamp", name])
        # SMARD hour records are labelled by interval start; use completed-hour time.
        frame["timestamp"] = pd.to_datetime(frame.timestamp, unit="ms", utc=True) + pd.Timedelta(hours=1)
        duplicates = frame[frame.timestamp.duplicated(False)]
        if len(duplicates) and duplicates.groupby("timestamp")[name].nunique(dropna=False).max() > 1:
            raise ValueError("conflicting SMARD timestamps")
        frame = frame.drop_duplicates("timestamp").set_index("timestamp")
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        frames.append(frame)
        print(f"SMARD {name}: {len(selected)} chunks", flush=True)
    return pd.concat(frames, axis=1)


def weather(fetcher):
    frames, types = [], {}
    base = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/"
    fetcher.get(base + "readme.txt", "noaa/readme.txt")
    for station in STATIONS:
        years, type_counts = [], {}
        for year in (2023, 2024, 2025):
            filename = f"CRNH0203-{year}-{station}.txt"
            data = fetcher.get(base + f"{year}/" + filename, "noaa/" + filename)
            lines = [line.split() for line in data.decode("ascii").splitlines() if line.strip()]
            if any(len(row) != 38 for row in lines):
                raise ValueError("USCRN hourly03 expected exactly 38 columns")
            fields = np.asarray(lines)
            times = pd.to_datetime([row[1] + row[2].zfill(4) for row in fields], format="%Y%m%d%H%M", utc=True)
            values = {}
            for name, (column, flag) in WEATHER.items():
                v = fields[:, column].astype(float)
                invalid = ~np.isfinite(v) | (v <= -999)
                if flag is not None:
                    invalid |= fields[:, flag].astype(float) != 0
                v[invalid] = np.nan
                values[station + "_" + name] = v
            unique, count = np.unique(fields[:, 19], return_counts=True)
            for kind, n in zip(unique, count):
                type_counts[kind] = type_counts.get(kind, 0) + int(n)
            years.append(pd.DataFrame(values, index=times))
        frame = pd.concat(years)
        if frame.index.has_duplicates:
            raise ValueError("duplicate USCRN hours")
        frames.append(frame)
        types[station] = type_counts
        print(f"NOAA {station}: {len(frame)} hours", flush=True)
    return pd.concat(frames, axis=1), types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(root / "raw")
    grid = pd.date_range(START, END, freq="h", inclusive="left", name="date")
    frames = {"smard_de": smard(fetcher)}
    frames["uscrn_two_station"], types = weather(fetcher)
    metadata = {"period": [str(START), str(END)], "expected_hours": len(grid), "datasets": {},
                "noaa_surface_temperature_types": types,
                "smard_attribution": "Bundesnetzagentur | SMARD.de", "smard_license": "CC BY 4.0",
                "availability": "retrospective archive, completed observations assumed available; not historical vintages"}
    for name, frame in frames.items():
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            frame = frame.sort_index()
        if not frame.index.is_unique:
            raise ValueError("nonunique time index")
        frame = frame.reindex(grid)
        path = root / f"{name}_2023_2025.csv"
        frame.to_csv(path, float_format="%.9g", date_format="%Y-%m-%dT%H:%M:%SZ")
        metadata["datasets"][name] = {"csv": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "channels": frame.columns.tolist(), "rows": len(frame),
                                      "missing_by_channel": frame.isna().sum().to_dict(),
                                      "complete_rows": int(frame.notna().all(axis=1).sum())}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (root / "download_manifest.json").write_text(json.dumps(sorted(fetcher.records, key=lambda r: r["file"]), indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
