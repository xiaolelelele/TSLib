#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dtl_root="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${TSLIB_TREND_PYTHON:-/home/lgc/.conda/envs/TFB/bin/python}"
gpu_weather="${TSLIB_TREND_GPU_WEATHER:-2}"
gpu_exchange="${TSLIB_TREND_GPU_EXCHANGE:-3}"
fusion_parent="${TSLIB_TREND_FUSION_RUNS_DIR:-$dtl_root/trend_lab_fusion_runs}"
fusion_root="${TSLIB_TREND_FUSION_RUN_ROOT:-$fusion_parent/fusion_validation_$(date -u +%Y%m%dT%H%M%SZ)}"
library_root="$dtl_root/../TSLib/Time-Series-Library"
weather_csv="$library_root/dataset/weather/weather.csv"
exchange_csv="$library_root/dataset/exchange_rate/exchange_rate.csv"
reference_dlinear="$library_root/models/DLinear.py"
reference_decomp="$library_root/layers/Autoformer_EncDec.py"
protocol_path="$dtl_root/research/trend_fusion_validation_protocol_2026-09-27.md"

if [[ ! -x "$python_bin" ]]; then
  echo "Python executable is unavailable: $python_bin" >&2
  exit 2
fi
for required_path in "$weather_csv" "$exchange_csv" "$reference_dlinear" "$reference_decomp" "$protocol_path"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Required fusion input is missing: $required_path" >&2
    exit 2
  fi
done
if [[ "$gpu_weather" != "2" || "$gpu_exchange" != "3" ]]; then
  echo "This authorized run is fixed to Weather on GPU2 and ExchangeRate on GPU3" >&2
  exit 2
fi
mkdir -p "$fusion_root"
if [[ -n "$(find "$fusion_root" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Fusion run directory is not empty: $fusion_root" >&2
  exit 2
fi
mkdir -p "$fusion_root/prepared" "$fusion_root/configs" "$fusion_root/logs" \
  "$fusion_root/source/trend_lab/configs" "$fusion_root/source/trend_lab/tests" \
  "$fusion_root/source/trend_lab/scripts" \
  "$fusion_root/reference/Time-Series-Library/models" \
  "$fusion_root/reference/Time-Series-Library/layers"

run_stage="initialized"
record_exit_status() {
  local status=$?
  trap - EXIT
  {
    printf 'exit_code=%s\n' "$status"
    printf 'last_stage=%s\n' "$run_stage"
    printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$fusion_root/exit_status.txt"
  exit "$status"
}
trap record_exit_status EXIT

# Config template paths below are package-relative.
cd "$dtl_root"

export FUSION_ROOT="$fusion_root"
export WEATHER_RAW_CSV="$weather_csv"
export WEATHER_PREPARED_CSV="$fusion_root/prepared/weather_deduplicated.csv"
export EXCHANGE_RAW_CSV="$exchange_csv"
export PYTHONUNBUFFERED=1

echo "Fusion run root: $fusion_root"
echo "Preflighting both full CSV files and preparing the ExchangeRate date-only copy"
run_stage="preflight"
"$python_bin" - <<'PY' 2>&1 | tee "$fusion_root/logs/data_preflight.log"
import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np

root = Path(os.environ["FUSION_ROOT"])
weather_path = Path(os.environ["WEATHER_RAW_CSV"])
derived_weather_path = Path(os.environ["WEATHER_PREPARED_CSV"])
exchange_path = Path(os.environ["EXCHANGE_RAW_CSV"])
derived_exchange_path = root / "prepared" / "exchange_rate_iso.csv"


def parse_time(value, mode):
    if mode == "exchange_slash":
        return datetime.strptime(value, "%Y/%m/%d %H:%M")
    return datetime.fromisoformat(value)


def derive_exchange_dates(source, destination):
    with source.open("r", newline="", encoding="utf-8-sig") as in_stream, \
         destination.open("w", newline="", encoding="utf-8") as out_stream:
        reader = csv.reader(in_stream)
        writer = csv.writer(out_stream, lineterminator="\n")
        header = next(reader)
        writer.writerow(header)
        count = 0
        for row in reader:
            if not row or len(row) != len(header):
                raise ValueError(f"Malformed ExchangeRate row {count + 2}: expected {len(header)} fields")
            converted = parse_time(row[0], "exchange_slash").isoformat(sep=" ")
            writer.writerow([converted, *row[1:]])
            count += 1
    return count


def derive_weather_deduplicated(source, destination):
    dropped = []
    interval_anomalies = []
    with source.open("r", newline="", encoding="utf-8-sig") as in_stream, \
         destination.open("w", newline="", encoding="utf-8") as out_stream:
        reader = csv.reader(in_stream)
        writer = csv.writer(out_stream, lineterminator="\n")
        header = next(reader)
        if not header or header[0] != "date" or len(set(header)) != len(header):
            raise ValueError(f"Expected a unique Weather header beginning with 'date': {source}")
        writer.writerow(header)
        previous_time = None
        previous_values = None
        previous_kept_source_index = None
        kept_count = 0
        for source_row_index, row in enumerate(reader):
            if len(row) != len(header):
                raise ValueError(f"Malformed Weather row {source_row_index + 2}: expected {len(header)} fields")
            current_time = parse_time(row[0], "iso")
            if previous_time is not None:
                delta = (current_time - previous_time).total_seconds()
                if delta < 0:
                    raise ValueError(f"Weather time goes backwards at CSV line {source_row_index + 2}")
                if delta == 0:
                    if row[1:] != previous_values:
                        raise ValueError(
                            f"Weather has equal timestamps with different numeric strings at CSV line "
                            f"{source_row_index + 2}; refusing to choose a row"
                        )
                    row_hash = hashlib.sha256(
                        json.dumps(row[1:], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()
                    dropped.append({
                        "source_data_row_index_zero_based": source_row_index,
                        "source_csv_line_number_one_based_with_header": source_row_index + 2,
                        "date_original": row[0],
                        "numeric_strings_identical_to_previous_kept_row": True,
                        "numeric_strings_sha256": row_hash,
                    })
                    continue
                if delta != 600:
                    interval_anomalies.append({
                        "from_source_data_row_index_zero_based": previous_kept_source_index,
                        "to_source_data_row_index_zero_based": source_row_index,
                        "from_date": previous_time.isoformat(sep=" "),
                        "to_date": current_time.isoformat(sep=" "),
                        "delta_seconds": delta,
                    })
            writer.writerow(row)
            previous_time = current_time
            previous_values = row[1:]
            previous_kept_source_index = source_row_index
            kept_count += 1
    return {"kept_rows": kept_count, "dropped_rows": dropped, "interval_anomalies": interval_anomalies}


def inspect_csv(path, time_mode, allow_equal_timestamps=False):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or header[0] != "date" or len(set(header)) != len(header):
            raise ValueError(f"Expected a unique header beginning with 'date': {path}")
        value_columns = header[1:]
        if not value_columns:
            raise ValueError(f"No numeric value columns in {path}")
        row_count = 0
        first_time = last_time = None
        previous_time = None
        deltas = []
        duplicate_pairs = 0
        invalid_cells = []
        sentinel_counts = {str(value): 0 for value in (-999.0, -9999.0, 99999.0)}
        for row in reader:
            row_count += 1
            if len(row) != len(header):
                raise ValueError(f"Malformed row {row_count + 1} in {path}: {len(row)} fields")
            try:
                current_time = parse_time(row[0], time_mode)
            except ValueError as exc:
                raise ValueError(f"Unparseable date at row {row_count + 1} in {path}: {row[0]!r}") from exc
            if previous_time is not None:
                delta = (current_time - previous_time).total_seconds()
                if delta < 0 or (delta == 0 and not allow_equal_timestamps):
                    raise ValueError(f"Time column is not strictly increasing at row {row_count + 1} in {path}")
                if delta == 0:
                    duplicate_pairs += 1
                deltas.append(delta)
            else:
                first_time = current_time
            previous_time = current_time
            last_time = current_time
            for column, raw in zip(value_columns, row[1:]):
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    invalid_cells.append({"row": row_count + 1, "column": column, "value": raw})
                    continue
                if not math.isfinite(value):
                    invalid_cells.append({"row": row_count + 1, "column": column, "value": raw})
                for sentinel in (-999.0, -9999.0, 99999.0):
                    if value == sentinel:
                        sentinel_counts[str(sentinel)] += 1
        if row_count == 0:
            raise ValueError(f"CSV has no records: {path}")
    if invalid_cells:
        raise ValueError(f"Found nonnumeric or nonfinite values in {path}: {invalid_cells[:5]}")
    bounds = np.floor(np.array([0.0, 0.6, 0.75, 0.85, 1.0]) * row_count).astype(int)
    train_end = int(bounds[1])
    train_matrix = np.empty((train_end, len(value_columns)), dtype=np.float64)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        next(reader)
        for i, row in enumerate(reader):
            if i >= train_end:
                break
            train_matrix[i] = [float(raw) for raw in row[1:]]
    train_mean = train_matrix.mean(axis=0)
    train_std = train_matrix.std(axis=0)
    if not np.isfinite(train_mean).all() or not np.isfinite(train_std).all():
        raise ValueError(f"Train-only normalization stats are not finite for {path}")
    delta_array = np.asarray(deltas, dtype=np.float64)
    return {
        "path": str(path.resolve()),
        "rows": int(row_count),
        "header": header,
        "value_columns": value_columns,
        "channels": len(value_columns),
        "time_format": time_mode,
        "strictly_increasing": duplicate_pairs == 0,
        "adjacent_duplicate_timestamp_pairs": duplicate_pairs,
        "first_time": first_time.isoformat(sep=" "),
        "last_time": last_time.isoformat(sep=" "),
        "delta_seconds_min": float(delta_array.min()),
        "delta_seconds_median": float(np.median(delta_array)),
        "delta_seconds_max": float(delta_array.max()),
        "invalid_value_cell_count": 0,
        "sentinel_like_counts": sentinel_counts,
        "split_bounds": bounds.tolist(),
        "train_rows": train_end,
        "train_stats_only": {
            column: {
                "mean": float(train_mean[i]),
                "std": float(train_std[i]),
                "min": float(train_matrix[:, i].min()),
                "max": float(train_matrix[:, i].max()),
                "zero_or_near_zero_variance": bool(train_std[i] < 1e-8),
            }
            for i, column in enumerate(value_columns)
        },
    }


weather_transform = derive_weather_deduplicated(weather_path, derived_weather_path)
weather_drop_report = {
    "source_csv": str(weather_path.resolve()),
    "prepared_csv": str(derived_weather_path.resolve()),
    "dropped_observations": weather_transform["dropped_rows"],
    "interval_anomalies_seconds_not_interpolated": weather_transform["interval_anomalies"],
}
(root / "prepared" / "weather_dropped_rows.json").write_text(
    json.dumps(weather_drop_report, ensure_ascii=True, indent=2) + "\n"
)
exchange_row_count = derive_exchange_dates(exchange_path, derived_exchange_path)
weather_original = inspect_csv(weather_path, "iso", allow_equal_timestamps=True)
weather_prepared = inspect_csv(derived_weather_path, "iso")
exchange_original = inspect_csv(exchange_path, "exchange_slash")
exchange_derived = inspect_csv(derived_exchange_path, "iso")
if weather_transform["kept_rows"] != weather_prepared["rows"]:
    raise ValueError("Weather prepared row count differs from the deduplication report")
if len(weather_transform["dropped_rows"]) != weather_original["adjacent_duplicate_timestamp_pairs"]:
    raise ValueError("Weather duplicate count differs from the preflight report")
if any(not item["numeric_strings_identical_to_previous_kept_row"] for item in weather_transform["dropped_rows"]):
    raise ValueError("Weather dropped row did not pass exact numeric string equality")
weather_dropped_indices = {
    item["source_data_row_index_zero_based"] for item in weather_transform["dropped_rows"]
}
with weather_path.open("r", newline="", encoding="utf-8-sig") as old_stream, \
     derived_weather_path.open("r", newline="", encoding="utf-8-sig") as new_stream:
    old_rows, new_rows = csv.reader(old_stream), csv.reader(new_stream)
    if next(old_rows) != next(new_rows):
        raise ValueError("Weather source/prepared headers differ")
    new_row = next(new_rows, None)
    for source_row_index, old_row in enumerate(old_rows):
        if source_row_index in weather_dropped_indices:
            continue
        if old_row != new_row:
            raise ValueError(f"Weather prepared row differs from source at data row {source_row_index}")
        new_row = next(new_rows, None)
    if new_row is not None or next(new_rows, None) is not None:
        raise ValueError("Weather prepared CSV has extra rows")
if exchange_row_count != exchange_original["rows"] or exchange_derived["rows"] != exchange_original["rows"]:
    raise ValueError("ExchangeRate source/derived row counts differ")
if exchange_original["header"] != exchange_derived["header"]:
    raise ValueError("ExchangeRate source/derived headers differ")
with exchange_path.open("r", newline="", encoding="utf-8-sig") as old_stream, \
     derived_exchange_path.open("r", newline="", encoding="utf-8-sig") as new_stream:
    old_rows, new_rows = csv.reader(old_stream), csv.reader(new_stream)
    next(old_rows); next(new_rows)
    for line_number, (old_row, new_row) in enumerate(zip(old_rows, new_rows), start=2):
        if old_row[1:] != new_row[1:]:
            raise ValueError(f"ExchangeRate numeric strings changed on row {line_number}")
        if parse_time(old_row[0], "exchange_slash") != parse_time(new_row[0], "iso"):
            raise ValueError(f"ExchangeRate timestamp changed on row {line_number}")

report = {
    "split_ratios": [0.60, 0.15, 0.10, 0.15],
    "weather_original": weather_original,
    "weather_prepared": weather_prepared,
    "weather_dropped_observations": weather_transform["dropped_rows"],
    "weather_non_nominal_intervals": weather_transform["interval_anomalies"],
    "weather_values_and_row_order_preserved_except_exact_duplicate_removal": True,
    "exchange_rate_original": exchange_original,
    "exchange_rate_prepared": exchange_derived,
    "exchange_rate_value_cells_preserved_exactly": True,
    "test_numeric_summaries_requested_or_computed": False,
}
(root / "data_preflight.json").write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
print(json.dumps({
    name: {k: v for k, v in report[name].items() if k not in ("train_stats_only", "value_columns")}
    for name in ("weather_original", "weather_prepared", "exchange_rate_original", "exchange_rate_prepared")
}, ensure_ascii=True, indent=2))
print("exchange_rate_value_cells_preserved_exactly=true")
sentinel_locations = {
    name: item["sentinel_like_counts"]
    for name, item in (("weather_original", report["weather_original"]),
                       ("weather_prepared", report["weather_prepared"]),
                       ("exchange_rate_original", report["exchange_rate_original"]),
                       ("exchange_rate_prepared", report["exchange_rate_prepared"]))
    if any(item["sentinel_like_counts"].values())
}
if sentinel_locations:
    raise ValueError(f"Sentinel-like finite values require review; no imputation was applied: {sentinel_locations}")
PY

if [[ "${TSLIB_TREND_FUSION_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight-only mode completed; CUDA smoke and training were not started"
  run_stage="preflight_complete"
  exit 0
fi

echo "Checking only the assigned physical GPUs 2 and 3 before CUDA smoke"
run_stage="gpu_preflight"
gpu_status="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu_weather,$gpu_exchange")"
printf '%s\n' "$gpu_status" | tee "$fusion_root/logs/gpu_preflight.txt"
while IFS=, read -r gpu_index memory_used utilization; do
  gpu_index="${gpu_index//[!0-9]/}"
  memory_used="${memory_used//[!0-9]/}"
  utilization="${utilization//[!0-9]/}"
  if [[ "$gpu_index" == "$gpu_weather" || "$gpu_index" == "$gpu_exchange" ]]; then
    if (( memory_used > 1024 || utilization > 20 )); then
      echo "Assigned GPU $gpu_index is not idle (memory=${memory_used}MiB utilization=${utilization}%)" >&2
      exit 3
    fi
  fi
done <<< "$gpu_status"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
{
  echo "fusion_root=$fusion_root"
  echo "python=$python_bin"
  echo "Weather_GPU=$gpu_weather"
  echo "ExchangeRate_GPU=$gpu_exchange"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for physical_gpu in "$gpu_weather" "$gpu_exchange"; do
    echo "--- physical_gpu=$physical_gpu ---"
    CUDA_VISIBLE_DEVICES="$physical_gpu" "$python_bin" - <<'PY'
import sys
import numpy
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Fusion validation requires CUDA; refusing to run on CPU")
print(f"python_version={sys.version.replace(chr(10), ' ')}")
print(f"numpy_version={numpy.__version__}")
print(f"torch_version={torch.__version__}")
print(f"cuda_version={torch.version.cuda}")
print(f"device_0={torch.cuda.get_device_name(0)}")
PY
  done
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv -i "$gpu_weather,$gpu_exchange"
} > "$fusion_root/environment.txt" 2>&1

run_stage="config_generation"
"$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["FUSION_ROOT"])
preflight = json.loads((root / "data_preflight.json").read_text())
template = json.loads(Path("trend_lab/configs/etth1_pilot.json").read_text())
methods = template["methods"]
if len(methods) != 11 or "event" in methods or "event_nohazard" in methods:
    raise RuntimeError("Expected the frozen original 11-method matrix without event methods")

datasets = {
    "weather": (preflight["weather_prepared"], Path(os.environ["WEATHER_PREPARED_CSV"]).resolve()),
    "exchange_rate": (preflight["exchange_rate_prepared"], root / "prepared" / "exchange_rate_iso.csv"),
}
for name, (metadata, path) in datasets.items():
    cfg = json.loads(json.dumps(template))
    cfg["backbone"] = "linear"
    cfg["backbones"] = ["linear", "dlinear"]
    cfg["methods"] = methods
    cfg["data"] = {
        "kind": "csv",
        "csv_path": str(path.resolve()),
        "value_columns": metadata["value_columns"],
        "time_column": "date",
        "channels": metadata["channels"],
        "lookback": 96,
        "horizon": 24,
        "split_ratios": [0.6, 0.15, 0.1, 0.15],
        "train_stride": 4,
        "eval_stride": 24,
    }
    (root / "configs" / f"{name}_matrix.json").write_text(json.dumps(cfg, ensure_ascii=True, indent=2) + "\n")
    smoke = json.loads(json.dumps(cfg))
    smoke.pop("backbones", None)
    smoke["backbone"] = "dlinear"
    (root / "configs" / f"{name}_smoke.json").write_text(json.dumps(smoke, ensure_ascii=True, indent=2) + "\n")
PY

cp "$dtl_root"/trend_lab/*.py "$fusion_root/source/trend_lab/"
cp "$dtl_root"/trend_lab/README.md "$dtl_root"/trend_lab/requirements.txt "$fusion_root/source/trend_lab/"
cp "$dtl_root"/trend_lab/configs/*.json "$fusion_root/source/trend_lab/configs/"
cp "$dtl_root"/trend_lab/tests/*.py "$fusion_root/source/trend_lab/tests/"
cp "$script_dir"/*.sh "$fusion_root/source/trend_lab/scripts/"
(cd "$fusion_root/source" && find trend_lab -type f -print0 | sort -z | xargs -0 sha256sum) \
  > "$fusion_root/source_sha256.txt"
cp "$reference_dlinear" "$fusion_root/reference/Time-Series-Library/models/DLinear.py"
cp "$reference_decomp" "$fusion_root/reference/Time-Series-Library/layers/Autoformer_EncDec.py"
cp "$protocol_path" "$fusion_root/experiment_protocol.md"
sha256sum "$weather_csv" "$fusion_root/prepared/weather_deduplicated.csv" \
  "$exchange_csv" "$fusion_root/prepared/exchange_rate_iso.csv" \
  "$fusion_root/prepared/weather_dropped_rows.json" \
  > "$fusion_root/data_sha256.txt"
sha256sum "$fusion_root/reference/Time-Series-Library/models/DLinear.py" \
  "$fusion_root/reference/Time-Series-Library/layers/Autoformer_EncDec.py" \
  "$fusion_root/experiment_protocol.md" > "$fusion_root/reference_sha256.txt"
sha256sum "$fusion_root/configs"/*.json > "$fusion_root/config_sha256.txt"

cd "$dtl_root"
echo "Running one-seed DLinear CUDA smoke for both datasets"
run_stage="cuda_smoke"
CUDA_VISIBLE_DEVICES="$gpu_weather" "$python_bin" -u -m trend_lab.run \
  --config "$fusion_root/configs/weather_smoke.json" \
  --output "$fusion_root/cuda_smoke/Weather" --smoke \
  > "$fusion_root/logs/weather_smoke.log" 2>&1 &
pid_smoke_weather=$!
CUDA_VISIBLE_DEVICES="$gpu_exchange" "$python_bin" -u -m trend_lab.run \
  --config "$fusion_root/configs/exchange_rate_smoke.json" \
  --output "$fusion_root/cuda_smoke/ExchangeRate" --smoke \
  > "$fusion_root/logs/exchange_rate_smoke.log" 2>&1 &
pid_smoke_exchange=$!
status_smoke_weather=0
status_smoke_exchange=0
wait "$pid_smoke_weather" || status_smoke_weather=$?
wait "$pid_smoke_exchange" || status_smoke_exchange=$?
echo "Weather smoke exit=$status_smoke_weather; ExchangeRate smoke exit=$status_smoke_exchange"
if [[ "$status_smoke_weather" -ne 0 || "$status_smoke_exchange" -ne 0 ]]; then
  exit 1
fi

echo "Starting Weather/GPU$gpu_weather and ExchangeRate/GPU$gpu_exchange training"
run_stage="training"
CUDA_VISIBLE_DEVICES="$gpu_weather" "$python_bin" -u -m trend_lab.run \
  --config "$fusion_root/configs/weather_matrix.json" \
  --output "$fusion_root/Weather" > "$fusion_root/logs/weather_matrix.log" 2>&1 &
pid_weather=$!
CUDA_VISIBLE_DEVICES="$gpu_exchange" "$python_bin" -u -m trend_lab.run \
  --config "$fusion_root/configs/exchange_rate_matrix.json" \
  --output "$fusion_root/ExchangeRate" > "$fusion_root/logs/exchange_rate_matrix.log" 2>&1 &
pid_exchange=$!
echo "Weather PID=$pid_weather GPU=$gpu_weather log=$fusion_root/logs/weather_matrix.log"
echo "ExchangeRate PID=$pid_exchange GPU=$gpu_exchange log=$fusion_root/logs/exchange_rate_matrix.log"

status_weather=0
status_exchange=0
wait "$pid_weather" || status_weather=$?
wait "$pid_exchange" || status_exchange=$?
echo "Weather exit=$status_weather"
echo "ExchangeRate exit=$status_exchange"
if [[ "$status_weather" -ne 0 || "$status_exchange" -ne 0 ]]; then
  exit 1
fi
echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$fusion_root/environment.txt"
run_stage="completed"
echo "Completed fusion training artifacts: $fusion_root"
