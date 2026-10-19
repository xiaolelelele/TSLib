#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dtl_root="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${TSLIB_TREND_PYTHON:-/home/lgc/.conda/envs/TFB/bin/python}"
cuda_device="${TSLIB_TREND_CUDA_DEVICE:-1}"
run_parent="${TSLIB_TREND_RUNS_DIR:-$dtl_root/trend_lab_runs}"
config_path="$dtl_root/trend_lab/configs/etth1_pilot.json"
data_path="$dtl_root/../TSLib/Time-Series-Library/dataset/ETT-small/ETTh1.csv"
utc_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="$run_parent/etth1_pilot_$utc_stamp"

if [[ ! -x "$python_bin" ]]; then
  echo "Python executable is unavailable: $python_bin" >&2
  exit 2
fi
if [[ ! -f "$config_path" || ! -f "$data_path" ]]; then
  echo "Pilot config or ETTh1 CSV is missing: $config_path ; $data_path" >&2
  exit 2
fi
if [[ -e "$run_root" ]]; then
  echo "Run directory already exists: $run_root" >&2
  exit 2
fi

mkdir -p "$run_root/logs" "$run_root/source/trend_lab/configs" \
  "$run_root/source/trend_lab/tests" "$run_root/source/trend_lab/scripts"
cp "$dtl_root"/trend_lab/*.py "$run_root/source/trend_lab/"
cp "$dtl_root"/trend_lab/README.md "$dtl_root"/trend_lab/requirements.txt "$run_root/source/trend_lab/"
cp "$dtl_root"/trend_lab/configs/*.json "$run_root/source/trend_lab/configs/"
cp "$dtl_root"/trend_lab/tests/*.py "$run_root/source/trend_lab/tests/"
cp "$script_dir/run_etth1_pilot.sh" "$run_root/source/trend_lab/scripts/"
(cd "$run_root/source" && find trend_lab -type f -print0 | sort -z | xargs -0 sha256sum) > "$run_root/source_sha256.txt"
cp "$config_path" "$run_root/config.json"
sha256sum "$data_path" > "$run_root/data_sha256.txt"

export CUDA_VISIBLE_DEVICES="$cuda_device"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1

{
  echo "run_root=$run_root"
  echo "config=$config_path"
  echo "data=$data_path"
  echo "python=$python_bin"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$python_bin" - <<'PY'
import sys
import numpy
import torch
if not torch.cuda.is_available():
    raise RuntimeError("ETTh1 pilot requires CUDA; refusing to continue on CPU")
print(f"python_version={sys.version.replace(chr(10), ' ')}")
print(f"numpy_version={numpy.__version__}")
print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_version={torch.version.cuda}")
    print(f"visible_device_count={torch.cuda.device_count()}")
    print(f"device_0={torch.cuda.get_device_name(0)}")
PY
  nvidia-smi
} > "$run_root/environment.txt" 2>&1

cd "$dtl_root"
echo "Run artifacts: $run_root"
echo "[1/3] Running existing trend_lab unit tests"
"$python_bin" -m unittest discover -s trend_lab/tests -v 2>&1 | tee "$run_root/logs/unit_tests.log"

echo "[2/3] Running ETTh1 CSV CUDA smoke"
"$python_bin" -m trend_lab.run --config "$config_path" \
  --output "$run_root/cuda_smoke" --smoke 2>&1 | tee "$run_root/logs/cuda_smoke.log"

echo "[3/3] Running fixed three-seed ETTh1 pilot"
"$python_bin" -m trend_lab.run --config "$config_path" \
  --output "$run_root/pilot" 2>&1 | tee "$run_root/logs/pilot.log"

echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$run_root/environment.txt"
echo "Completed ETTh1 pilot: $run_root"
