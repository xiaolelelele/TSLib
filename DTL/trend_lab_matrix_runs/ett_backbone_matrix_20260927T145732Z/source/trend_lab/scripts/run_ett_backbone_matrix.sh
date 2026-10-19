#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dtl_root="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${TSLIB_TREND_PYTHON:-/home/lgc/.conda/envs/TFB/bin/python}"
gpu_etth1="${TSLIB_TREND_GPU_ETTH1:-1}"
gpu_etth2="${TSLIB_TREND_GPU_ETTH2:-2}"
run_parent="${TSLIB_TREND_MATRIX_RUNS_DIR:-$dtl_root/trend_lab_matrix_runs}"
ett_source_root="$dtl_root/../TSLib/Time-Series-Library"
data_root="$ett_source_root/dataset/ETT-small"
protocol_path="$dtl_root/research/etth_idea_screening_protocol_2026-09-27.md"
utc_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
matrix_root="$run_parent/ett_backbone_matrix_$utc_stamp"

if [[ ! -x "$python_bin" ]]; then
  echo "Python executable is unavailable: $python_bin" >&2
  exit 2
fi
for required_path in \
  "$dtl_root/trend_lab/configs/dlinear_cuda_smoke.json" \
  "$dtl_root/trend_lab/configs/etth1_architecture_matrix.json" \
  "$dtl_root/trend_lab/configs/etth2_architecture_matrix.json" \
  "$data_root/ETTh1.csv" "$data_root/ETTh2.csv" \
  "$ett_source_root/models/DLinear.py" "$ett_source_root/layers/Autoformer_EncDec.py" \
  "$protocol_path"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Required matrix input is missing: $required_path" >&2
    exit 2
  fi
done
if [[ -e "$matrix_root" ]]; then
  echo "Run directory already exists: $matrix_root" >&2
  exit 2
fi

mkdir -p "$matrix_root/logs" "$matrix_root/configs" \
  "$matrix_root/source/trend_lab/configs" "$matrix_root/source/trend_lab/tests" \
  "$matrix_root/source/trend_lab/scripts" \
  "$matrix_root/reference/Time-Series-Library/models" \
  "$matrix_root/reference/Time-Series-Library/layers"
cp "$dtl_root"/trend_lab/*.py "$matrix_root/source/trend_lab/"
cp "$dtl_root"/trend_lab/README.md "$dtl_root"/trend_lab/requirements.txt "$matrix_root/source/trend_lab/"
cp "$dtl_root"/trend_lab/configs/*.json "$matrix_root/source/trend_lab/configs/"
cp "$dtl_root"/trend_lab/tests/*.py "$matrix_root/source/trend_lab/tests/"
cp "$script_dir"/*.sh "$matrix_root/source/trend_lab/scripts/"
cp "$dtl_root/trend_lab/configs/dlinear_cuda_smoke.json" \
  "$dtl_root/trend_lab/configs/etth1_architecture_matrix.json" \
  "$dtl_root/trend_lab/configs/etth2_architecture_matrix.json" "$matrix_root/configs/"
(cd "$matrix_root/source" && find trend_lab -type f -print0 | sort -z | xargs -0 sha256sum) \
  > "$matrix_root/source_sha256.txt"
cp "$ett_source_root/models/DLinear.py" \
  "$matrix_root/reference/Time-Series-Library/models/DLinear.py"
cp "$ett_source_root/layers/Autoformer_EncDec.py" \
  "$matrix_root/reference/Time-Series-Library/layers/Autoformer_EncDec.py"
sha256sum "$matrix_root/reference/Time-Series-Library/models/DLinear.py" \
  "$matrix_root/reference/Time-Series-Library/layers/Autoformer_EncDec.py" \
  > "$matrix_root/reference_sha256.txt"
cp "$protocol_path" "$matrix_root/experiment_protocol.md"
sha256sum "$matrix_root/experiment_protocol.md" >> "$matrix_root/reference_sha256.txt"
sha256sum "$data_root/ETTh1.csv" "$data_root/ETTh2.csv" > "$matrix_root/data_sha256.txt"
cp "$dtl_root/trend_lab/configs/etth1_architecture_matrix.json" "$matrix_root/configs/"
cp "$dtl_root/trend_lab/configs/etth2_architecture_matrix.json" "$matrix_root/configs/"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
{
  echo "matrix_root=$matrix_root"
  echo "python=$python_bin"
  echo "ETTh1_GPU=$gpu_etth1"
  echo "ETTh2_GPU=$gpu_etth2"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for physical_gpu in "$gpu_etth1" "$gpu_etth2"; do
    echo "--- physical_gpu=$physical_gpu ---"
    CUDA_VISIBLE_DEVICES="$physical_gpu" "$python_bin" - <<'PY'
import sys
import numpy
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Architecture matrix requires CUDA; refusing to run on CPU")
print(f"python_version={sys.version.replace(chr(10), ' ')}")
print(f"numpy_version={numpy.__version__}")
print(f"torch_version={torch.__version__}")
print(f"cuda_version={torch.version.cuda}")
print(f"device_0={torch.cuda.get_device_name(0)}")
PY
  done
  nvidia-smi
} > "$matrix_root/environment.txt" 2>&1

cd "$dtl_root"
echo "Matrix artifacts: $matrix_root"
echo "[1/3] Running the unit suite on CPU"
CUDA_VISIBLE_DEVICES='' "$python_bin" -m unittest discover -s trend_lab/tests -v \
  2>&1 | tee "$matrix_root/logs/unit_tests.log"

echo "[2/3] Running DLinear CUDA smoke on ETTh1"
CUDA_VISIBLE_DEVICES="$gpu_etth1" "$python_bin" -m trend_lab.run \
  --config "$dtl_root/trend_lab/configs/dlinear_cuda_smoke.json" \
  --output "$matrix_root/dlinear_cuda_smoke" --smoke \
  2>&1 | tee "$matrix_root/logs/dlinear_cuda_smoke.log"

echo "[3/3] Starting the fixed ETTh1 x ETTh2 architecture matrix on GPUs $gpu_etth1 and $gpu_etth2"
CUDA_VISIBLE_DEVICES="$gpu_etth1" "$python_bin" -u -m trend_lab.run \
  --config "$dtl_root/trend_lab/configs/etth1_architecture_matrix.json" \
  --output "$matrix_root/ETTh1" > "$matrix_root/logs/ETTh1_matrix.log" 2>&1 &
pid_etth1=$!
CUDA_VISIBLE_DEVICES="$gpu_etth2" "$python_bin" -u -m trend_lab.run \
  --config "$dtl_root/trend_lab/configs/etth2_architecture_matrix.json" \
  --output "$matrix_root/ETTh2" > "$matrix_root/logs/ETTh2_matrix.log" 2>&1 &
pid_etth2=$!
echo "ETTh1 PID=$pid_etth1 GPU=$gpu_etth1 log=$matrix_root/logs/ETTh1_matrix.log"
echo "ETTh2 PID=$pid_etth2 GPU=$gpu_etth2 log=$matrix_root/logs/ETTh2_matrix.log"

status_etth1=0
status_etth2=0
wait "$pid_etth1" || status_etth1=$?
wait "$pid_etth2" || status_etth2=$?
echo "ETTh1 exit=$status_etth1"
echo "ETTh2 exit=$status_etth2"
if [[ "$status_etth1" -ne 0 || "$status_etth2" -ne 0 ]]; then
  exit 1
fi
echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$matrix_root/environment.txt"
echo "Completed architecture matrix: $matrix_root"
