#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_root="$script_dir"
package_root="/home/lgc/TSLib/DTL"
python_bin="/home/lgc/.conda/envs/TFB/bin/python"
phase="${1:-}"
if [[ "$phase" != "smoke" && "$phase" != "full" ]]; then
  echo "Usage: $0 smoke|full" >&2
  exit 2
fi

run_stage="starting_$phase"
record_status() {
  local code=$?
  trap - EXIT
  {
    printf 'phase=%s\n' "$phase"
    printf 'exit_code=%s\n' "$code"
    printf 'last_stage=%s\n' "$run_stage"
    printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$run_root/phase_${phase}_status.txt"
  exit "$code"
}
trap record_status EXIT

cd "$package_root"
if [[ "$phase" == "full" ]]; then
  "$python_bin" - "$run_root/cuda_smoke" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for dataset in ("Electricity", "Traffic"):
    for backbone in ("linear", "dlinear"):
        path = root / dataset / f"data__{backbone}" / "results.json"
        if not path.is_file():
            raise FileNotFoundError(f"CUDA smoke output missing: {path}")
        result = json.loads(path.read_text())
        if result.get("smoke") is not True or result.get("checkpoint_reload_verified") is not True:
            raise RuntimeError(f"CUDA smoke did not pass reload checks: {path}")
print("Both datasets and backbones passed one-seed CUDA smoke/reload checks")
PY
fi

run_stage="gpu_preflight_$phase"
gpu_status="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i 2,3)"
printf '%s\n' "$gpu_status" > "$run_root/logs/gpu_preflight_${phase}.txt"
while IFS=, read -r gpu_index memory_used utilization; do
  gpu_index="${gpu_index//[!0-9]/}"
  memory_used="${memory_used//[!0-9]/}"
  utilization="${utilization//[!0-9]/}"
  if [[ "$gpu_index" == "2" || "$gpu_index" == "3" ]]; then
    if (( memory_used > 1024 || utilization > 20 )); then
      echo "Assigned GPU $gpu_index is not idle (memory=${memory_used}MiB utilization=${utilization}%)" >&2
      exit 3
    fi
  fi
done <<< "$gpu_status"

run_stage="environment_$phase"
{
  echo "run_root=$run_root"
  echo "phase=$phase"
  echo "python=$python_bin"
  echo "OMP_NUM_THREADS=2"
  echo "MKL_NUM_THREADS=2"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for physical_gpu in 2 3; do
    echo "--- physical_gpu=$physical_gpu ---"
    CUDA_VISIBLE_DEVICES="$physical_gpu" "$python_bin" - <<'PY'
import sys, numpy, torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required; refusing CPU fallback")
print(f"python_version={sys.version.replace(chr(10), ' ')}")
print(f"numpy_version={numpy.__version__}")
print(f"torch_version={torch.__version__}")
print(f"cuda_version={torch.version.cuda}")
print(f"device_0={torch.cuda.get_device_name(0)}")
PY
  done
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv -i 2,3
} > "$run_root/environment_${phase}.txt" 2>&1

if [[ "$phase" == "smoke" ]]; then
  output_root="$run_root/cuda_smoke"
  log_suffix="smoke"
  smoke_flag="--smoke"
else
  output_root="$run_root/results"
  log_suffix="matrix"
  smoke_flag=""
fi
mkdir -p "$output_root" "$run_root/logs"

run_stage="launch_${phase}"
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$python_bin" -u -m trend_lab.run \
  --config "$run_root/configs/electricity_matrix.json" \
  --output "$output_root/Electricity" $smoke_flag \
  > "$run_root/logs/electricity_${log_suffix}.log" 2>&1 &
pid_electricity=$!
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$python_bin" -u -m trend_lab.run \
  --config "$run_root/configs/traffic_matrix.json" \
  --output "$output_root/Traffic" $smoke_flag \
  > "$run_root/logs/traffic_${log_suffix}.log" 2>&1 &
pid_traffic=$!
echo "Electricity PID=$pid_electricity GPU=2 log=$run_root/logs/electricity_${log_suffix}.log"
echo "Traffic PID=$pid_traffic GPU=3 log=$run_root/logs/traffic_${log_suffix}.log"

status_electricity=0
status_traffic=0
wait "$pid_electricity" || status_electricity=$?
wait "$pid_traffic" || status_traffic=$?
echo "Electricity exit=$status_electricity"
echo "Traffic exit=$status_traffic"
if [[ "$status_electricity" -ne 0 || "$status_traffic" -ne 0 ]]; then
  exit 1
fi
run_stage="completed_$phase"
echo "Completed $phase run under $output_root"
