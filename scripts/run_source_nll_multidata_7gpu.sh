#!/usr/bin/env bash
# First distill full-dimensional students; run explicit diagnostic phases later.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}"
student_root="${SOURCE_NLL_OUTPUT_ROOT:-/data/lthung/flow2pc/distill/source-nll-multidata-v1}"
diagnostic_root="${SOURCE_NLL_DIAGNOSTIC_ROOT:-/data/lthung/flow2pc/diagnostics/source-nll-multidata-v1}"
gpu_list="${GPUS:-0,1,2,4,5,6,7}"
phase=train
forward=()
while (($#)); do
  case "$1" in
    --phase)
      if (($# < 2)); then echo '--phase needs train, diagnose or consistency' >&2; exit 2; fi
      phase="$2"; shift 2 ;;
    --gpus)
      if (($# < 2)); then echo '--gpus needs a comma-separated GPU list' >&2; exit 2; fi
      gpu_list="$2"; shift 2 ;;
    --gpus=*) gpu_list="${1#*=}"; shift ;;
    *) forward+=("$1"); shift ;;
  esac
done
IFS=',' read -r -a devices <<< "$gpu_list"
for device in "${devices[@]}"; do
  if [[ ! "$device" =~ ^[0-9]+$ ]]; then echo 'GPU IDs must be nonnegative integers.' >&2; exit 2; fi
  if ((10#$device == 3)); then echo 'GPU 3 is excluded from this batch.' >&2; exit 2; fi
done
resources=(--gpus "$gpu_list" --max-parallel "${MAX_PARALLEL:-7}"
           --threads "${CPU_THREADS_PER_RUN:-4}" --mode "${WANDB_MODE:-online}")
datasets=(--datasets cite5 cite50 cite100 knot)
weights=(--weights 0 0.003 0.01 0.03 0.1)
export PYTHONDONTWRITEBYTECODE=1
case "$phase" in
  train)
    exec "$python_bin" "$repo_root/scripts/run_source_nll_pilot.py" \
      --method both "${datasets[@]}" "${weights[@]}" --lr-factors 1 \
      --output-root "$student_root" "${resources[@]}" "${forward[@]}"
    ;;
  diagnose)
    exec "$python_bin" "$repo_root/scripts/run_pc_consistency_pilot.py" \
      --methods alicfm otpfm "${datasets[@]}" "${weights[@]}" \
      --pilot-root "$student_root" --output-root "$diagnostic_root/direct-query-v1" \
      --direct-only --query-checks --samples 2048 --metric-samples 512 \
      --batch-size 64 --tail-count 16 "${resources[@]}" "${forward[@]}"
    ;;
  consistency)
    exec "$python_bin" "$repo_root/scripts/run_pc_consistency_pilot.py" \
      --methods alicfm otpfm "${datasets[@]}" "${weights[@]}" \
      --pilot-root "$student_root" --output-root "$diagnostic_root/density-velocity-v1" \
      --query-checks --samples 128 --metric-samples 128 --batch-size 32 --tail-count 16 \
      --max-nfe 10000 "${resources[@]}" "${forward[@]}"
    ;;
  *) echo '--phase needs train, diagnose or consistency' >&2; exit 2 ;;
esac
