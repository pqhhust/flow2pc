#!/usr/bin/env bash
# Two frozen ST pairs; existing main.py E3 runners, no new experiment engine.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}"
output_root="${E3_PILOT_OUTPUT_ROOT:-/data/lthung/flow2pc/experiments/source-nll-pilot-v1}"
ali_gpu="${ALI_GPU:-0}"
otp_gpu="${OTP_GPU:-1}"
phase=all
preview=false
while (($#)); do
  case "$1" in
    --dry-run) preview=true; shift ;;
    --phase)
      if (($# < 2)); then echo '--phase needs e3_1, e3_2 or all' >&2; exit 2; fi
      phase="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$phase" in
  all) experiments=(e3_1 e3_2) ;;
  e3_1|e3_2) experiments=("$phase") ;;
  *) echo '--phase needs e3_1, e3_2 or all' >&2; exit 2 ;;
esac
if [[ ! "$ali_gpu" =~ ^[0-9]+$ || ! "$otp_gpu" =~ ^[0-9]+$ ||
      "$ali_gpu" == "$otp_gpu" || "$ali_gpu" == 3 || "$otp_gpu" == 3 ]]; then
  echo 'Select two distinct physical GPU indices with ALI_GPU/OTP_GPU; GPU 3 is excluded.' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${CPU_THREADS_PER_RUN:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export WANDB_MODE="${WANDB_MODE:-online}"
cd "$repo_root"

# Resolve every requested config and validate checkpoint metadata before any
# worker starts. --validate-only for experiment configs loads no models/data.
for experiment in "${experiments[@]}"; do
  for method in alicfm otpfm; do
    config="configs/experiments/$experiment/$method/source_nll_pilot/${method}_st_hclt.yaml"
    if "$preview"; then
      "$python_bin" main.py --config "$config" --validate-only \
        --set "output.dir=$output_root/$experiment/$method/st" \
        --set "logging.mode=$WANDB_MODE"
    else
      "$python_bin" main.py --config "$config" --validate-only \
        --set "output.dir=$output_root/$experiment/$method/st" \
        --set "logging.mode=$WANDB_MODE" >/dev/null
    fi
  done
done
if "$preview"; then exit 0; fi

mkdir -p "$output_root/_launcher"
failed=0
for experiment in "${experiments[@]}"; do
  pids=()
  for method in alicfm otpfm; do
    gpu="$ali_gpu"
    if [[ "$method" == otpfm ]]; then gpu="$otp_gpu"; fi
    config="configs/experiments/$experiment/$method/source_nll_pilot/${method}_st_hclt.yaml"
    log="$output_root/_launcher/${experiment}_${method}.log"
    echo "Launching $experiment/$method on physical GPU $gpu; log: $log"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u main.py --config "$config" \
      --set "output.dir=$output_root/$experiment/$method/st" \
      --set "logging.mode=$WANDB_MODE" >"$log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
done
echo "Pilot outputs: $output_root/<e3_1|e3_2>/<alicfm|otpfm>/st/"
exit "$failed"
