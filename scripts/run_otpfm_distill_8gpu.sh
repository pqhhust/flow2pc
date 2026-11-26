#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}" \
  "$repo_root/scripts/run_otpfm_distill_parallel.py" \
  --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
  --workers-per-gpu "${WORKERS_PER_GPU:-2}" \
  --max-parallel "${MAX_PARALLEL:-16}" \
  --threads "${CPU_THREADS_PER_RUN:-4}" \
  --min-free-gb "${MIN_FREE_GB:-12}" \
  --min-disk-free-gb "${MIN_DISK_FREE_GB:-20}" \
  --mode "${WANDB_MODE:-online}" \
  --output-root "${OTP_DISTILL_ROOT:-/data/lthung/flow2pc/distill/otpfm}" \
  "$@"
