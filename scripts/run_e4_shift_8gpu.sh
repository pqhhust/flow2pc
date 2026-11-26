#!/usr/bin/env bash
# Four evaluation-only pilot jobs: ALI-CFM/OTP-FM x Beijing/ST.
# Use --datasets beijing for just the 1D pilot; --dry-run reads metadata only.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}" \
  "$repo_root/scripts/run_experiments.py" --experiment e4 \
  --methods alicfm otpfm --datasets beijing st \
  --gpus "${GPUS:-0,1,2,3,4,5,6,7}" --workers-per-gpu 1 \
  --max-parallel "${MAX_PARALLEL:-4}" --threads "${CPU_THREADS_PER_RUN:-4}" \
  --min-free-gb "${MIN_FREE_GB:-4}" --min-disk-free-gb "${MIN_DISK_FREE_GB:-20}" \
  --mode "${WANDB_MODE:-online}" \
  --output-root "${E4_ROOT:-/data/lthung/flow2pc/experiments/e4/teacher-relative-v1}" "$@"
