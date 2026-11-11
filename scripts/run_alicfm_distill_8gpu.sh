#!/usr/bin/env bash
# Seven selected ALI students; uses existing training/evaluation and GPU queue.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}" \
  "$repo_root/scripts/run_distill_parallel.py" --method alicfm \
  --datasets cite5 cite50 cite100 multi5 multi50 multi100 cell_tracking \
  --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
  --workers-per-gpu 1 --max-parallel 8 \
  --threads "${CPU_THREADS_PER_RUN:-4}" \
  --min-free-gb "${MIN_FREE_GB:-12}" \
  --min-disk-free-gb "${MIN_DISK_FREE_GB:-20}" \
  --mode "${WANDB_MODE:-online}" \
  --output-root "${ALI_DISTILL_ROOT:-/data/lthung/flow2pc/distill/alicfm/selected-students}" \
  "$@"
