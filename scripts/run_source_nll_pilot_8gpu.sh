#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}"
exec "$python_bin" "$repo_root/scripts/run_source_nll_pilot.py" \
  --gpus "${GPUS:-0,1,2,4,5,6,7}" --max-parallel "${MAX_PARALLEL:-7}" \
  --workers-per-gpu 1 --threads "${CPU_THREADS_PER_RUN:-4}" \
  --mode "${WANDB_MODE:-online}" "$@"
