#!/usr/bin/env bash
# Only missing student sweeps: 8 ALI-CFM + 11 OTP-FM; never train teachers.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}" \
  "$repo_root/scripts/run_student_sweep.py" \
  --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
  --workers-per-gpu "${WORKERS_PER_GPU:-1}" \
  --max-parallel "${MAX_PARALLEL:-8}" \
  --threads "${CPU_THREADS_PER_RUN:-4}" \
  --min-free-gb "${MIN_FREE_GB:-12}" \
  --min-disk-free-gb "${MIN_DISK_FREE_GB:-20}" \
  --mode "${WANDB_MODE:-online}" \
  --teacher-root "${TEACHER_ROOT:-/data/lthung/flow2pc/pretrained-teachers}" \
  --output-root "${STUDENT_SWEEP_ROOT:-/data/lthung/flow2pc/distill/student-sweep-100}" \
  "$@"
