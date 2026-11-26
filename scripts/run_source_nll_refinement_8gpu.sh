#!/usr/bin/env bash
# ST: 7 initial-NLL weights x 3 learning rates x 2 frozen teachers = 42 runs.
# GPU 3 is reserved; this refinement uses the other seven devices.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo_root/scripts/run_source_nll_pilot_8gpu.sh" \
  --gpus 0,1,2,4,5,6,7 --max-parallel 7 \
  --weights 0 0.003 0.01 0.03 0.1 0.3 1 \
  --lr-factors 1 0.3333333333333333 0.1 \
  --output-root /data/lthung/flow2pc/distill/source-nll-refinement-v1 "$@"
