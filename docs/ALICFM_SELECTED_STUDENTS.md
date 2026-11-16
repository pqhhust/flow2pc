# Selected ALI-CFM student recipes

The seven configs in `configs/distill/alicfm/` now record the best completed
historical single-cell and tracking sweep hyperparameters. All use an HCLT with
256 latent states, time-feature seed 0, student seed 0, batch size 64, LR 0.001,
and 50,000 updates. Fourier feature counts are:

| Config suffix | Time features | Historical selection score |
| --- | ---: | ---: |
| cite5_hclt | 64 | 0.000134803554 |
| cite50_hclt | 64 | 0.000998102274 |
| cite100_hclt | 32 | 0.001225615139 |
| multi5_hclt | 64 | 0.000081286220 |
| multi50_hclt | 32 | 0.000659957026 |
| multi100_hclt | 32 | 0.000994463051 |
| cell_tracking_hclt | 32 | 0.000160372322 |

Single-cell scores select on reserved observed-time velocity MSE. Tracking uses
the existing worst global/late/final velocity MSE normalized by teacher energy;
its score is not a raw MSE. Population EMD does not select students.

Historical selection records are stored at:

```text
/data/lthung/flow2pc/outputs/single_cell/best_ali_cfm_sweeps_t1/sweeps/
  {cite5d-t1-teacher43,cite50d-t1-teacher42,cite100d-t1-teacher44,
   multi5d-t1-teacher42,multi50d-t1-teacher43,multi100d-t1-teacher45}/selected_student.json
/data/lthung/flow2pc/cell_tracking/distillation_sweep/selected_student.json
```

## Teacher choice for the new experiments

At the user's request, the six single-cell configs use newly pretrained teachers:
`/data/lthung/flow2pc/pretrained-teachers/alicfm/DATASET/checkpoints/last.pt`.
The student hyperparameters transfer from the historical sweeps; these runs do
not claim an exact replay of their scores, teacher seeds, or historical pipeline.
The new runs follow the current shared sampler, optimizer, source initialization,
and dataset-specific validation code without modifications.

Tracking uses the successful normalized-coordinate teacher:
`/data/pqhung_group/lthung/ali-cfm/changed_files/ali_cfm_model.pth`.
It does not use the recent raw-pixel teacher whose ODE evaluation failed.
`configs/teacher_metadata/alicfm_cell_tracking.json` declares its existing
legacy interface: global model time, SELU CFM, ELU interpolant, border OT, and
normalized velocities. The existing legacy loader recovers the saved min/max
transform directly from the checkpoint; no scaler is fitted. The launcher
checks the pinned teacher SHA-256 before admitting this job.

Tracking metrics and overlays explicitly use the six dataset README frames
1, 25, 50, 75, 100, 115. Its historical sweep report used eight frames, so those
mean EMD values are not directly comparable to the new six-frame mean.

## Run in parallel

```bash
bash scripts/run_alicfm_distill_8gpu.sh --dry-run
bash scripts/run_alicfm_distill_8gpu.sh
```

The gitignored local convenience entry point is:

```bash
bash local_runs/distill_selected_alicfm.sh
```

This reuses `DistillationQueue` and the existing `main.py --config` training
path. The common Python entry point is `scripts/run_distill_parallel.py`;
the original OTP entry point retains its OTP defaults. No student training loop,
sampling mathematics, teacher loss, or PC implementation was added or changed.

The pool contains GPUs 0–7, with at most one job per GPU and eight workers total.
There are seven selected students: all seven can run concurrently when memory
is available; the eighth GPU has no additional experiment. GPU selection uses
`nvidia-smi`, with a default admission threshold of 12 GiB free. New launches
wait if the output filesystem has fewer than 20 GiB free. The queue retains
its existing completed-run skipping and checkpoint resume behavior. Each
worker keeps its own logs and output; teacher pretraining is never queued.

W&B project is `flow2pc`; names include dataset, teacher/student type, widths,
LR, batch size, budget, and seed. Existing W&B fallback behavior is retained.
Final evaluation saves both teacher and student ODE figures, plus the existing
direct-PC diagnostics; all training supervision remains teacher velocity MSE.

Outputs default to:

```text
/data/lthung/flow2pc/distill/alicfm/selected-students/
  DATASET/DATASET-alicfm-hclt-h256-fFEATURES-lr0.001-b64-n50000-s0/
    checkpoints/{best,last}.pt
    eval/metrics.json
    eval/media.json
    eval/population/evaluation.json
  summary.csv
  distillation_plan.json
  _launcher/status.json
  _launcher/logs/
```

Single-cell figures include `eval/comparison_marginals.png`; tracking produces
`eval/comparison_overlay.png`. Use `eval/media.json` for all saved image paths.
To create a separate experiment root, pass `--output-root PATH`.

Only syntax/static checks and metadata-only dry runs were performed while
preparing these files. Training, model evaluation, and tests were not executed.

## Complete all 15 ALI-CFM students

One local launch file covers every existing ALI-CFM distillation preset:

```bash
bash local_runs/distill_all_alicfm.sh --dry-run
bash local_runs/distill_all_alicfm.sh
```

This includes the six CITE/Multi profiles, EB, ST, knot, tracking, GOM, Beijing,
Arch, Sphere, and LiDAR. It selects framework teachers from
`/data/lthung/flow2pc/pretrained-teachers/alicfm/DATASET/` through the explicit
`--teacher-root` option. Tracking retains its declared successful legacy teacher.
The seven completed students above are reused at the same output paths; the
other eight are queued with their existing YAML student settings and budgets.
Only the seven historical winners have tuned settings recorded here; the
remaining presets are not relabeled as sweep winners.

The queue uses the same eight-GPU pool, W&B project `flow2pc`, final images,
and output root. Resolved per-run configs, including the selected teacher paths,
are saved under `_configs/` and in `distillation_plan.json`. Neither teacher
optimization nor UQ evaluation is part of this command.
