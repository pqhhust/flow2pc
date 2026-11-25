# Student sweeps for the current frozen teachers

`scripts/run_student_sweep.py` generalizes the existing CITE student sweep to
ALI-CFM and OTP-FM. It uses the existing `DistillationQueue`, shared GPU
admission logic, and `main.py --config` distillation path. It never trains a
teacher or changes the PC implementation, velocity-MSE objective, optimizer,
initialization, teacher query sampler, coupling, preprocessing, or time mapping.

## Coverage and grid

Only pairs without a previous student sweep are included. A historical sweep
counts even when its winning settings were transferred to a newer teacher.
There are 19 pairs:

| Teacher | Datasets needing a student sweep |
| --- | --- |
| ALI-CFM (8) | EB, ST, knot, GOM, Beijing, Arch, Sphere, LiDAR |
| OTP-FM (11) | CITE-5D, Multi-5D/50D/100D, EB, ST, knot, cell tracking, Arch, Sphere, LiDAR |

Excluded because of previous student sweeps: ALI-CFM six CITE/Multi profiles and
cell tracking; OTP-FM CITE-50D/100D and GOM/Beijing. The historical OTP transport
selection records are under `/data/lthung/flow2pc/otp-fm/corrected/sweeps/`.
An explicit `--datasets` filter still excludes these already-swept pairs.

There are 100 student configurations per pair, 1,900 total:

| Student setting | Values |
| --- | --- |
| HCLT latent states | 64, 96, 128, 192, 256 |
| Fourier time features | 16, 32, 64, 128 |
| Learning rate | 0.0001, 0.0003, 0.001, 0.002, 0.003 |

The preset's control configuration is scheduled first whenever it belongs to
the requested grid. Later configurations use a reproducible shuffled order,
interleaved across methods/datasets. Existing preset budgets, seed 0, batch size,
validation frequency, and final evaluation settings are inherited. In
particular, 50,000-step presets retain 50,000 steps; 20,000-step presets retain
20,000. This is a configuration search at one seed, not repeated-seed evidence.

Frozen teachers default to `/data/lthung/flow2pc/pretrained-teachers/METHOD/`.
The framework manifest selects its checkpoint.
The teacher checkpoint hash is part of each generated run's namespace.

## Validation and selection

Within a run, the existing runner selects `checkpoints/best.pt` using its
velocity-validation score. Across configurations, the sweep minimizes that
same saved score, independently for each frozen teacher/dataset pair:

- GOM/Beijing use velocity MSE on the designated temporal validation holdouts.
- CITE/Multi/EB and ST preserve reserved observed-time row validation.
- Knot and synthetic profiles preserve their preset observed-row policy;
  Arch/Sphere middle-time populations do not select a student.
- Tracking preserves its existing worst global/late/final velocity MSE divided
  by teacher velocity energy. This is labeled as a normalized score, not raw
  MSE. Teacher, seed and validation settings stay fixed within a cohort.

Population EMD/W2, direct-PC diagnostics, images, and UQ coverage never rank
these students. This does not change the real-data validation populations into
an untouched final test. No UQ calibration procedure is added.

Only verified complete runs with a finite nonnegative selection score are
eligible. Completion uses the existing checkpoint-step, evaluation and media
checks. Winners during an unfinished sweep are provisional; the output records
the completed and planned counts. Failed configurations are not silently
counted as completed trials.

## Run commands

From the repository root:

```bash
# Metadata preview: no GPU query, Torch, output writes, W&B or model execution.
bash local_runs/sweep_students.sh --dry-run

# Start the sweep on GPUs 0–7, eight independent workers maximum.
bash local_runs/sweep_students.sh

# Example: Beijing/GOM schedules ALI-CFM only; OTP already has student sweeps.
bash local_runs/sweep_students.sh --datasets beijing gom
```

The tracked entry point is `bash scripts/run_student_sweep_8gpu.sh` with the
same options. `--method alicfm|otpfm|both` restricts the teacher family.
Previously swept pairs remain excluded. `--latents`, `--time-features`,
`--learning-rates` customize the grid; `--steps` explicitly overrides budgets.

The launcher defaults to one worker per GPU, up to eight concurrent workers,
12 GiB free GPU memory per new worker and 20 GiB filesystem headroom. The
existing queue pauses admission when storage is low, preserves active workers,
and resumes compatible checkpoints. A controller lock prevents two queues
owning the same output root. It does not stop or replace any existing jobs.

Environment overrides: `GPUS`, `WORKERS_PER_GPU`, `MAX_PARALLEL`,
`CPU_THREADS_PER_RUN`, `MIN_FREE_GB`, `MIN_DISK_FREE_GB`, `WANDB_MODE`,
`TEACHER_ROOT`, `STUDENT_SWEEP_ROOT`, and `FLOW2PC_PYTHON`. W&B project stays
`flow2pc`; run names include dataset, method, HCLT capacity/features, LR, batch
size, budget, seed and scientific-config hash. The current W&B fallback remains.
Final evaluation produces the existing dataset-specific teacher/student images.

## Outputs and chosen students

Outputs stay outside the repository:

```text
/data/lthung/flow2pc/distill/student-sweep-100/
  summary.csv
  selected_students.json
  distillation_plan.json
  _configs/DATASET/RUN.yaml
  _launcher/{status.json,logs/}
  METHOD/DATASET/
    best_models.json          # Top ten by velocity-validation score
    selected_student.json     # Current winner and selection scope/counts
    best_config.yaml          # Exact winning run config, including its checkpoint directory
    RUN/checkpoints/{best,last}.pt
    RUN/eval/{metrics.json,media.json,...}
```

The controller records winners automatically but does not overwrite tracked
presets or redirect existing E3.1 experiments. `best_config.yaml` references the
winning run's existing output directory; it is not a fresh-training recipe.
Use the selected `checkpoints/best.pt` for subsequent experiments. Refresh
rankings without workers using `--summarize-only` with the same grid/dataset list.

After all 1,900 runs completed, the 19 corresponding tracked
`configs/distill/{alicfm,otpfm}/*_hclt.yaml` presets were manually synchronized
with their winning `best_config.yaml` files on 2026-11-21. Each preset retains
the exact winning configuration and existing run output directory, with source
and selected-step comments. This manual snapshot does not change the
controller's automatic behavior. Override `output.dir` for a new replication.

The existing E3.1 configs still search their original student directories and
rank available candidates by EMD. To use a new MSE-selected winner without
reversing this selection, point `selection.student_roots` at that single
winning run directory, not the entire sweep dataset directory. The winner's
teacher must match the experiment's frozen teacher. Existing experiment results
and bindings remain unchanged; use a new experiment output directory.

Regression cases cover coverage, teacher rebinding, protocol preservation,
MSE ranking when EMD disagrees, tracking score labels, and metadata-only preview.
Tests and model jobs were not run during preparation; syntax and metadata-only
dry runs are separate from numerical verification.
