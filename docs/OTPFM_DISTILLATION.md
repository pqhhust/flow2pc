# Restored OTP-FM distillation

The successful historical implementation is at Git commit `f2c6942`, in
`repc/otpfm.py`, `repc/otpfm_data.py`, `repc/otpfm_student.py`,
`scripts/train_otpfm.py` and `README_OTPFM_DISTILLATION.md`.
Its teacher-side conditional path is now implemented in
`src/flow2pc/distill/otpfm_sampling.py` and dispatched through the existing
`flow2pc.distill` runner. The retired standalone trainers are not required.

## What is preserved

Student construction, empirical source initialization, the HCLT implementation,
AdamW with weight decay 0.01, cosine learning-rate decay, gradient clipping,
velocity-MSE loss and best-checkpoint loading use the current shared code.
No student parameter is frozen by this port. There is no source-likelihood fit,
source-NLL objective, anchored-source variant, extra boundary loss, scale clamp,
or change to the PC mathematics. The new presets copy each corresponding
existing ALI student and training configuration, replacing the teacher sampler.
Historical student checkpoints are not automatically converted or resumed.

Each update constructs a conditional OTP state and queries the frozen teacher:

```python
x_t = conditional_otpfm_path(observed_tuple, t, fixed_alpha)
target = teacher.model.v_func_ema(x_t, t, t)
loss = (student.velocity(x_t, t) - target).square().mean()
```

The conditional path's derivative is not the target. The off-diagonal MeanFlow
prediction and finite-step native sampling are also not substituted for the
instantaneous diagonal EMA field. No ODE is used to construct training batches.

The position-only sampling path skips the unused conditional derivative. For
box potentials, it evaluates all time factors together using the native
piecewise expressions, removing the two Python CUDA-scalar waits per potential
per batch. Potential constants and the LU factorization are prepared once;
tuples and times are still drawn afresh. Other potential shapes keep their
native position formulas. No training population, coupling, time distribution,
alpha, teacher target, student architecture or optimizer setting changes.
The derivative remains available explicitly for diagnostics. Numeric parity
regressions cover native forces, both position/derivative paths, and all 118
knot support boundaries in float32/float64; these tests have not been run.

Gaussian time factors are also evaluated together, including the 113-potential
tracking recipe. Their native CDF/PDF expressions and operation ordering are
retained. Knots, widths, normalizers, zero-time CDF/PDF values and the native
Python-computed endpoint constants are prepared once on the teacher device.
The full-strength LU factorization is still prepared once; each fresh independent
tuple still gets its own solve. No pool of conditional trajectories replaces
fresh training queries. Float32/float64 regression cases compare all 113 factors
and the returned states/teacher targets against the native expressions, and
check that the RNG draw order is unchanged. These new cases have not been run.

### Distillation performance diagnostics

The shared runner enables the following execution changes for OTP-FM only:

- Intermediate population diagnostics reuse a run-local CPU cache of frozen
  teacher rollouts and teacher EMD/W2 values. Keys include exact source states,
  time grids, solver settings and comparison populations. Teacher tensor
  replacement/version changes invalidate the cache. Student ODE predictions and
  direct PC samples are recomputed at every diagnostic checkpoint. Numerical
  failures are not cached. No persistent cache or training tuple cache is added.
- The dataset population view is materialized once for these repeated checks.
- Tracking population EMD/direct-PC diagnostics use the six configured
  `evaluation.time_indices`, rather than all 115 frames. Velocity-MSE selection,
  including late/final tracking queries, is unchanged. The ODE solver still uses
  the same full time grid and tolerance. Other datasets keep their existing
  intermediate population scoring times.
- At logged/validation updates, synchronized elapsed timings are saved locally
  and to W&B as `timing/query_sampling_seconds`,
  `timing/teacher_target_seconds`, `timing/student_update_seconds`,
  `timing/velocity_validation_seconds`, `timing/population_diagnostics_seconds`
  and `timing/checkpoint_io_seconds`. Cache hits/misses are logged too. These are
  sampled wall times under the actual shared-GPU workload, not averages over all
  updates or exclusive GPU kernel times. Synchronization is limited to those
  sampled updates; the default logging interval is 100 steps.

These changes leave `src/flow2pc/pc/`, `distill/student.py`, velocity-MSE math,
optimizer/scheduler, teacher source and configurations untouched. They do not
remove external GPU contention or change the selected teacher network size.
Running Python workers keep the code they already imported: the optimization
applies to newly started workers. Known conditional-v1 sampler hashes are
explicitly allowlisted for resume, while all teacher/data/student/training
binding checks and saved best-checkpoint selection history remain enforced.
Regression tests are in `tests/test_distill_performance.py` and
`tests/test_otpfm_distillation.py`; training/evaluation/tests were not run during
this edit.

## Recovered details that matter

- The full-strength W2Inf linear system is factored once. Alpha multiplies its
  resulting position/velocity correction outside that solve, matching native
  `forward_with_loss`. Residual and endpoint identities are checked.
- Alpha is fixed from the selected teacher checkpoint. Older final-epoch shared
  exports recover it from their recorded update count and native curriculum at
  the last teacher update. Unknown metadata requires an explicit original alpha;
  it does not silently select 1 or restart a curriculum during student training.
- A native OT teacher uses squared-cost POT row-argmax chains, retaining duplicate
  target indices for unequal marginal sizes. Saved native chains are reused when
  querying the complete matching pools. Coupling on smaller reserved-row pools
  is recomputed using the same rule and recorded as the student query protocol.
- Independently coupled teachers draw fresh independent observed tuples. They
  reuse the same LU factorization, with tuple-dependent forces computed for each
  batch. A fixed OT coupling is not imposed on these teachers. Analytic knot
  training retains fresh independent noise, as recorded in its teacher metadata.
- Real held-out-time observations never construct training tuples. The shared
  dataset owns the student train/validation rows and saved preprocessing. No
  Gaussian source, rescaling or new time normalization is introduced.
- Native rounded potential knots and exact dataset observation times stay
  separate. The sampler uses the saved potential knots; scoring uses the dataset
  time grid and saved coordinates.

This port supports the direct **W2Inf potential** used by both the native `W2`
and `W2Inf` recipes and all ten selected shared-data teachers. Auto-selected
native direct solvers are accepted. KL/MMD nonlinear conditional paths fail
clearly; the historical fast cache did not implement them. Empty-potential,
two-observation teachers use their native straight conditional path.

## Validation and evaluation

The current approved dataset policy is retained: transport uses held-out-time
velocity validation; single-cell/ST/knot/tracking use their existing observed-row
policy, including the existing late/final tracking checks. Fixed validation
queries use OTP sampling rather than an ALI generator. Validation velocity MSE
selects `best.pt`; EMD does not select the student.

The historical successful GOM/Beijing sweep selected using a 95/5 source-row
tuple split. That protocol is documented in Git history; it is not silently
restored over the subsequently approved dataset-specific policy. These presets
therefore do not claim to replay its numerical scores. A teacher's training
subset budget is not reused as a new student validation split: the student keeps
the existing dataset train/validation rows, as the shared ALI runner does.

Teacher/student ODE predictions and direct-PC samples remain separate evaluation
modes. The shared final evaluator uses EMD/W1 in all dimensions, with W2 also
reported for GOM/Beijing. It reloads the selected student, uses empirical observed
source cells for ODE comparison, and saves dataset-specific figures, including
ST/tracking overlays. Direct PC sampling is diagnostic, with no extra loss.

The current PC has the existing free initial density: initialization does not
assert an exactly fixed source distribution. ODE population fidelity from the
same empirical initial cells and direct-PC density fidelity remain distinct.

## Configs and commands

For the 100-config-per-dataset CITE-50D/CITE-100D student tuning plan, see
[the focused CITE sweep](OTPFM_CITE_STUDENT_SWEEP.md). Its local launch recipe is
`bash local_runs/sweep_otpfm_cite.sh`; it uses a separate output root and the
same velocity-only student training.

For the remaining selected datasets in parallel, use the existing GPU queue:

```bash
# Metadata preview only; no GPU query, writes or model execution.
bash scripts/run_otpfm_distill_8gpu.sh --dry-run

# Launch the ten selected-teacher configs, skipping verified completed runs.
bash scripts/run_otpfm_distill_8gpu.sh
```

The machine-specific all-preset recipe is
`local_runs/distill_all_otpfm.sh` (gitignored). It explicitly includes all 17
supported HCLT presets: CITE/Multi at 5D/50D/100D, EB, ST, knot, cell tracking,
GOM/Beijing with both W2/W2Inf reference teachers, and Arch/Sphere/LiDAR. Run
`bash local_runs/distill_all_otpfm.sh --dry-run` to preview it, or omit
`--dry-run` to launch. Its defaults are GPUs 0–7, one worker per GPU, eight
workers total, and the existing output root. It skips complete students and
resumes compatible checkpoints. New students keep final teacher/student
comparison images and W&B project `flow2pc`. This recipe covers existing
supported distill presets; KL/MMD conditional samplers are not implemented.

The reusable `scripts/run_otpfm_distill_8gpu.sh` defaults to GPUs 0–7, up to
two workers per GPU and 16 workers total, subject to
12 GiB free memory per new worker and 20 GiB disk headroom. It schedules one
student per selected teacher; it is not a new hyperparameter sweep. The completed
CITE-50D run is reused, leaving nine new runs at the time this batch was prepared.
Student training settings and selected teacher checkpoints come from the presets.

`GPUS`, `WORKERS_PER_GPU`, `MAX_PARALLEL`, `MIN_FREE_GB`, `CPU_THREADS_PER_RUN`,
`WANDB_MODE`, `FLOW2PC_PYTHON` and `OTP_DISTILL_ROOT` override launcher defaults.
For example, `GPUS=0,1,2,3 MAX_PARALLEL=8` selects four GPUs with up to two
workers each. A busy GPU is skipped until enough memory is free. The controller
uses `nvidia-smi` and never allocates a CUDA context.

The batch keeps all outputs under `/data/lthung/flow2pc/distill/otpfm/` and writes
`summary.csv`, `_launcher/status.json`, per-worker logs under `_launcher/logs/`,
and the resolved batch in `distillation_plan.json`. Existing model subdirectories
are preserved. Repeating the command skips verified complete runs and resumes
compatible checkpoints; if their updates already finished, only the normal final
evaluation remains. An incomplete directory without a checkpoint requires
`--retry-incomplete`, which archives it under `_attempts/` before restarting.
Only one controller may own the output root at a time.

W&B stays on project `flow2pc`; online startup failures use offline/local logging,
including the SDK's busy-run-ID error during offline fallback. Final teacher/
student ODE comparison images and direct-PC diagnostics use the current evaluator.

Use `--datasets cite5 cite100 multi5 multi50 multi100 eb st knot cell_tracking`
to select only those nine explicitly. The optional `gom_w2`, `gom_w2inf`,
`beijing_w2`, `beijing_w2inf` reference-teacher configs are included only when
requested with `--datasets`. `--summarize-only` refreshes summaries without
starting workers. This launcher was prepared without executing model jobs or tests.

The selected synthetic teachers also have `otpfm_{arch,sphere,lidar}_hclt.yaml`
presets. Request them explicitly with `--datasets arch sphere lidar`; the default
batch remains the ten single-cell/ST/knot/tracking profiles. Their selection
keeps the middle populations out of tuning, as recorded in
[the synthetic teacher report](SELECTED_TEACHERS.md#selected-synthetic-teachers).

All selected-teacher presets are under `configs/distill/otpfm/`:

```text
otpfm_{cite5,cite50,cite100,multi5,multi50,multi100,eb}_hclt.yaml
otpfm_{knot,st,cell_tracking}_hclt.yaml
otpfm_{gom,beijing}_{w2,w2inf}_hclt.yaml
```

The first ten load the selected existing sweep checkpoints indexed in
`docs/selected_teachers.json`. The transport presets load existing migrated
seed-42 reference teachers, not newly selected winners or historical seed-44
artifacts. Override `teacher.checkpoint` for another compatible migrated teacher.
The checkpoint's manifest restores its model, data, time and preprocessing.

Prepared Arch/Sphere/LiDAR snapshots may specify `teacher.data_root` when their
directory has moved. The three presets now point to
`/data/pqhung_group/pdtung/flow2pc/data/DATASET/seed42`. Restore accepts only a
location change: the prepared manifest identity, every array checksum, saved
split indices, normalization and time grid must still match the frozen teacher.
The original teacher artifact is untouched, and the student records its actual
data directory. The parallel launcher checks the prepared files before starting
a worker. Missing data previously caused these three runs to exit before any
student checkpoint was written. `bash local_runs/rerun_failed_otpfm.sh` retries
them while retaining the full 17-run summary and skipping completed students.

Commands for the user to execute, from the repository root:

```bash
PYTHONPATH=src /data/nvidia/conda_envs/uffm/bin/python main.py \
  --config configs/distill/otpfm/otpfm_cite5_hclt.yaml

PYTHONPATH=src /data/nvidia/conda_envs/uffm/bin/python main.py \
  --config configs/distill/otpfm/otpfm_knot_hclt.yaml

# Checkpoint/sampler setup inspection; no training or teacher forward passes.
PYTHONPATH=src /data/nvidia/conda_envs/uffm/bin/python main.py \
  --config configs/distill/otpfm/otpfm_cell_tracking_hclt.yaml --validate-only

# Reevaluate an already trained selected student.
PYTHONPATH=src /data/nvidia/conda_envs/uffm/bin/python main.py \
  --config configs/distill/otpfm/otpfm_cite5_hclt.yaml --evaluate student
```

The existing `scripts/run_local_data.py` also accepts OTP `distill`/`both` for
W2/W2Inf. That launcher explicitly overrides the teacher/output paths to its
local run namespace; use `--teacher-checkpoint` when consuming an existing model.
Direct `main.py` invocation uses the selected preset checkpoint as written.

Presets log to W&B project `flow2pc`, with dataset/method/student/hyperparameters
in the run name. Outputs are under
`/data/lthung/flow2pc/distill/otpfm/DATASET_OR_RECIPE/`, including:

```text
checkpoints/{best,last}.pt
manifest.json                 teacher/data/query binding, including fixed alpha
resolved_config.yaml
metrics.jsonl
eval/{metrics.json,media.json,comparison_*.png,...}
eval/population/{evaluation.json,comparison.png,media.json,...}
```

Resume retains the existing strict teacher/data/sampler/model/training binding,
optimizer, scheduler and RNG state. Changing the checkpoint, coupling, alpha or
sampler implementation requires a new output directory.
The original conditional-v1 source and its position-only/vectorized-box and
vectorized-Gaussian revisions
are the sole explicit exception: `otpfm_compat.py` lists their exact hashes and
still requires every other binding field to agree. Resume retains the original
binding/best-checkpoint identity and separately records the current sampler
source in `sampler_execution_sha256`. Unknown source revisions remain rejected.

The CITE-5D and Multi-5D students reached step 20,000 but their final comparisons
failed with `Found array with dim 3, while dim <= 2 is required`. The paired
trajectory diagnostic sent `[time, cells, features]` to native inverse whitening,
whose sklearn implementation accepts only 2D rows. Inverse preprocessing now
reshapes leading dimensions into rows and restores the original shape after the
same float32 operations. Training queries, velocity MSE, historical EMD units
and saved checkpoints are unchanged. Regression coverage exercises the full
whitened teacher/student evaluator in both historical and raw metric spaces;
it has not been executed.

After the current controller and its workers exit, rerun the existing launcher
to refresh all ten run summaries and resume only the incomplete runs:

```bash
bash scripts/run_otpfm_distill_8gpu.sh
```

The two failed students already have `best.pt` and `last.pt` at the configured
final step. Their normal resume skips optimizer updates and performs final
evaluation/publication; completed runs are skipped. Do not start a second
controller against this output root while its existing lock is owned.

Regression cases are in `tests/test_otpfm_distillation.py` and the existing
validation/selection tests. Only static inspection, metadata comparisons and
syntax checks were authorized for this port. No training, distillation,
evaluation, test suite, or numerical model-parity check was run.
