# Initial-density anchoring pilot

Implemented software; the completed ST pilot shows improvement but does not
establish reliable direct-PC queries. See the [saved-result audit and optional
tail-recording commands](SOURCE_DENSITY_AUDIT.md). No runs were launched
while preparing the software. The teacher checkpoints and all PC/student-model files are
unchanged. This phase covers the ALI-CFM/OTP-FM distillation runner only;
MFM/MMFM retain their existing velocity-only objective.

## Objective

`mean((v_PC(x,t) - v_teacher(x,t))**2)
 + source_nll_weight * mean(-PC.log_prob(x0, t0)) / D`

Here `x0` comes exclusively from the earliest observed **training** population,
`t0` is its saved model-normalized time, and `D` is the state dimension. Both
terms use the teacher's saved input coordinates. The NLL does not refit
preprocessing, pair cells across times, sample the PC, or query the teacher.
NLL may be negative for continuous densities; it is not a probability or MSE.
Its absolute value depends on coordinate units, so compare lambdas within the
same teacher/data representation. Do not pool NLL values across representations.

`distillation.source_nll_weight` is optional and defaults mathematically to zero.
Omitting it preserves the old velocity-only binding and update path. Explicit
zero enables source-density validation for the ablation without computing a
training NLL. Positive values explicitly change the objective to
`teacher_velocity_MSE_plus_initial_NLL`; all objective settings enter the resume
binding. Changing lambda requires a fresh run directory.

Source batches have the existing training batch size and use their own seeded,
checkpointed RNG. They cannot consume the teacher-query RNG stream. The source
initializer, topology, optimizer, velocity queries and PC implementation remain
unchanged. There is no additional NLL at later observation times.

## Validation and selection

Source NLL is evaluated on a fixed subset of earliest-time **val** cells, with
at most `validation_batches * validation_batch_size` rows. There is no fallback
to train/test rows if that split is empty. The subset uses a separate fixed seed.
The initial score is saved before optimization in `source_density_initial.json`.
Subsequent scores are logged to the existing metrics/W&B stream.

`best.pt` still uses the existing dataset-specific validation **velocity MSE**.
Its source-validation NLL is also saved in the checkpoint and final status.
The pilot does not automatically choose lambda. Compare validation velocity
error and source-validation NLL first; final population metrics/coverage are
diagnostics, not a lambda-selection criterion. Initial NLL alone does not
guarantee correct later marginals or conditionals. A large weight can damage
velocity fit, and unconstrained Gaussian mixture likelihoods can exhibit
degenerate scales; inspect density and numerical diagnostics rather than
assuming that a decreasing training objective proves success.

## Eight-run ST comparison

Separate configs retain the selected teacher/student recipes:

- `configs/distill/alicfm/source_anchor/alicfm_st_hclt.yaml`
- `configs/distill/otpfm/source_anchor/otpfm_st_hclt.yaml`

The shared GPU queue runs lambda `{0, 0.01, 0.1, 1}` for each teacher: eight jobs,
one per GPU by default. The existing 20,000-step budgets, per-method learning
rates, architecture and samplers are retained. Intermediate population ODE
diagnostics are disabled in these pilot configs; source/velocity validation
remains active. End-of-run population evaluation and comparison images remain
enabled, including direct PC sampling and student/teacher ODE diagnostics.
Each run saves `best.pt`/`last.pt`; these are new training runs, not fine-tuning
the failed retained students. Teacher pretraining is never launched.

```bash
bash scripts/run_source_nll_pilot_8gpu.sh --dry-run
bash scripts/run_source_nll_pilot_8gpu.sh
```

Default output: `/data/lthung/flow2pc/distill/source-nll-pilot-v1/`.
The controller uses W&B project `flow2pc`, dataset/method/lambda-specific names,
the existing logging fallback and resource-aware queue. Use `--mode offline`
when needed. `--summarize-only` refreshes `summary.csv` without model inference.
The summary contains validation MSE, source-validation NLL at its selected step,
initial NLL, direct-PC population W1, ODE W1 and image/checkpoint paths. No
existing selected YAML, checkpoint or experiment export is overwritten.

## ST NLL and learning-rate refinement

The refinement keeps exactly the same objective and teacher query distributions:
teacher-query velocity MSE plus initial training-population NLL/D. PC-generated
states are not used for training. The proposed additional PC-state velocity-MSE
ablation has been removed from the runner, objective helpers, configuration
schema and launch recipes. Density/velocity consistency and tail-recording
tools remain evaluation-only diagnostics.

The existing pilot controller now accepts `--lr-factors`. Its default `[1]`
preserves the original eight pilot run identities, checkpoint bindings and
resume paths. The refinement wrapper uses a separate external output root:

```bash
bash scripts/run_source_nll_refinement_8gpu.sh --dry-run
bash scripts/run_source_nll_refinement_8gpu.sh
```

The second command launches training and final evaluation; it is provided for
the user to run. No training or evaluation was executed while preparing it.

| Axis | Values |
|---|---|
| Teachers | Existing frozen ALI-CFM ST and OTP-FM ST |
| Initial NLL weight | 0, 0.003, 0.01, 0.03, 0.1, 0.3, 1 |
| LR factor | 1, 1/3, 1/10 of each method's pilot learning rate |
| ALI-CFM learning rates | 0.003, 0.001, 0.0003 |
| OTP-FM learning rates | 0.002, approximately 0.000666667, 0.0002 |
| Budget | 20,000 steps, batch size 64, seed 0 |
| Student | Existing dense Gaussian HCLT, 256 latents, 32 Fourier features |
| Parallelism | GPUs 0,1,2,4,5,6,7, up to 7 workers, one worker per GPU; GPU 3 excluded |

This gives 21 runs per teacher and 42 total, including velocity-only controls
at all three learning rates. All models start from the same per-method
initialization/query seeds rather than fine-tuning a previous pilot student.
Only NLL weight and learning rate vary. No data representation, topology,
student implementation, teacher, sampler, clipping rule or selection criterion
is changed. The existing cosine schedule scales with the configured initial LR.

Outputs go to `/data/lthung/flow2pc/distill/source-nll-refinement-v1/`.
Generated run configs, `best.pt`/`last.pt`, final population reports, direct-PC
comparison images, and logs stay there. W&B project is `flow2pc`; names include
ST, teacher method, NLL weight, learning rate, seed and recipe signature.
The existing local/offline fallback and resource-aware GPU queue are reused.
The refinement wrapper explicitly defaults to the seven listed GPUs even if
the general pilot wrapper's `GPUS` environment setting includes GPU 3. Its
historical `_8gpu.sh` filename is retained for command compatibility. The
scheduler uses physical GPU indices and gives each worker its selected GPU
UUID as `CUDA_VISIBLE_DEVICES`, avoiding logical device-index ambiguity.
Run the same refinement command after interruption: completed runs are skipped
and incomplete runs with `last.pt` resume in place. Do not change the output
root, weights or learning-rate factors when continuing this grid.
`--summarize-only` refreshes CSV/status from saved artifacts without inference.
Do not use the original pilot wrapper for summarizing this larger grid:

```bash
bash scripts/run_source_nll_refinement_8gpu.sh --summarize-only
```

`best.pt` remains selected by the existing validation velocity MSE. `summary.csv`
includes that score, source validation NLL at the selected step, direct-PC W1
against data, direct-PC W1 against the teacher at held-out times, teacher W1
references, and checkpoint/media paths. Direct-PC distances are reporting
diagnostics; no script automatically chooses a lambda or changes experiment
presets from held-out W1. Inspect the per-teacher validation velocity/density
tradeoff first. Source NLL alone does not establish accurate later marginals,
conditionals or controlled tails. Later repeated-seed/large-sample checks for
finalists are separate work, not claimed by this single-seed sweep.

A focused follow-up now prepares eight new students and larger direct-PC
checks while reusing four completed seed-0 runs. See
[the repeated-seed confirmation protocol](SOURCE_NLL_CONFIRMATION.md).

## Density–velocity consistency gate

Run this separately on an existing checkpoint **before interpreting the NLL
pilot as a fix**:

```bash
PYTHONPATH=src /data/nvidia/conda_envs/uffm/bin/python \
  scripts/check_pc_density_consistency.py \
  --config configs/distill/alicfm/alicfm_st_hclt.yaml --dry-run

PYTHONPATH=src /data/nvidia/conda_envs/uffm/bin/python \
  scripts/check_pc_density_consistency.py \
  --config configs/distill/alicfm/alicfm_st_hclt.yaml --device cuda:0
```

Use the OTP-FM ST config for the other teacher or `--checkpoint /path/best.pt`
to inspect a pilot checkpoint with its matching teacher config. Dry-run reads
configuration/checkpoint identity only; it does not load models/data or query
GPUs. Running without dry-run **does evaluate the model**.

The diagnostic compares direct `q_t` samples with the student ODE started from
its own `q_0` samples. A separate independent direct/direct W1 records finite
Monte Carlo discrepancy. An additional empirical-source student ODE comparison
isolates the effect of changing the initial law. None are paired real-cell
trajectories. Source-validation NLL uses the saved input coordinates; W1 uses
raw feature coordinates. Outputs include CSV/JSON and a comparison figure under
`/data/lthung/flow2pc/diagnostics/pc-density-consistency-v1/`, with W&B publication
through the existing logger.

The direct/direct discrepancy is not a statistical acceptance threshold.
Increase sample size and tighten ODE tolerances if the consistency residual is
material. Agreement supports the density–velocity construction numerically;
it does not show that either distribution matches the teacher or observations.

The earlier E3 direct-query edits remain a separate, unvalidated work-in-progress.
This pilot uses the established population evaluator and does not depend on
those new E3 runners or the new direct-query adapter.

## Parallel consistency and velocity-support checks

After the source-NLL pilot completes, prepare four explicit diagnostic cases:
lambda zero and 0.1, each for ALI-CFM/OTP-FM. This is a diagnostic comparison,
not automatic selection using held-out performance.

```bash
bash scripts/run_pc_consistency_pilot_8gpu.sh --dry-run
bash scripts/run_pc_consistency_pilot_8gpu.sh
```

The existing resource-aware queue uses the available GPU pool 0–7, with at most
four concurrent workers and one worker per GPU. It launches **evaluation only**.
Checkpoint identity, sample/solver budgets and implementation hashes determine
each output path. Default root:
`/data/lthung/flow2pc/diagnostics/source-nll-pilot-v1/`.
W&B remains `flow2pc`, with checkpoint-specific dataset/method names.
The default per-batch NFE cap is 10,000 in this batch (the standalone checker
retains its configurable 50,000 default). Budget exhaustion is an explicit
numerical failure, not evidence that the density formula is wrong.

Each worker now evaluates teacher–student velocity errors separately on:

- direct PC samples at the selected time;
- student ODE states transported from its own PC initial distribution;
- student ODE states transported from empirical source training cells;
- teacher ODE states transported from the same empirical source cells;
- reserved observed-time `val` cells, where available.

All rollouts in this diagnostic start at the earliest time, without the ST
benchmark's preceding-observed-time resets. The teacher-trajectory populations
are therefore explicitly distinct from those benchmark metrics. Teacher
interpolant/OTP conditional samplers are not instantiated by this checker.
At held-out times, empty `val` populations are marked unavailable; the checker
does not substitute temporal test data. These diagnostics neither modify
teacher-query sampling nor update the student.

`velocity_support.csv` reports mean MSE, relative RMSE against teacher RMS
speed, and median/p90/p99/max per-state coordinate-RMS errors. No tails are
clipped, removed or reweighted. Velocity units are the saved input coordinates
per unit model-normalized time; W1 remains in raw state coordinates.
A zero teacher-speed denominator gives N/A relative error.

Failed numerical rollout phases are recorded in `numerical_errors.json` and
`results.json`; other available populations are still scored. A report with
such errors is **not** marked successful. Contract/programming errors propagate
as failed jobs. No failed bank is replaced with empirical-source particles.
The controller writes `summary.csv`, `_launcher/status.json` and per-worker logs.
It does not automatically retry or archive partial diagnostics.

Inspect both the density-consistency discrepancy and velocity support before
deciding on another training change. Poor velocity matching on PC-generated
states is consistent with a coverage/approximation problem; it is not proof of
its cause or authorization to change
the teacher-query distribution or train on PC samples.
