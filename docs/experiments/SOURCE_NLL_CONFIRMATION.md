# ST source-NLL confirmation

Prepared software, not new experimental results. No training, model evaluation
or tests were run while preparing this phase. It preserves the NLL objective,
teacher queries, PC implementation and validation velocity-MSE checkpoint rule.

## Declared candidates

Four ordinary distillation YAMLs under each teacher's
`configs/distill/<method>/source_anchor/confirmation/` pin the completed seed-0
recipes from `source-nll-refinement-v1`. No experiment preset is replaced.

| Teacher | Initial NLL weight | LR | Seed-0 validation velocity MSE | Seed-0 source validation NLL/D |
|---|---:|---:|---:|---:|
| ALI-CFM | 0.003 | 0.003 | 0.00012963 | -0.44244 |
| OTP-FM | 0.01 | 0.002 | 0.00120885 | -0.31062 |
| OTP-FM | 0.03 | 0.002 | 0.00293756 | -0.43851 |
| OTP-FM | 0.1 | 0.002 | 0.00779581 | -0.45841 |

These span the observed validation density/velocity tradeoff. The lowest-test-W1
weight 0.3 recipes are not included: alternatives had lower errors on both
validation objectives. This is an exploratory confirmation on the same ST
benchmark, not a new independent test population. No automatic final winner or
scalar combination of validation objectives is introduced.

The controller matches each YAML's scientific configuration exactly against
the reference sweep plan and requires its seed-0 result to be complete. Seed-0
students are referenced in place and skipped, not copied or retrained. Only
training seeds 1 and 2 create new students: eight new runs, twelve checkpoints
in the full comparison. Architecture, fixed Fourier feature seed, dataset
splits/preprocessing, source-NLL weight, LR, batch size and 20,000-step budget
remain fixed within each recipe. The existing runner's training seed controls
its initialization, training draws and validation-query RNG as before.

## Run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 \
  bash scripts/run_source_nll_confirmation_7gpu.sh --dry-run

CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 \
  bash scripts/run_source_nll_confirmation_7gpu.sh
```

The non-preview command trains the missing students through the existing
resource-aware queue, then runs the frozen-checkpoint diagnostics. Each stage
uses up to seven concurrent workers, one per GPU. GPU 3 is excluded. No teacher
pretraining is launched. W&B remains project `flow2pc`, with the existing
local/offline fallback. Standard final 512-sample population reports/images
remain enabled for each new student.

Default output root:
`/data/lthung/flow2pc/distill/source-nll-confirmation-v1/`.

- `students/`: the eight new student runs, generated configs, checkpoints and
  the standard comparison report covering all twelve students.
- `diagnostics/`: frozen direct-PC checks, per-worker logs and aggregate CSVs.

The same command resumes interrupted training checkpoints and skips completed
work. To run stages separately, use `--phase train` or `--phase diagnose`.
Diagnosis requires all requested students to be complete. `--summarize-only`
updates saved CSV/status reports without inference; it does not launch missing
training or diagnostics. Partial diagnostics are preserved and blocked rather
than silently overwritten. For a fresh diagnostic attempt, retain the student
root and override only the diagnostic directory:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 \
  bash scripts/run_source_nll_confirmation_7gpu.sh --phase diagnose \
  --diagnostic-root /data/lthung/flow2pc/distill/source-nll-confirmation-v1/diagnostics-v2
```

## Larger direct-PC checks

For each of the twelve checkpoints, MC seeds 101, 202 and 303 generate separate
banks of 4,096 direct PC states at each of the four ST observation times.
This gives 36 frozen-checkpoint jobs and 12,288 draws per time/checkpoint.

This stage calls no teacher or student ODE and constructs no conditional teacher
sampler. It uses the existing PC sampler, density and velocity implementations,
plus the frozen teacher's velocity at the exact sampled states. It does not use
the unfinished E3 direct-query adapter and does not claim calibrated marginal
or conditional queries.

All 4,096 states contribute to velocity MSE and tail diagnostics. No clipping,
trimming or replacement sampling is used. The worst sixteen velocity-error
points per population/time retain model/raw coordinates, teacher/student
velocity vectors, exact PC log densities and shares of total error. Reserved
observed-time validation cells are also scored; no held-out population is
substituted when their validation split is empty.

To bound transport cost, W1 uses a deterministic 1,024-point subset of each
direct bank. The observed reference is the benchmark's all-population marginal,
capped at 1,024 rows (the held-out ST marginal has 870 rows). Uniform weights
allow unequal population sizes. This is explicitly different from the existing
512-sample evaluation, so do not silently mix their W1 values in one ranking.
The held-out observations are reporting references only and never affect the
optimizer, checkpoint selection or candidate selection.

An independent direct/direct W1 at the same cap estimates sampling discrepancy;
it is not a statistical acceptance threshold. Full raw/model direct banks are
saved in `direct_samples.npz`, including points excluded from the W1 subset.
`direct_state_summaries` records coordinate minima/maxima and quantiles over the
full bank. `direct_samples.png` shows the full, uncropped scatter against the
observed reference, so an extreme sample is visible rather than hidden by axis
limits. `consistency.png` in this mode plots direct W1 and velocity errors;
its presence is not an ODE-consistency test.

## Reporting

`diagnostics/summary.csv` contains one row per checkpoint/MC seed.
`checkpoint_summary.csv` averages MC seeds within each checkpoint and reports
their sample standard deviation. It also retains the largest absolute raw
coordinate across saved banks. `recipe_summary.csv` then averages these
checkpoint means across training seeds and reports a separate training-seed
standard deviation. Three MC draws from one student are not three independent
training replications. Missing/failed MC jobs leave that checkpoint incomplete;
only fully checked checkpoints enter recipe-level means, with completed/planned
training-seed counts explicitly reported. These descriptive standard deviations
are not significance tests or calibration guarantees.

Review the standard validation MSE/source NLL, direct W1, velocity errors under
the generated PC law, and extreme-state records together. These are confirmation
checks for four fixed candidates; no YAML synchronization, student promotion or
new training objective happens automatically.
