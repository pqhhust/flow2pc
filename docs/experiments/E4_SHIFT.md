# E4: shift propagation and teacher-relative uncertainty fidelity

This is the first ALI-CFM/OTP-FM phase of the plan in
`local_runs/Flow2PC_E4_Shift_Robustness_Codex_README.md`. The plan itself was not
an implementation. Code now lives in `src/flow2pc/experiments/e4/`, using the
existing experiment registry, checkpoint loaders, ODE solver and GPU queue.
Teacher code, student code and training are unchanged.

## Dataset choice and scope

Beijing is the primary 1D pilot; ST is the secondary 2D Euclidean check. Both
have completed frozen pairs and existing student fidelity evidence. This choice
precedes E4 results. Beijing models one PM2.5 variable, not a missing-coordinate
or multivariate time-series task. ST stores its coordinates in (y,x) order.

The four YAMLs pin exact `best.pt` files. ALI-CFM Beijing and both ST pairs use
the new 100-trial student winners. OTP-FM Beijing uses retained historical
winning settings; no fresh sweep under its present validation protocol is
claimed. Existing E3 results are not reassigned to these selected students.

The pilot rejects other datasets pending explicit shift/domain policies.
Sphere/terrain shifts need geometry constraints; high-dimensional single-cell
needs a declared feature/shift policy. Knot is deferred because the ALI teacher
has poor time-specific agreement, and analytic versus finite-source laws must
be distinguished. GOM is deferred after weak real-data spatial agreement.
MFM/MMFM are deferred at the user's request. Four pilot pairs do not establish
coverage of all methods, dataset families, dimensions or folds.

## Review findings and implemented decisions

- Arbitrarily shifted initial states have no observed real-world future
  outcomes here. Only **teacher-relative** mode is implemented. References are
  independent teacher ODE pushforwards from the same shifted initial law.
  `ground-truth-shifted` mode is rejected. Ordinary later observations are not
  shifted and mislabeled as intervention outcomes. Teacher self-consistency
  is not evidence of real-world calibration.
- The initial law is uniform over the declared empirical earliest population:
  retained metadata lists 744 source rows for Beijing and 755 for ST. Banks use
  independent IID draws with replacement. Source IDs can overlap; 2,000 draws
  do not create 2,000 independently observed cells. No smoothing is added.
- The PC's own density is not automatically the density pushed forward by its
  velocity from this empirical source. No validated normalized ODE density/HDR
  estimator exists for this phase. **HDR-ECE is N/A with a saved reason**.
  Neither a KDE-smoothed law nor an interval box is substituted as an HDR.
- MMD kernel widths are fixed using clean observed training rows before
  inspecting any shifted outcomes. Paired teacher/student MMD and an independent
  teacher/teacher floor have different couplings. The floor is diagnostic, not
  a pass threshold, and is not subtracted from the primary score.
- The E4 document's SD-ECE has an explicit operational definition. This phase
  implements that definition; E3.2's earlier removal of undefined SD-ECE does
  not remove this separately defined E4 interval statistic.

## Shifts, sampling and metric coordinates

Raw states are standardized using observed training rows for shifts and scores.
Saved checkpoint preprocessing remains separate and maps states into the actual
model input representation. No training normalization is changed.

Mean shifts are delta=0.25/0.5/1 along unit direction [1] for Beijing and [1,0]
for ST. The latter shifts the first stored spatial coordinate. Spread factors
are c=0.75/1.25/1.5 around the full clean source mean. Clean is a separate
condition, representing delta=0/c=1; it preserves the exact original model-space
initial states. Negative initial coordinates and movement beyond clean
coordinate ranges are logged without clipping or claiming physical validity.

Each MC seed has six independent streams: paired initial draws, independent
teacher floor, interval construction, teacher-reference observations, ES-a and
ES-b. Teacher/student and clean/shifted conditions share corresponding base
source IDs. Reference observations never tune a kernel or construct intervals.
Both roles see exactly the same independent teacher-reference samples.

Defaults: 2,000 paired draws, 2,000 interval draws, 2,000 reference draws, two
256-draw proper-score banks, seeds 0/1/2, levels 50/70/90/95%. Every distinct
source row is integrated once per condition/model; repeated draws reuse its
pushforward. Fingerprinted compressed caches preserve the same sampling law
while avoiding repeated ODE solves. Kernel duplicate compression preserves
empirical weights and diagonal mass exactly.

Both models use Dopri5, rtol=atol=1e-5 and the same physical/model time map.
Rollouts begin at the earliest time. Beijing scores indices 2/5/8/11, ST index
1; time zero is plotted but excluded from headline summaries. This is the E4
initial-law protocol, not a claim of reproducing a preceding-population-reset
benchmark. Nothing is extrapolated beyond the saved observation-time interval.

## Metrics and uncertainty

E4.1: biased multiscale Gaussian MMD squared; clean MMD squared; signed
shifted-minus-clean MMD squared; equal-bank-size independent teacher floor;
mean-shift response error (L2/sqrt(d)); componentwise variance-shift response
error (L1/d, population variance ddof=0); paired trajectory RMSE. Response
errors compare changes within each model from clean to shifted.

E4.2: central-interval SD-ECE; full-state Energy Score; mean scalar CRPS;
coverage and widths at each nominal level, including per-feature coverage.
SD-ECE averages absolute per-feature coverage errors, not absolute errors of
individual membership indicators. ES/CRPS use independent model score banks.
No Brier score or direct PC density/sample evaluation is introduced.

Per-time/seed results and clean-to-shifted deltas are saved. Summaries first
macro-average selected noninitial physical times per seed, then report mean,
MC SD, MC SE and Student-t 95% intervals across draw seeds. Trajectory RMSE uses
the square root of the mean squared per-time RMSE. These concern conditional
MC error for fixed checkpoints and finite empirical source, not independent
training seeds, observed trajectories or real-data uncertainty. Discrete atoms
can make attainable interval coverage differ from nominal. Budgets and three
seeds are pilot settings; publication-level convergence remains unverified.

## Commands and outputs

Metadata-only preview; no models, GPUs, W&B, datasets or output writes:

```bash
bash scripts/run_e4_shift_8gpu.sh --dry-run
```

Four independent frozen-pair jobs, using available GPUs from the 0–7 pool:

```bash
bash scripts/run_e4_shift_8gpu.sh
```

Append `--datasets beijing` for the 1D pilot only. `WANDB_MODE=offline` keeps SDK
logs locally; online mode also uses the existing resilient fallback. W&B project
is `flow2pc`. The multi-GPU queue requires at least two selected GPUs; use the
ordinary CLI for a single pair:

```bash
PYTHONPATH=src python main.py \
  --config configs/experiments/e4/alicfm/alicfm_beijing_hclt.yaml \
  --set device=cuda:0
```

Outputs default to
`/data/lthung/flow2pc/experiments/e4/teacher-relative-v1/METHOD/DATASET/`.
Override the batch root with `E4_ROOT` or `--output-root`. Each pair saves:

```text
resolved_config.yaml, binding.json, selection.json, status.json
run_metadata.json, samples_manifest.json, representation.json
dataset_contract.json, dataset_audit.csv, sampling.json
e41_per_time_severity.csv, e41_summary.csv
e42_per_time_severity.csv, e42_per_time_alpha.csv, e42_summary.csv
summary.csv, results.json, media.json, notes_and_limitations.md
cache/{clean,mean-*,spread-*}.npz
figures/{e41_metric_vs_severity,e41_trajectory_rmse_vs_time,
         e42_coverage_curves,e42_metrics_vs_severity}.png
```

Cache bindings cover configuration, source inputs, checkpoints and experiment
code. Different scientific inputs/code require a new output root. Failures
abort visibly in `status.json`; failed conditions are not excluded silently.
Checkpoints, raw data and model/training implementations are never modified.

Regression cases: `tests/experiments/test_e4.py`. Tests and model jobs were not
run during implementation. AST, shell syntax, configuration resolution and
metadata-only preview are separate from numerical validation.
