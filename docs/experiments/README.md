# Research experiments

Experiment implementations live together in `src/flow2pc/experiments/`.
The registry in `__init__.py` dispatches evaluation-only experiments through
the existing `main.py --config` CLI. Shared modules own configuration,
frozen-model adapters, checkpoint selection, result writing and the GPU queue.
Each experiment owns its metrics, figures and runner in a subfolder. Future
experiments add a registered module and YAMLs, rather than dataset scripts or
branches in teacher/student training.

```text
src/flow2pc/experiments/
  __init__.py             # lazy experiment registry
  config.py              # shared evaluate-only configuration
  adapters.py            # frozen ALI/OTP + HCLT loading and ODE sample banks
  analytic_knot.py       # explicit analytic knot source/reference law
  selection.py           # EMD ranking, checkpoint bindings, provenance
  io.py                  # atomic result writers, usable without Torch
  parallel.py            # shared multi-GPU experiment controller
  e3_1/{config,metrics,runner,figures}.py
configs/experiments/e3_1/{alicfm,otpfm}/
scripts/run_experiments.py
tests/experiments/
```

The E3.1 implementation is prepared, **not experimentally validated**. No
training, calibration evaluation or test suite was executed during its creation.
Do not interpret the software's presence as a measured calibration result.

## E3.1: marginal calibration

This implements E3.1 from the local `Flow2PC_E3_UQ_README_download.md`, restricted
to ALI-CFM and OTP-FM first. E3.2, MFM/MMFM adapters and additional student tuning
are future work. Student mathematics, velocity-MSE training, teacher training,
and distillation checkpoint selection are unchanged. The scientific question is
whether the **ODE-generated** student distribution preserves its teacher's
calibration against real observations. Internal PC log probabilities and direct
PC sampling are not used as the ODE distribution's density or sampler.

### Select one pair, not every student

Each YAML pins a teacher from the newly pretrained best-recipe collection and
searches explicit student directories for that exact teacher. ALI-CFM tracking
retains the successful historical teacher and its hash-pinned metadata.
OTP-FM transport uses the W2 recipe. No teacher is retrained or silently replaced.

Within the candidate roots, select the lowest **saved student ODE W1 EMD**:
held-out-time mean when available, otherwise the noninitial-time mean. Direct-PC
EMD and W2 are not substituted. Candidate report units, solver/sample protocol,
teacher hashes, completed status and evaluated checkpoint steps must agree.
The existing `best.pt` within each run was selected by velocity MSE; intermediate
EMD values whose weights were not retained cannot be recovered. EMD ranks those
retained checkpoints **across runs**, not unsaved training steps.

The initial presets search the new `selected-students/<dataset>/` cohorts. Where
only one matching run exists, it is the only candidate; this is not a claim of a
new exhaustive search. Additional comparable runs can be added through
`selection.student_roots`. The controller pins the chosen checkpoint when a
batch starts; later completions cannot change an active task. `selection.json`
records candidate scores, rejected runs, the chosen score, both selection rules
and checkpoint identities. Foreign-teacher students are rejected even when their
data names or hyperparameters match.

### Metrics, representations and sample separation

- Coverage at 50%, 70%, 90% and 95%; 90% coverage is a headline, **target 90%**.
- HDR-ECE: mean absolute coverage error over those four levels, then a macro
  average across selected physical times.
- Full-state Energy Score, on coordinates standardized using observed training
  rows, alongside the calibration metrics. This measures distribution quality;
  it is not itself proof of calibration.
- Teacher/student values and student-minus-teacher differences are retained.

Regions use a Gaussian KDE fitted to model samples. Independent model draws tune
bandwidth by predictive log score; another independent model bank sets density
quantile thresholds. Real evaluation observations are used **only for scoring**.
These are estimated, smoothed KDE ranking regions, not exact flow HDRs. Nominal
mass and density-score ties in the threshold bank are logged. No recalibration
correction is fitted to real observations. The HDR motivation follows
[Chung et al., ICML 2024](https://proceedings.mlr.press/v235/chung24a.html); the
specific estimator, discrete-level ECE and numerical protocol are this project's
declared implementation choices.

In empirical-source mode, teacher and student use identical earliest-marginal
source draws and ODE settings. Sampling with replacement uses independent RNG streams for
fit/tuning/threshold/energy banks. Repeated source IDs are expected under an
empirical law. Because the ODE is deterministic, each requested source row is
integrated once and its pushforward reused; no noise, new initial law or invented
cross-time observation pairing is introduced. Rollout starts at the **earliest
time**, even where the historical selection EMD used a preceding-observed-time
reset. Both protocols are saved separately; their numerical scores are not
described as the same experiment.

Default checks use 5,000/10,000 fit-and-threshold banks, 2,048 bandwidth-validation
draws, two independent 2,048-draw Energy Score banks, and three MC seeds. These
seeds measure sampling variability for fixed checkpoints, not independent
training seeds. Solver is Dopri5, tolerance `1e-5`. Equal ODE settings and source
draws are used within each pair. No large sample banks/checkpoints are copied
into experiment outputs.

Beijing is 1D and GOM is 2D. Single-cell HDRs use a declared training-fitted 2D
PCA projection; Energy Score still uses every state coordinate. Sphere and LiDAR
also use explicitly projected KDE regions, not claims about an exact intrinsic
surface density. Remaining low-dimensional profiles use full-state KDE ranking
regions. The projection, scales, axes and hash are saved. Compare Energy Scores
or projected HDRs across methods only after checking that their recorded scoring
representations match; the controller never pools different datasets/scales.

### Existing data, no retraining

The saved-population mode is a **retrospective calibration diagnostic** for existing
checkpoints. Some evaluation populations have already influenced EMD/model
selection, velocity validation or native preprocessing. Calling a partition
`test` in a loader does not make it an untouched final test. This reuse is
recorded in every dataset audit and in selection/result metadata. This phase
does not retrain models, change literature splits, or claim independent-test
generalization.

Default evaluation uses held-out physical times when they exist; otherwise it
uses reserved rows at noninitial times. The `protocol` reference policy prefers
reserved `test` rows and uses `val` only when a time has no test rows. There is
**no fallback to training/all rows**. A config may explicitly select another
reference split, which is recorded. Tracking uses README frames 1/25/50/75/100/115;
frame 1 defines the initial law and is excluded from transported-distribution
metrics. Its other five frames are scored. Independent-cell versus same-object
tracking interpretation must still be respected.

Each dataset/time is audited before sampling. Coverage with fewer than 20
reserved observations is marked N/A, not zero or artificially precise. In
particular, the saved knot snapshots have only one reserved observation per time;
coverage is N/A when evaluating those finite snapshots. The knot presets now use
the explicit analytic mode below instead.
Empty reserved populations are recorded and excluded. Appropriate independent
group identities have not been established for these prepared populations, so
no observational/bootstrap confidence intervals are invented. Figures show MC
seed ranges and tables report MC standard deviations, explicitly labeled as such.

### Analytic knot evaluation (existing checkpoints)

The knot presets now explicitly set **both** `source_split` and `reference_split`
to `analytic_knot`. This is an evaluation-law choice, not a change to the teacher,
student loss, distillation queries, dataset splits or stored checkpoints. Both
saved knot teachers use fresh analytic noise in their native training loops;
the ten-point saved snapshot pools are a separate evaluation/student-query
contract. This mode addresses their inadequate coverage-test sample counts.

The existing shared `knot_means` function supplies the exact saved time grid and
the middle-branch `+1` shift. Saved `noise_std` supplies independent coordinate
noise. At time zero we draw from the known analytic distribution
`N(knot_mean[0], noise_std**2 I)`, then integrate the existing teacher/student
velocity fields. This **explicitly replaces the finite empirical evaluation
initial law**. `analytic_protocol.json` records both laws, the original grid,
noise, RNG namespaces and unchanged-training declaration. Saved preprocessing
transforms the new raw-coordinate draws into model coordinates.

Fit, bandwidth-tuning, threshold and Energy Score source banks use disjoint,
independently drawn initial particles for each MC seed. Teacher/student receive
identical initial particles. Independent reference streams generate **10,000
new ground-truth observations per evaluated time**; they are not recycled
training rows, and are never used to tune density bandwidths or thresholds.
Source/reference hashes and seeds are logged. No real cross-time pairings are
created. Existing EMD scores still select checkpoints; calibration never selects
or tunes a model in this phase.

The knot grid stays at the saved **120 time points**, with the 119 noninitial
times scored by default. It is not changed to the paper's 1,200-point main
example. Both teachers/students remain frozen. The 5k/10k model-bank checks and
three MC seeds remain in place. This genuinely integrates tens of thousands of
fresh initial particles and evaluates every chosen time, so it costs more than
bootstrapping the old ten-point source. `evaluation.time_indices` can explicitly
limit a run to a preregistered subset, with a new output root.

For these narrow noisy marginals, the YAMLs set `bandwidth_scale: model_std`:
the KDE bandwidth search is scaled by the geometric mean of coordinate standard
deviations in its **model fit bank**. Independent model draws still select the
bandwidth. Test observations do not choose the scale. Other datasets retain
`bandwidth_scale: unit`, preserving their estimator choice. The shared scoring
representation is still fit only on observed training rows.

Prepare and run both knot pairs through the existing launcher:

```bash
bash local_runs/run_e3_1.sh --datasets knot \
  --output-root /data/lthung/flow2pc/experiments/e3_1/knot-analytic-v1 --dry-run

bash local_runs/run_e3_1.sh --datasets knot \
  --output-root /data/lthung/flow2pc/experiments/e3_1/knot-analytic-v1
```

There are two jobs, scheduled concurrently on separate available GPUs. The fresh
output root preserves previous empirical-mode results. Normal single-config
CLI commands also use this fresh root from their YAMLs. Outputs additionally
include `analytic_protocol.json` and `figures/knot_geometry.png`, which overlays
the true knot geometry with generated teacher/student marginal clouds. Existing
calibration, region-overlay, budget and Energy Score figures remain available.

New analytic test observations are independent of the old finite observations
used in checkpoint selection. This does not create independent training runs;
the reported MC variability still concerns fixed checkpoints. Formal
observational confidence intervals are not added in this phase.

### Commands

Metadata-only preflight (no writes, Torch, GPU query, W&B or model loading):

```bash
bash local_runs/run_e3_1.sh --dry-run
```

The default first batch is **Beijing + GOM, ALI-CFM + OTP-FM**, four paired jobs:

```bash
bash local_runs/run_e3_1.sh
```

Expand through the same eight-GPU queue:

```bash
bash local_runs/run_e3_1.sh --datasets all
```

This is 14 profiles × 2 methods = **28 jobs**: Arch/Sphere/LiDAR, six CITE/Multi
representations, ST, knot, tracking, Beijing and GOM. EB is excluded as requested
by the UQ specification. One worker per GPU; at most eight workers globally.
The four-job first batch naturally uses at most four GPUs. More jobs use the
remaining GPUs as available. The scheduler honors inherited GPU allocations and
checks memory/disk availability through the existing shared queue.

`local_runs/` is gitignored. A clone can use the tracked launcher directly:

```bash
/data/nvidia/conda_envs/uffm/bin/python scripts/run_experiments.py \
  --experiment e3_1 --datasets beijing gom --methods alicfm otpfm \
  --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 1 --max-parallel 8
```

Or evaluate one configured pair through the normal CLI:

```bash
PYTHONPATH=src /data/nvidia/conda_envs/uffm/bin/python main.py \
  --config configs/experiments/e3_1/alicfm/alicfm_beijing_hclt.yaml
```

Add `--validate-only` for a metadata-only checkpoint preflight. No `--evaluate`
flag is needed: `stage: evaluate` plus `experiment: e3_1` cannot dispatch training.
Use normal `--set evaluation.bank_sizes=[5000,10000,20000]` overrides to expand
one pair's budget check, with a new output directory for changed settings.

W&B project is `flow2pc`; scalars/images are saved locally first, and network
failure uses the existing offline/local fallback. The parallel launcher accepts
`--mode offline`. Output root can be changed via `--output-root`; the local
wrapper also accepts `GPUS`, `FLOW2PC_PYTHON`, `MIN_DISK_FREE_GB` and `WANDB_MODE`.
The disk reserve defaults to 20 GiB; freeing space resumes a waiting queue.

Missing/incomplete students are reported as blocked, never replaced with an old
teacher or retrained automatically. Finish those students, then rerun the same
command; completed matching experiments are skipped. A changed selected student,
configuration or implementation requires a new `--output-root`.

### Outputs

Default root: `/data/lthung/flow2pc/experiments/e3_1/selected-emd-v1/`.
Each `<method>/<dataset>/` contains:

- `selection.json`, `binding.json`, `resolved_config.yaml`: inputs and provenance.
- `dataset_audit.csv`, `dataset_contract.json`, `representation.json`,
  `sampling.json`: counts, eligibility, split reuse, scoring and ODE conventions.
- `by_time.csv`: every seed/budget/time/nominal-level result.
- `summary.csv`: teacher/student headline values with MC variability.
- `comparison.csv`: teacher/student values and their differences.
- `figures/`, `media.json`: calibration curves, time curves, budget sensitivity,
  Energy Score and estimated-region overlays at representative times.
- `density_diagnostics/`, `metrics.jsonl`, `tracking_status.json`, `status.json`.

The controller writes combined `summary.csv`, `comparisons.csv`,
`dataset_audit.csv`, `experiment_plan.json` and `_launcher/status.json` at its
output root. Logs/generated worker YAMLs also stay there, outside Git.

### Verification status

Regression tests are supplied under `tests/experiments/` for known Energy Score,
independent-bank Gaussian coverage, no fitting on real observations, projection
scope, discrete-score ties, EMD/provenance selection, and metadata-only launches.
Analytic knot regressions additionally cover lifted means/noise, independent RNG
streams, disjoint source banks, saved preprocessing, matching teacher/student
particles, dataset/grid guards and model-only bandwidth scaling.
They have not been executed because the repository instructions require explicit
test authorization. No test, training or E3.1 evaluation command was run.
