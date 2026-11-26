# E3.2: same-time conditional uncertainty

The first presets evaluate frozen **ALI-CFM and OTP-FM on ST**: observe spatial
coordinate 0 and predict coordinate 1, then reverse. Both use all 870 held-out U3
observations (index 1, model time 1/3, physical coordinate 3), the same masks,
training-derived scaling and sampling budgets. This is not future forecasting
or trajectory conditioning. No teacher or student is retrained.

**Beijing is ineligible with the current checkpoints:** both have one state
variable, Dingling PM2.5. The proposal's hypothetical six-feature Beijing task
does not describe these saved models. E3.2 rejects a 1D state before launching
workers rather than inventing additional variables or pairings.

This implementation has not been numerically validated by the coding agent.
No training, distillation, evaluation or numerical tests were executed.

## Selection and initial law

Configs live in `configs/experiments/e3_2/{alicfm,otpfm}/`. They use the existing
ST teachers and select completed, teacher-matched students by saved
`validation_selection_score` (velocity MSE for ST). The controller pins its
winner before queueing and records the candidate count and selection finality.
Set `selection.pinned_checkpoint` explicitly to reuse a previous experiment's
exact student. An incomplete sweep produces a provisional winner.

Both ODEs start from the same saved empirical U2 source (`source_split: all`).
Each distinct requested source row is integrated once per model. There is no
added source noise, new initial distribution, direct PC sampling, or use of the
internal PC density as the ODE distribution's density.

## Conditional estimator

The proposal permits sample-based conditional inference. This implementation
fits the same **Gaussian joint KDE** to each model's generated state particles
and conditions that mixture analytically. In training-standardized coordinates,

`q_h(z) = sum_j m_j Normal(z; z_j, h^2 I)`.

For observed values `o`, component weights are proportional to
`m_j exp(-||o-z_j,O||^2 / (2 h^2))`; each hidden component is
`Normal(z_H; z_j,H, h^2 I)`.

This is an **evaluation-only smoothed distribution**, not an exact conditional
of a finite atomic ODE law. Hidden-coordinate kernel noise is part of this
density estimator; it changes neither model nor its training. Density scores,
HDR threshold draws and proper-score draws all concern the same normalized
conditional mixture. Do not call these exact flow likelihoods.

Center/scale use observed-time training rows only. Full hidden coordinates are
scored without PCA. Duplicate fit particles retain their mass but are collapsed
for computation and support diagnostics. Larger fit banks cannot create more
than the original 755 initial cells.

Bandwidths **0.05/0.1/0.2** are fixed in standardized units; **0.1** is the
predeclared primary, not selected by test coverage. All bandwidths are reported
at fit-bank sizes **2,000/5,000** and conditional draw sizes **256/512**, for
three MC seeds. Bank-size comparisons hold bandwidth fixed.

## Metrics

The requested headline metrics are:

- **Conditional HDR-ECE:** construct each condition's density threshold at
  masses `{0.50, 0.70, 0.90, 0.95}` using independent conditional draws. Average
  real hidden-value membership across conditions first, then average absolute
  deviations from nominal levels. Averaging `|indicator-alpha|` per observation
  would be wrong. This uses the same four-level convention as E3.1.
- **Conditional Energy Score:** for hidden truth `y` and independent conditional
  draws `A,B`, use `mean(0.5*(||A-y||+||B-y||) - 0.5*||A-B||)`. This is an
  unbiased paired-bank MC estimator, in full hidden standardized coordinates.
  It measures distribution quality, not calibration alone.

**SD-ECE is omitted at the user's request.**

Additional outputs include 90% HDR coverage, scalar central-interval coverage
and its four-level error, conditional CRPS, interval width, uniformly averaged
interval score, and **estimated conditional KDE NLL per hidden dimension**.
For ST's single hidden coordinate, Energy Score and CRPS coincide and are not
independent evidence. Central intervals and HDRs can differ for multimodal laws.

Threshold, score-A and score-B streams are independent conditional on the fitted
mixture. Scoring score-A against the threshold gives an independent model-mass
check. Common random numbers across roles do not pair real observations across
time. Metrics macro-average masks/times within each MC seed; reported standard
deviations concern MC resampling, not observational confidence intervals.

HDR calibration and multivariate proper scoring are discussed in
[Chung et al., ICML 2024](https://proceedings.mlr.press/v235/chung24a.html).
This code evaluates calibration; it does not implement post-hoc recalibration.

## Reliability

- Report posterior effective support `1/sum_j w_j^2` after collapsing duplicate
  fit particles. The minimum, mean and fraction below **20** flag sparse
  conditional support; that threshold is a diagnostic, not proof of reliability.
- Keep all reserved observations, including poorly supported conditions.
- Report calibration by observed-value quartiles with boundaries fit on
  training data. Aggregate coverage does not establish pointwise calibration.
- Inspect bandwidth, fit-bank and conditional-draw sensitivity before making
  claims. Increasing resampling alone cannot cure finite source support.
- Real-data results remain **retrospective**: teacher selection and saved
  preprocessing may have used temporal holdouts. No untouched-test or
  independent-training-run claim is made.

## Manual launch

The ignored local wrapper uses the shared GPU queue with GPUs 0–7 available
and two parallel method jobs. Existing student runs are not stopped/restarted.

```bash
bash local_runs/run_e3_2.sh --dry-run
bash local_runs/run_e3_2.sh
tail -f /data/lthung/flow2pc/experiments/e3_2/st-mse-v1/_launcher/controller.log
```

The preview reads metadata only: no Torch, GPU query, W&B, output writes or
workers. Regular execution uses W&B project **flow2pc** with the existing
offline/local fallback. ODE sampling uses GPUs; conditional KDE scoring uses CPU.
This is a sensitivity experiment, not a promise of a seconds-long run.

For a single method, the existing CLI also works:

```bash
PYTHONPATH=src /data/nvidia/conda_envs/uffm/bin/python main.py \
  --config configs/experiments/e3_2/alicfm/alicfm_st_hclt.yaml
```

Results go under `/data/lthung/flow2pc/experiments/e3_2/st-mse-v1/METHOD/st/`:

```text
selection.json                 # MSE winner, candidate count, finality
binding.json                   # config/checkpoint/code hashes
dataset_audit.csv              # references, splits, retrospective qualification
representation.json            # training-derived scoring coordinates
sampling.json                  # empirical source and conditional estimator
by_time_mask.csv               # all bank/draw/bandwidth/seed/time/mask results
by_condition.csv               # individual outcomes at primary setting
by_observed_quartile.csv       # stratified calibration at primary setting
summary.csv                    # primary bandwidth, largest bank/draw budgets
comparison.csv                 # teacher/student metrics and differences
estimator_diagnostics.json
results.json
figures/calibration.png
figures/sensitivity_and_energy.png
figures/missing_coordinates.png # central intervals against hidden observations
```

Use a new output root when scientific settings/checkpoints/code change.
Incomplete numerical work is recomputed on restart; matching complete artifacts
are reused. The wrapper's `E3_2_ROOT` or `--output-root` sets a different root.

## Additional pilots: CITE-5D and Multi-5D

These two datasets provide substantially more initial support than ST while
remaining low-dimensional enough for a conditional-KDE diagnostic. Their saved
temporal holdout is index 1, with 6,999 CITE cells and 8,544 Multi cells available.
The earliest marginals have 7,476 and 7,264 cells respectively. The experiments
use the actual saved five-dimensional embeddings, not individual genes or
physical trajectories. They do not establish gene-level imputation quality.

Four new presets use the frozen ALI-CFM/OTP-FM pairs, with two predeclared
complementary masks:

- Observe coordinates `[0, 1]` (40%), predict `[2, 3, 4]` jointly (3D).
- Observe coordinates `[2, 3, 4]` (60%), predict `[0, 1]` jointly (2D).

This avoids adding another PCA projection and avoids 4D hidden-state HDRs.
All hidden coordinates enter joint density ranking and Energy Score. The two
mask dimensions are recorded separately; aggregate scores macro-average masks
consistently across roles. Five dimensions cannot express exact 10% or 75%
observed masks.

The first pass scores a deterministic **512-cell subset** of the temporal test
population, identical across methods/roles, using the existing seed
`2026 + time_index`. This is a declared computational pilot budget, not a
subset selected by model performance. Source banks, conditional budgets,
bandwidth grid/primary and three MC seeds remain the same as ST. Larger source
support does not guarantee adequate conditional effective support, especially
when conditioning on three coordinates: inspect the existing ESS diagnostics.

ALI-CFM searches the retained `selected-students/<dataset>` cohorts, where a
single eligible current-teacher run may be the only candidate. OTP-FM searches
the existing `student-sweep-100/<dataset>` cohorts. Both select by the saved
velocity-validation score and record counts/finality. This does not create a
new student sweep or claim old transferred recipes were retuned to this teacher.

```bash
bash local_runs/run_e3_2_single_cell.sh --dry-run
bash local_runs/run_e3_2_single_cell.sh
tail -f /data/lthung/flow2pc/experiments/e3_2/single-cell-mse-v1/_launcher/controller.log
```

This queues four jobs concurrently over GPUs 0–7. It uses the existing E3.2
runner and shared launcher. No changes to the ST presets or numerical estimator
are needed, and its saved outputs remain separate. The equivalent command is:

```bash
bash local_runs/run_e3_2.sh --datasets cite5 multi5 --max-parallel 4 \
  --output-root /data/lthung/flow2pc/experiments/e3_2/single-cell-mse-v1
```

Outputs are under that root's `METHOD/DATASET/` folders, with W&B project
`flow2pc`, HDR-ECE, Energy Score, per-mask and per-observed-quartile tables,
calibration images and sensitivity images. The ST-only scalar missing-coordinate
band plot is not produced for multivariate hidden masks. KDE scoring uses CPU
and costs more with these larger fit supports. The experiment remains
retrospective, and its accuracy and convergence must be assessed from results.
