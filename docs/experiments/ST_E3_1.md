# ST marginal uncertainty calibration

This is an evaluation-only experiment for the existing frozen ALI-CFM and
OTP-FM ST teachers and one MSE-selected student per teacher. No teacher or
student is trained by this experiment. The student sweep can continue separately.

## Data and selection

The saved ST representation contains **two spatial coordinates**, not the
high-dimensional gene-expression vector. Both teachers use the saved U2/U3/U4/U5
populations at physical coordinates `[2, 3, 4, 5]` and model times
`[0, 1/3, 2/3, 1]`. U3 is held out, index 1, with **870**
reference observations. The empirical U2 source has **755** observations.

The experiment evaluates U3 only. It uses the complete reserved U3 population,
full 2D regions without PCA, and the saved preprocessing/time convention.
It does not add noise, invent trajectory pairings, or replace the initial law.

ST YAMLs search `distill/student-sweep-100/METHOD/st/`. They select the lowest
saved `status.json:best_selection_score`, using validation student/teacher
velocity MSE on the existing reserved observed-time queries. The selector
requires complete runs, matching frozen teacher hashes and selected checkpoint
steps, finite nonnegative scores, and comparable validation data/seed/sampler
settings. Population EMD never ranks these ST candidates.

The controller pins the chosen checkpoint before queueing it. Later sweep
completions cannot replace that checkpoint in an active job. Candidate counts,
planned counts and `selection_final` are saved. While the 100-configuration
sweep is unfinished, this is a **provisional best-completed** choice. It is not a
claim that the complete search has finished.

The real populations were already involved in earlier teacher selection and
native preprocessing. This remains a retrospective diagnostic, not an untouched
final-test claim. Student selection itself remains the existing observed-row
velocity-validation policy.

## Two separate result categories

Calibration against the real U3 population retains coverage at 50/70/90/95%,
HDR-ECE, and Energy Score. KDE fit, bandwidth tuning and density-rank thresholds
use separate model banks. Real reference observations never fit or tune KDE.

Uncertainty fidelity uses the already generated `energy_a` model bank,
independent of the fit/tune/threshold draws, and reports:

- Teacher-defined region mass on independent teacher and student draws.
- Student-defined region mass on independent student and teacher draws.
- Cross-model minus own-model region-mass gaps: zero is the fidelity target.
- Paired ODE RMSE and mean displacement on shared initial particles.
- Relative covariance error, interval-width ratios and relative width error.

Location/covariance/width statistics use full-state coordinates standardized on
observed training rows. Region masses use the declared HDR coordinates. Paired
particles mean identical initial states across the two models, not paired real
trajectories. Undefined spread ratios for collapsed teachers are recorded as
missing, without inserting an artificial scale floor. These are complementary
diagnostics, not an equivalence proof or a new training objective.

The fidelity figure compares each region's own-model and cross-model mass.
Comparing teacher/student calibration scores alone does not establish fidelity.
Good fidelity can coexist with a poorly calibrated teacher.

## Estimator limitations and diagnostics

The empirical source law is discrete. A 10,000-draw model bank resamples the
same 755 source observations, rather than generating 10,000 distinct initial
states. `sampling.json` records the source support, and each density diagnostic
records the unique fitted model rows, bandwidth candidates, chosen index and
whether the choice lies on the grid boundary. `estimator_diagnostics.json`
summarizes boundary frequency and the initial-law qualification.

A boundary optimum or bank-budget stability does not establish that smoothing
has converged. The experiment does not automatically change bandwidth rules or
the physical initial law to obtain more favorable coverage. KDE regions remain
estimated density-ranking regions, not exact flow-density HDRs.

The 5,000/10,000 bank sizes and three MC seeds are unchanged. Reported error bars
are MC standard deviations for frozen models, not biological-observation
confidence intervals. Reference/group bootstrap CIs and ODE tolerance convergence
are not added in this phase.

## User commands

From the repository root:

```bash
# Metadata only; no Torch, GPU queries, writes, W&B, training or evaluation.
bash local_runs/run_e3_1_st.sh --dry-run

# Two evaluation jobs through the existing shared GPU queue.
bash local_runs/run_e3_1_st.sh

# Live combined controller log after launching.
tail -f /data/lthung/flow2pc/experiments/e3_1/st-mse-v1/_launcher/controller.log
```

The script logs to W&B project `flow2pc`, allows at most two workers, and uses
the existing GPU/disk admission guards. It does not stop or reconfigure the
student sweep. `GPUS`, `MIN_FREE_GB`, `CPU_THREADS_PER_RUN`, `WANDB_MODE`,
`FLOW2PC_PYTHON`, and `E3_ST_ROOT` configure the local launch recipe.

The tracked Python entry point is:

```bash
/data/nvidia/conda_envs/uffm/bin/python scripts/run_experiments.py \
  --experiment e3_1 --datasets st --methods alicfm otpfm \
  --output-root /data/lthung/flow2pc/experiments/e3_1/st-mse-v1 \
  --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 1 --max-parallel 2
```

To evaluate a later winner or changed experiment code, choose a fresh root,
for example `E3_ST_ROOT=/data/lthung/flow2pc/experiments/e3_1/st-mse-v2`.
The existing binding guards prevent silently replacing a completed result.

## Outputs and verification

`/data/lthung/flow2pc/experiments/e3_1/st-mse-v1/` contains combined calibration,
comparison and fidelity summaries. Each `METHOD/st/` contains:

```text
selection.json                   # Metric, checkpoint, hashes, candidate counts
dataset_audit.csv                 # Actual reference split/time/counts
sampling.json                    # Initial law, support, NFE and budgets
summary.csv / comparison.csv      # Calibration against real observations
fidelity_by_time.csv
fidelity_summary.csv              # Direct teacher/student uncertainty diagnostics
estimator_diagnostics.json
density_diagnostics/*.json
figures/{calibration,fidelity,regions_time_1,...}.png
```

Python/Bash syntax and metadata-only checkpoint previews were checked. Numerical
regression cases were added for MSE selection, incompatible validation queries,
launch pinning, and fidelity under identity, translation, scaling and collapse.
Tests and model evaluations were not executed during preparation.
