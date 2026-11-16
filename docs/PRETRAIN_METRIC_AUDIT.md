# Pretraining metric audit — 2026-11-13

Remediation update: F1's evaluator/selector correction and evaluation-only
recovery scripts are now implemented; see [TRANSPORT_W2_CORRECTION.md](TRANSPORT_W2_CORRECTION.md).
The findings below describe the reviewed historical state. Existing checkpoints
are retained, and numerical W2 rescoring remains for the user to launch.

This is a code and saved-metadata review, not a fresh numerical evaluation or a
claim that the entire repository is correct. No training, distillation, model
inference, optimal-transport calculation, tests, or checkpoint changes were run.
The only repository change in this review is this report. Earlier uncommitted
work remains separate.

## Scope and verified evidence

- Read the metric formulas, evaluation dispatch, dataset policies, pretraining
  evaluation calls, sweep rankings, and portable exporter for ALI-CFM/OTP-FM.
- Checked all 30 portable teacher READMEs against their recorded source runs.
  The 29 nonhistorical teachers' displayed score fields match their source
  `eval/metrics.json` exactly. Historical ALI-CFM tracking deliberately has no
  claimed evaluation score. The inventory is 27 W1 reports, two W2 reports, and
  one unscored historical teacher.
- Recomputed averages **of already saved scalar metrics only**: all 29 reports'
  applicable noninitial/held-out means agree with their saved per-time scores.
  This does not validate the underlying predictions or distances numerically.
- All 512 GOM and all 512 Beijing ALI-CFM sweep directories retain
  `checkpoints/last.pt`, `eval/teacher_predictions.pt`, and `eval/metrics.json`.
  File presence was checked; the 1,024 weight files were not loaded or hashed.
- Inspected MFM/MMFM evaluation code and presets for integration consistency.
  Their trained artifacts are not in the 30-teacher pack and were not audited.

## Findings

### F1 — P1: ALI-CFM transport selection uses the wrong benchmark distance

**Confirmed in saved results and code.** The shared historical evaluator calls
`emd(real, prediction)` without an order. `evaluation/ode.py:emd` defaults to
W1 with Euclidean ground costs. The ALI-CFM transport sweep sorts
`teacher_heldout_emd` and explicitly records W1 as its selection metric.
GOM/Beijing's common dataset policy instead declares W2 as primary.

Evidence:

- `src/flow2pc/evaluation/runner.py`: `teacher_emd` calculation and aggregation.
- `src/flow2pc/evaluation/ode.py`: W1/W2 implementations and default `order=1`.
- `scripts/run_alicfm_transport_sweep.py`: W1 completion check, summary and ranking.
- `src/flow2pc/utils/experiment_protocols.py`: transport `primary_distance: w2`.
- `configs/pretrain/alicfm/alicfm_{gom,beijing}.yaml` and
  `docs/selected_teachers.json`: settings inherited from those W1 winners.

The exported ALI-CFM scores are genuinely W1: GOM 0.13390336979068315 and
Beijing 12.581962202402977 on the saved holdouts. The exporter did not turn a W2
score into W1 or change the weights. It copied the inconsistent source report.

**Impact:** these configurations are demonstrated W1 winners, not demonstrated
W2 winners. The current YAML/pack cannot establish the best W2 configuration.
The W1 numeric values must not be renamed to W2. Training losses are unaffected.

**Correction:** define the complete transport scoring protocol, calculate W2
for the saved candidate teachers and rerank; then update selection metadata,
winning configs and pack reports. Exact-time W2 can use saved observation-time
rollouts with restored references. A denser/different time-scoring protocol may
need new rollouts, but neither requires retraining teachers. If the winning
teacher changes, only students intended to represent that new winner need new
distillation; existing students remain bound to their original teachers.

### F2 — P1: MMFM tracking's default scoring frames disagree with its figures

**Confirmed code path; no MMFM result was numerically rerun.**
`configs/pretrain/mmfm/mmfm_cell_tracking.yaml` enables evaluation without
`time_indices`. `metric_flow.evaluate` defaults to every available time; its
noninitial mean therefore includes all 114 noninitial frames for a 115-frame
tracking dataset. In the same evaluator, `save_figures` explicitly shows only
indices `[0,24,49,74,99,114]` for that dataset.

ALI-CFM/OTP-FM tracking's saved benchmark uses the six frames requested in the
dataset README. Thus the generic MMFM scalar and its six-frame illustration
can suggest agreement while representing different evaluation populations.

Evidence:

- `configs/pretrain/mmfm/mmfm_cell_tracking.yaml`: no frame selection.
- `src/flow2pc/utils/metric_config.py:prepare_mmfm`: no tracking frame default.
- `src/flow2pc/evaluation/metric_flow.py:evaluate`: all-time default and mean.
- `src/flow2pc/evaluation/metric_flow.py:save_figures`: six-frame plot selection.

**Correction:** use the dataset's six-frame policy for scoring and plotting
across teacher adapters, with regression coverage for the scored index list.
Existing MMFM tracking checkpoints can be rescored; teacher retraining is not
required for this issue. Historical ALI tracking still needs a separately
declared evaluation if a comparable six-frame score is required.

### F3 — P2: later single-cell/ST holdouts get different rollout origins

**Confirmed supported-configuration inconsistency.** The shared historical
pretraining evaluator always starts at marginal 0. The common benchmark
evaluator uses the preceding observed marginal for single-cell/ST heldouts.
Original ALI-CFM single-cell code also starts at `removed_t - 1`.

Evidence:

- `src/flow2pc/evaluation/runner.py:evaluate`: initial population is index 0.
- `src/flow2pc/evaluation/metric_flow.py:evaluate`: preceding-observed branch.
- `../ali-cfm/ali_cfm/scripts/train_ali_single_cell.py`: preceding-marginal
  CFM benchmark; `single_cell_checkpoints.py` records that source index.

**Impact boundary:** all current packed single-cell/ST teachers have held-out
index 1, so their prescribed preceding origin is also 0. This finding does
**not** prove those current scores have the wrong origin. For an accepted
later holdout, e.g. index 2, the two evaluation paths measure different tasks.
Their values should not be compared as the same benchmark.

**Correction:** make rollout origin a dataset protocol used by both evaluators
and record the actual source index per scored time. Preserve a full earliest-
time trajectory as a separately named diagnostic where needed.

### F4 — P2: portable synthetic packs omit their actual ranking evidence

**Confirmed exporter omission.** Arch/Sphere/LiDAR configurations were chosen
using `eval/selection.json` endpoint-row validation W1. Their middle-time test
W1 was deliberately excluded from selection. The exporter copies the main
`eval/metrics.json`, while `EXTRAS` does not include `eval/selection.json` and
`selection` is taken from the manifest's within-run checkpoint rule.

Consequently the portable README reports the final-checkpoint rule and test/
reconstruction scores, but omits the endpoint-validation score that justified
calling its configuration the sweep winner. This does not mean test scores
were used to select it; the saved sweep code explicitly uses validation.

Evidence: `evaluation/prepared_mfm.py`, `scripts/run_mfm_teacher_sweep.py`,
`scripts/teacher_checkpoint_pack.py:framework_bundle` and `checkpoint_readme`.

**Correction:** embed the small cross-configuration selection record in each
bundle and display its metric, value, populations and excluded test times.
Re-exporting metadata requires neither new model training nor changed weights.

### F5 — P2: MFM snapshot teachers lack the common final benchmark entry point

**Confirmed integration gap, not evidence of an incorrect native loss.** MFM
snapshot pretraining exports its validation-selected flow/geopath pair and
marks the run complete. Its config validator rejects an `evaluation` key, and
the standalone MFM evaluation config currently rejects `role: teacher`.
Thus it cannot produce the same final teacher benchmark report through the
common entry point used by the other integrations. Native validation losses or
native logs must not be substituted for a missing common W1/W2 report.

Evidence: `pretrain/models/mfm/snapshots.py:prepare/run` and
`utils/metric_config.py:prepare`'s MFM teacher-role rejection.

**Correction:** an evaluation-only adapter for the existing frozen flow/geopath
pair, respecting the same dataset metric contract. Do not change MFM losses,
its selected pair or source implementations. No trained MFM pair was loaded in
this audit.

## Important comparison differences — not proven migration bugs

1. **Native OTP-FM transport scoring differs beyond W1 versus W2.** Native GOM
   takes the minimum W2 over up to seven nearby stored times for a holdout
   (`target_idx-3` through `target_idx+3`). Native Beijing uses the nearest stored
   time. The selected runs use 25 and 20 finite-interval sampling steps,
   respectively. Shared ALI evaluation requests exact observation times from a
   diagonal-velocity ODE. The native behavior is preserved upstream behavior,
   not an implementation to silently rewrite. Correcting the distance name or
   order alone does not align these comparison protocols.
2. **Population policies differ.** Native W2 truncates both sets to their first
   `min(N,M)` rows. Shared W1 supports unequal counts with uniform masses and
   seeded subsets. A common benchmark must explicitly settle the population
   rule, as well as the metric order, units and time-scoring rule.
3. **Finite-interval OTP sampling and diagonal-velocity ODE sampling are
   different inference procedures.** A native W2 score is not automatically
   the reference score for distilling `v_func_ema(x,t,t)`. Keep native-method
   results and common instantaneous-field comparisons separately identified.
4. **Units and budgets are not universal.** Current 5D single-cell/EB and LiDAR
   reports retain standardized benchmark coordinates; 50D/100D and ST report
   raw feature coordinates. The inspected README units match their saved source
   reports. Intermediate student diagnostics commonly use fewer cells than
   pretraining evaluation, so their teacher scores need not equal pack scores.
5. **Knot's saved metric budget is capped by available observations.** Both
   packed knot source populations have ten particles, despite the nominal
   `metric_samples: 512`. Per-time metadata records actual counts. This is a
   limited evaluation sample, not evidence that teacher training used only ten
   samples: the current knot teachers draw fresh analytic training noise.
6. **Tuning holdouts are not untouched tests.** Transport and the saved
   real-data sweeps explicitly use their temporal holdout scores for
   hyperparameter selection. Synthetic Arch/Sphere uses reserved observed-row
   validation and excludes the middle test time. Preserve those distinctions.

## Regression gap and required verification

Existing tests check W1 versus W2 mathematics, naming/config provenance and
synthetic validation selection. The inspected test suite has no end-to-end
contract asserting that GOM/Beijing pretraining produces W2, that transport
sweep ranking uses it, and that the pack preserves the same primary metric.
Config fidelity tests can preserve an incorrect original setting; they are not
proof of benchmark correctness.

Before claiming correction, verify dataset → formula/units/populations/time
protocol → per-time aggregation → cross-run ranking → selected YAML/checkpoint
→ portable README/bundle. Add regressions for transport W2 and unequal counts,
six-frame tracking aggregation, later single-cell holdouts, stale metric report
rejection, and synthetic selection provenance. Execution of tests/evaluation
still requires the user's authorization under this repository's instructions.

## Recovery order

1. Resolve the common transport comparison protocol while preserving native
   reproduction reports separately. Fix evaluator/selector/report contracts.
2. Rescore and rerank the 1,024 saved transport candidates; preserve old reports
   as W1 results and write new results under a versioned protocol.
3. Update winning recipe/checkpoint references only after ranking is complete.
4. Correct tracking/later-holdout defaults and missing selection provenance;
   produce fresh evaluation reports for affected existing checkpoints.
5. Repack accurate metadata. Redistill a student only when intentionally
   switching its teacher. Do not launch a blanket teacher retraining campaign
   to repair metric/reporting errors.

## Portable teacher inventory

| Teacher | Dataset | Saved score field | Value | Source-particle metadata | Review note |
| --- | --- | --- | ---: | --- | --- |
| alicfm | arch | `teacher_heldout_emd` | 0.631537395 | 512 | F4: ranking record omitted |
| alicfm | beijing | `teacher_heldout_emd` | 12.5819622 | 744 | F1: W1 instead of transport W2 |
| alicfm | cell_tracking | — | — | — | Intentionally unscored historical teacher |
| alicfm | cite100 | `teacher_heldout_emd` | 49.268522 | 2048 | W1 label matches saved formula |
| alicfm | cite5 | `teacher_heldout_emd` | 0.965036007 | 2048 | W1 label matches saved formula |
| alicfm | cite50 | `teacher_heldout_emd` | 41.8294296 | 2048 | W1 label matches saved formula |
| alicfm | eb | `teacher_heldout_emd` | 1.03976326 | 512 | W1 label matches saved formula |
| alicfm | gom | `teacher_heldout_emd` | 0.13390337 | 111 | F1: W1 instead of transport W2 |
| alicfm | knot | `teacher_mean_emd` | 0.127513032 | 10 | W1 label matches saved formula |
| alicfm | lidar | `teacher_mean_emd` | 0.0416879888 | 512 | F4: ranking record omitted |
| alicfm | multi100 | `teacher_heldout_emd` | 53.6775234 | 2048 | W1 label matches saved formula |
| alicfm | multi5 | `teacher_heldout_emd` | 1.04325019 | 2048 | W1 label matches saved formula |
| alicfm | multi50 | `teacher_heldout_emd` | 46.7420293 | 2048 | W1 label matches saved formula |
| alicfm | sphere | `teacher_heldout_emd` | 0.659762887 | 512 | F4: ranking record omitted |
| alicfm | st | `teacher_heldout_emd` | 120.987388 | 755 | W1 label matches saved formula |
| otpfm | arch | `teacher_heldout_emd` | 0.627394492 | 512 | F4: ranking record omitted |
| otpfm | beijing | `teacher_heldout_w2` | 18.7065334 | native cap 1000 | Native W2; distinct sampling/time rule |
| otpfm | cell_tracking | `teacher_mean_emd` | 34.8255858 | 4154 | W1 label matches saved formula |
| otpfm | cite100 | `teacher_heldout_emd` | 48.9235526 | 2048 | W1 label matches saved formula |
| otpfm | cite5 | `teacher_heldout_emd` | 0.867000244 | 2048 | W1 label matches saved formula |
| otpfm | cite50 | `teacher_heldout_emd` | 43.1210421 | 2048 | W1 label matches saved formula |
| otpfm | eb | `teacher_heldout_emd` | 0.424751178 | 512 | W1 label matches saved formula |
| otpfm | gom | `teacher_heldout_w2` | 0.107283778 | native cap 111 | Native W2; distinct sampling/time rule |
| otpfm | knot | `teacher_mean_emd` | 0.0152081271 | 10 | W1 label matches saved formula |
| otpfm | lidar | `teacher_mean_emd` | 0.0468273015 | 512 | F4: ranking record omitted |
| otpfm | multi100 | `teacher_heldout_emd` | 52.9162535 | 2048 | W1 label matches saved formula |
| otpfm | multi5 | `teacher_heldout_emd` | 0.912809079 | 2048 | W1 label matches saved formula |
| otpfm | multi50 | `teacher_heldout_emd` | 46.2373293 | 2048 | W1 label matches saved formula |
| otpfm | sphere | `teacher_heldout_emd` | 0.760534062 | 512 | F4: ranking record omitted |
| otpfm | st | `teacher_heldout_emd` | 117.998783 | 755 | W1 label matches saved formula |

Native caps are configured limits, not assertions of actual output counts.
This table inventories saved reports; it is not a new performance evaluation.
