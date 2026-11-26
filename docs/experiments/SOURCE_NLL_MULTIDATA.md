# Full-dimensional CITE and knot distillation pilot

This phase prepares new students for later E3 probability/conditional queries.
It uses frozen ALI-CFM and OTP-FM teachers, existing selected student recipes,
and the already implemented objective
`teacher-query velocity MSE + lambda * initial train NLL / state_dimension`.
The PC model files, teacher models, and query-sampling algorithms are unchanged.
No training, inference, model evaluation, or tests were launched while preparing
this phase. Numerical consistency and distillation quality remain to be checked
on the outputs of the supplied commands.

## Grid and data contracts

Both methods cover CITE-5D, CITE-50D, CITE-100D and knot. Five source-NLL weights
`0, 0.003, 0.01, 0.03, 0.1` give **40 runs**. The zero-weight control uses the
same source initialization and data as the other weights. There is no PC-sampled
velocity objective, bootstrapping, new loss, source Gaussian assumption, or
post-hoc recalibration. NLL is per coordinate, but its numerical scale still
depends on each dataset's saved representation.

The ordinary YAMLs are
`configs/distill/<method>/source_anchor/<method>_<dataset>_hclt.yaml`.
They copy each current selected recipe's teacher path, architecture, time
features, training steps, LR, batch size, sampler and validation scope. In
particular, OTP-FM CITE-50D/100D retain the selected sweep checkpoints named in
their current distillation presets; the launcher does not substitute different
weights from the pretrained-teacher export. All CITE dimensions are retained:
no new PCA, projection or normalization is fitted.

| Method | Dataset | Latents | Time features | LR | Steps |
|---|---|---:|---:|---:|---:|
| ALI-CFM | CITE-5D | 256 | 64 | 0.001 | 50,000 |
| ALI-CFM | CITE-50D | 256 | 64 | 0.001 | 50,000 |
| ALI-CFM | CITE-100D | 256 | 32 | 0.001 | 50,000 |
| ALI-CFM | knot | 128 | 128 | 0.003 | 20,000 |
| OTP-FM | CITE-5D | 256 | 64 | 0.003 | 20,000 |
| OTP-FM | CITE-50D/100D | 128 | 32 | 0.003 | 20,000 |
| OTP-FM | knot | 256 | 128 | 0.003 | 20,000 |

`best.pt` remains selected by validation velocity MSE. Source validation NLL is
reported at that checkpoint, with no automatic lambda selection from test W1
or calibration. Intermediate population ODE diagnostics are disabled. Final
teacher/student comparisons and images remain enabled. Each selected recipe's
entire final `evaluation` mapping is preserved, including CITE's 2,048-particle
caps, source/reference splits, solver, tolerance and seed. Every distance uses
the complete state dimension. Plots may display selected coordinates without
changing the metric dimension.

### Dataset benchmark metrics remain distinct

| Dataset | Native benchmark quantity | Coordinates | ODE benchmark origin / scored times |
|---|---|---|---|
| CITE-5D | Uniform-mass W1 (EMD), all 5 features | Benchmark-whitened saved PCA features | Preceding observed population to the designated held-out time |
| CITE-50D/100D | Uniform-mass W1 (EMD), all 50/100 features | Original saved PCA units | Preceding observed population to the designated held-out time |
| Knot | W1 and geometry diagnostics, both coordinates | Original lifted-curve coordinates | Earliest population; configured observed times, no temporal holdout |

The final shared benchmark is saved under `eval/population/`; its protocol
metadata identify coordinates, source indices and prediction modes. The legacy
historical report is separate and must not override this dataset policy. CITE's
configured finite caps are diagnostics rather than the original all-cell table
replication; no claim of full-population parity is made. Knot's ten-point saved
benchmark remains a finite-reference diagnostic, not a calibration result.

Source NLL, velocity MSE and query-consistency checks do not replace these
benchmark quantities. The standalone diagnostics below deliberately report
raw-coordinate W1 with their own sample/time budgets, so CITE-5D raw diagnostic
W1 must not be presented as its whitened benchmark W1. Neither diagnostic W1
nor test calibration chooses `best.pt`; it still uses the dataset-specific
velocity-validation protocol. GOM/Beijing are outside this batch and retain
W2 under their OTP-FM dataset policy. E3's calibration metrics are additional
query-task measurements, not replacements for native W1/W2.

See [the dataset-level training/evaluation reference](../DATASET_TRAINING_EVALUATION.md)
for the source protocols, time handling, preprocessing and known parity limits.

### Larger knot student populations

`distillation.student_data` explicitly declares `independent_knot`, 1,024 points
per time and seed 20261126. The existing 120-time curve, middle-branch `+1`,
noise standard deviation 0.01, observed times, and saved teacher preprocessing
are retained. Each time has 820 training, 102 validation and 102 test points.
New populations are independent noisy marginals, not paired trajectories.

The teacher's original ten-point dataset and checkpoint metadata are preserved.
New pools feed student initialization, the existing teacher-query sampler,
velocity validation and source NLL. The OTP sampler may already generate fresh
analytic training tuples when its checkpoint declares that protocol; that
behavior is retained. No teacher is retrained and no saved preprocessing is
refitted. The original `dataset` checkpoint binding remains the teacher and
benchmark identity; `student_training_data` separately records the full new
dataset contract, seed, counts and implementation hash. Resume requires both
contracts to match. `student_training_data.json` exposes this information.

The legacy final knot benchmark still uses the teacher-bound ten-point
populations. Its direct sample count is limited by that empirical source, so
it is not a large-sample density check. The explicit diagnostics below use
2,048 PC draws. Later knot E3 coverage should use the already-supported fresh
analytic reference, rather than treating ten saved points as adequate ground
truth. Newly generated student test rows are not used to choose a checkpoint.

## Commands

```bash
# Metadata-only preview: no Torch, GPU queries, outputs, models or W&B jobs.
bash scripts/run_source_nll_multidata_7gpu.sh --dry-run

# Train the 40 students and produce standard final evaluations/images.
bash scripts/run_source_nll_multidata_7gpu.sh
```

GPU pool: **0,1,2,4,5,6,7**, one worker per GPU, at most seven concurrent jobs.
GPU 3 is rejected through both `GPUS` and the `--gpus` option. The existing shared
resource-aware queue handles memory, disk space, locks and resume. W&B project
is `flow2pc`, with method/dataset/lambda-specific names and the existing local
fallback. `WANDB_MODE=offline` is accepted. Rerunning the same command skips
completed runs and resumes compatible checkpoints. No existing student or
experiment output is overwritten.

Student root:
`/data/lthung/flow2pc/distill/source-nll-multidata-v1/`.
Override it explicitly with `SOURCE_NLL_OUTPUT_ROOT` for a different batch.
The root contains `summary.csv`, `_launcher/status.json`, per-worker logs and
`<method>/<dataset>/<run>/` outputs.
The training summary includes the benchmark metric, coordinate space, rollout
origin and report path once evaluation has produced them. Diagnostic summaries
explicitly label their raw W1 columns as additional consistency measurements.

After training, run **direct-query implementation and tail checks**:

```bash
bash scripts/run_source_nll_multidata_7gpu.sh --phase diagnose --dry-run
bash scripts/run_source_nll_multidata_7gpu.sh --phase diagnose
```

This evaluates every declared candidate, without choosing a winner. It samples
2,048 PC states at four recorded times, uses a 512-point W1 cap, and scores all
states for velocity error and the sixteen worst-error states. Full raw/model
sample banks are retained. For knot, source-NLL and reserved-cell velocity
checks use the enlarged student validation pools; benchmark W1 references
remain the saved teacher populations and are explicitly not fresh analytic
calibration. For CITE, held-out time 1 and all five/fifty/hundred features are
included. There is no student or teacher ODE in this phase.

The extra numerical query audit checks:

- Full joint direct-query log density versus the original student `log_prob`,
  including the affine Jacobian in the actual raw coordinates.
- Empty marginal and unbounded rectangle normalization.
- Conditional density versus the original joint density divided by evidence
  density, for observed coordinate 0, the last coordinate, and all-but-last.
- Independent conditional samples versus exact coordinate-event probabilities,
  integrating other variables out. A six-binomial-standard-error tolerance is
  a conservative implementation check, not a calibration confidence interval.

The audits retain the full state dimension. They use no teacher KDE, no 2D PCA,
and no real test targets to construct conditioning evidence. Failures remain
failed/blocked with `query_consistency.json` for inspection. A numerical query
pass does not establish student-teacher fidelity, tail reliability, calibration
on real data or correctness of every possible query.

The optional **density–velocity transport check** is separate:

```bash
bash scripts/run_source_nll_multidata_7gpu.sh --phase consistency
```

It compares direct PC samples with the student ODE from its own PC initial law,
plus separate transports from empirical training source points. It uses 128
particles, Dopri5 tolerance 1e-5 and a 10,000-NFE cap; failures or budget limits
are reported, not interpreted automatically as a mathematical inconsistency.
It can be substantially slower than direct query checks, especially for
high-dimensional students with poor tails.

Diagnostic roots are
`/data/lthung/flow2pc/diagnostics/source-nll-multidata-v1/direct-query-v1/` and
`.../density-velocity-v1/`. Override their parent with
`SOURCE_NLL_DIAGNOSTIC_ROOT`. Partial results are preserved. For a fresh retry
after a code/protocol correction, choose another diagnostic root while keeping
the student root. Both phases require the training plan and complete students.

## Next E3 phase

The agreed high-dimensional experiment is feature-probability and
missing-feature queries from the **full 50D/100D PC models**. It does not replace
them with a 2D PCA model or pretend that a 100D teacher KDE is a reliable joint
HDR reference. This phase prepares distillation and numerical query checks;
it does not yet launch or report those high-dimensional E3 calibration runs.
The low-dimensional E3.1 full-state representation now permits explicit 5D
profiles, with teacher KDE sensitivity still required. High-dimensional query
calibration needs its own explicit query masks and teacher-reference protocol.
