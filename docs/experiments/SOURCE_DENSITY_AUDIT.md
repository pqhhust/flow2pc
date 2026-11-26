# ST source-density and tail audit — 2026-11-25

## Scope and evidence

This audit reads the eight completed ST source-NLL pilot histories under
`/data/lthung/flow2pc/distill/source-nll-pilot-v1/` and four completed diagnostic
reports under `/data/lthung/flow2pc/diagnostics/source-nll-pilot-v1/`.
No model was run during the audit. Student/teacher implementations, training,
sampling, loss weights and checkpoint-selection rules were not changed.

Machine-readable snapshots are saved locally, outside Git:

- `local_runs/source_density_audit_20261125/checkpoint_audit.csv`
- `local_runs/source_density_audit_20261125/tail_aggregate_audit.csv`

These snapshots summarize saved metrics; they do not establish new inference
results. The newly prepared optional tail recorder below has not been run.

## Checkpoint selection is implemented as configured

All eight selected steps and scores in `status.json` exactly match the minimum
logged validation velocity MSE. Five select step 20,000; OTP-FM at weights
0.01/0.1/1 selects steps 18,000/17,000/17,000. The final evaluator explicitly
loads `best.pt`; the runner does not silently evaluate the last student.

For these ST recipes, validation is `global_velocity_mse` on fixed teacher
interpolant/conditional queries built from reserved observed-time validation
cells. It is neither velocity MSE on teacher ODE trajectories nor velocity MSE
on direct PC samples. The held-out ST population is not used in this criterion.
Source NLL uses 38 reserved initial validation cells. Its training term uses
initial training cells and an independent RNG stream. Existing saved
`native_all` preprocessing is retained; this audit does not establish a new
train-only preprocessing protocol.

| Method | NLL weight | Selected step | Selected validation velocity MSE | Selected source val NLL/D | Final source val NLL/D | Logged updates clipped |
|---|---:|---:|---:|---:|---:|---:|
| ALI-CFM | 0 | 20,000 | 0.00003241 | 0.69091 | 0.69091 | 0% |
| ALI-CFM | 0.01 | 20,000 | 0.00036559 | -0.36157 | -0.36157 | 0% |
| ALI-CFM | 0.1 | 20,000 | 0.00081210 | -0.43006 | -0.43006 | 42.3% |
| ALI-CFM | 1 | 20,000 | 0.00130961 | -0.41550 | -0.41550 | 99.0% |
| OTP-FM | 0 | 20,000 | 0.00077707 | 0.75397 | 0.75397 | 0% |
| OTP-FM | 0.01 | 18,000 | 0.00120885 | -0.31062 | -0.32150 | 0% |
| OTP-FM | 0.1 | 17,000 | 0.00779581 | -0.45841 | -0.47588 | 47.8% |
| OTP-FM | 1 | 17,000 | 0.01451031 | -0.48634 | -0.48285 | 99.0% |

Clipping percentages use 201 logged updates per run, not all 20,000 updates.
The threshold is the existing combined-gradient norm limit of 1.0. Largest
logged norms reach 20.56 for ALI-CFM weight 1 and 17.12 for OTP-FM weight 1.
This shows frequent clipping, not divergent gradients or proof of opposing
loss gradients; separate gradient directions were not recorded.

There is an objective-selection tradeoff, not a demonstrated best/last bug.
For OTP-FM weight 0.1, moving from selected step 17,000 to final step 20,000:

- validation velocity MSE worsens from 0.00779581 to 0.00803381 (about 3.1%);
- source validation NLL/D improves from -0.45841 to -0.47588;
- final direct-PC quality is unknown, because population evaluation used only
  the selected checkpoint.

Only `best.pt` and `last.pt` are retained. The lowest-NLL intermediate
checkpoint generally cannot be reconstructed from a log. Do not report its
unmeasured population quality or replace selected configs based on test W1.

## What the tail aggregates establish

At weight 0.1, with 512 samples per state population:

| Method | Population | Physical time | Population velocity MSE | Largest point's share of MSE |
|---|---|---:|---:|---:|
| ALI-CFM | Direct PC | 3 | 0.00090700 | 7.99% |
| ALI-CFM | Direct PC | 4 | 0.00232995 | 66.71% |
| ALI-CFM | Direct PC | 5 | 0.02132094 | 48.21% |
| OTP-FM | Direct PC | 3 | 0.04505978 | 78.02% |
| OTP-FM | Direct PC | 4 | 0.20389506 | 59.60% |
| OTP-FM | Direct PC | 5 | 0.91402778 | 30.99% |
| ALI-CFM | Student ODE from PC initial law | 5 | 29.14402 | 99.990% |

The share follows exactly from saved aggregates:
`largest_point_RMSE**2 / (state_count * population_velocity_MSE)`.
No rows are removed or winsorized. Velocity units are saved model coordinates
per model-time unit. Relative errors use teacher RMS speed on the same states.

For the last ALI-CFM row, median and p99 pointwise RMSE are approximately
0.04077 and 0.12873, while the maximum is 122.15. The median is close to the
direct-PC median (0.03952), despite a much larger aggregate MSE. This is an
extreme-tail finding, not evidence that every sample has a large error.

The reports do **not** retain per-point coordinates, log densities or separate
teacher/student velocity vectors. Consequently, these aggregates cannot tell
which vector is extreme, how likely the offending state is, or whether it is
near training-query states. The existing population evaluator retained samples
from a different seed/protocol; those cannot identify these exact particles.

## Training-state coverage: code evidence versus hypothesis

`distill/sampling.py` queries the ALI-CFM interpolant using the saved coupling
(`border_ot` for this teacher), training endpoints and uniformly sampled global
model times. `distill/otpfm_sampling.py` constructs native conditional tuples
from observed training populations, recovers the saved coupling/alpha, and
supervises with the frozen EMA diagonal velocity. It does not supervise with
the derivative of an individual conditional path.

Neither sampler queries the current PC distribution. NLL is applied only at
the initial time. Therefore the training objective does not directly measure
velocity error under the PC's current distribution at later times. Poor
matching on PC-generated states is consistent with a coverage/approximation
problem. It does not prove an individual offending point was outside the
training-query distribution: exact training batches were not retained.

In the dense Gaussian HCLT path used here, velocity uses posterior-weighted
`mean_dt + log_scale_dt * (x - mean)`. Density and sampling use the corresponding
Gaussian means/scales; no separate clipping or time conversion was found in
this path. Fourier means and log scales remain trainable at the initial time,
and scales are exponentiated without a hard clamp. Broad or low-mass components
and large parameter derivatives are plausible tail mechanisms, not confirmed
causes. No PC implementation change is justified by the aggregates alone.

The direct-PC/own-initial-ODE W1 discrepancies remain comparable to independent
direct/direct sample discrepancies. At one seed and 512 samples, this is
supporting evidence for density–velocity consistency, not an equivalence test
or a guarantee of tail accuracy. Distributional W1 similarity does not bound
the squared velocity error at rare points.

## Prepared next diagnostic; not executed

The existing checker and parallel launcher now accept `--tail-count N`.
Zero is the default and does not query PC density. Positive values save the
largest-error scored rows per population/time in `tail_samples.json`, including:

- index in the scored population and error rank;
- state in saved model coordinates and original raw coordinates;
- separate student and teacher velocity vectors in model-time units;
- per-point MSE, fraction of total population MSE, and PC log density/NLL per
  coordinate at that exact state.

The same sampling calls, populations, error reductions and model weights are
used. Tail records do not change training, discard observations, trim metrics,
estimate training-query coverage, or fit a new distribution. Their settings
enter the diagnostic identity and use new output directories.

Metadata preview (does not load models or query GPUs):

```bash
bash scripts/run_pc_consistency_pilot_8gpu.sh \
  --weights 0.1 --tail-count 16 \
  --output-root /data/lthung/flow2pc/diagnostics/source-nll-tail-audit-v1 \
  --dry-run
```

User-run evaluation command (two frozen models, concurrent GPU workers):

```bash
bash scripts/run_pc_consistency_pilot_8gpu.sh \
  --weights 0.1 --tail-count 16 \
  --output-root /data/lthung/flow2pc/diagnostics/source-nll-tail-audit-v1
```

W&B remains project `flow2pc`, with the existing local/offline fallback.
This repeats the diagnostic to retain previously missing evidence; it does
not retrain either model. Inspect the vectors/densities before changing a
sampler or declaring an optimization-only remedy.

After this audit, the user approved a separate NLL/LR refinement sweep. Its
[42-run recipe](SOURCE_DENSITY_ANCHOR.md#st-nll-and-learning-rate-refinement)
is now prepared but has not been launched by the assistant. It keeps teacher
queries and optional initial NLL only. The subsequently proposed PC-state
velocity training term was removed at the user's request; it is not part of
the refinement.
