# Small ST source-NLL query pilot

This evaluates the existing E3.1 marginal and E3.2 same-time conditional
queries on two frozen source-NLL students. It neither trains models nor changes
PC/student/teacher implementations. The source-NLL confirmation diagnostic
measured W1 and velocity fidelity, not query calibration; this pilot checks
coverage, HDR-ECE and Energy Score using the actual PC law.

The fixed candidates are ALI-CFM ST weight 0.003, LR 0.003, training seed 0 and
OTP-FM ST weight 0.1, LR 0.002, training seed 0, from the declared confirmation
recipes. Each uses its existing validation-velocity-MSE-selected `best.pt`.
These are explicitly pinned exploratory candidates, not newly declared winners
of a calibration sweep. The ordinary E3 presets and earlier results stay intact.

All four YAMLs live in
`configs/experiments/<e3_1|e3_2>/<method>/source_nll_pilot/`.
The thin launcher calls `main.py --config` and preflights those YAMLs and
checkpoint metadata before starting any workers. It does not implement another
training or evaluation engine.

```bash
# Metadata only: no GPU queries, models, dataset loads, W&B or output writes.
bash scripts/run_e3_source_nll_pilot.sh --dry-run

# E3.1 first, then E3.2; each phase runs both methods concurrently on GPUs 0/1.
bash scripts/run_e3_source_nll_pilot.sh

# Optional: only marginal calibration, or use different physical GPUs.
ALI_GPU=4 OTP_GPU=5 bash scripts/run_e3_source_nll_pilot.sh --phase e3_1
```

GPU 3 is excluded. Physical GPU IDs refer to the complete indices supplied by
`ALI_GPU` and `OTP_GPU`; each worker sees its own selected GPU as `cuda:0`.
The default output root is
`/data/lthung/flow2pc/experiments/source-nll-pilot-v1/`; override it with
`E3_PILOT_OUTPUT_ROOT` for a new experiment version. W&B project is `flow2pc`,
using the existing local/offline fallback; `WANDB_MODE=offline` is also accepted.

The pilot uses ST held-out index 1 (physical time 3, model time 1/3), one MC
seed, one 1,024-particle bank budget and up to 128 reserved test observations.
Both models use the same deterministic subset of held-out observations. Density
coordinate standardization is fitted only on observed training cells.
The teacher starts from the empirical earliest marginal (`source_split: all`);
for this target it is also the preceding observed marginal. Teacher density is
a numerical ODE particle pushforward with KDE, as in the existing E3 protocol.
Teacher solver is Dopri5, tolerance 1e-5, batch size 64, maximum 10,000 NFE.

The student uses its own direct PC distribution at the requested time. E3.1
scores the PC density directly and estimates HDR mass thresholds using PC draws.
E3.2 conditions the PC directly, without a student KDE or student ODE; the two
tasks are predict y given x and predict x given y. It uses 128 conditional draws
per bank and observation, and the predeclared teacher KDE bandwidth 0.1, without
a bandwidth sensitivity sweep. Levels are 50%, 70%, 90% and 95%. No recalibration
is fitted and no tail samples are removed.

Each `<experiment>/<method>/st/` folder contains `summary.csv`, `comparison.csv`,
`results.json`, the split/selection audits, and `figures/` with calibration and
comparison images. E3.1 also reports teacher/student region-mass fidelity;
E3.2 reports each mask separately in `by_time_mask.csv` and saves the conditional
mean/interval illustration in `figures/missing_coordinates.png`.

```bash
tail -f /data/lthung/flow2pc/experiments/source-nll-pilot-v1/_launcher/e3_1_alicfm.log
```

One MC seed and 128 test observations make this a diagnostic, not a publication
calibration result or a timing benchmark. ST is a previously inspected temporal
holdout, so the result is retrospective. A good pilot should be confirmed with
larger banks, more observations/seeds and teacher KDE sensitivity before making
calibration or speed claims. In particular, a near-90% empirical coverage alone
does not establish the quality of rarer events or all conditional distributions.
