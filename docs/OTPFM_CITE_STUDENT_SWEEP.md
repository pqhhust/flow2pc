# CITE-50D and CITE-100D OTP-FM student sweep

The default plan has 100 student configurations per dataset, 200 total. It loads
the existing frozen teacher from each base preset:

- `configs/distill/otpfm/otpfm_cite50_hclt.yaml`
- `configs/distill/otpfm/otpfm_cite100_hclt.yaml`

Only student capacity/time-feature size and learning rate vary:

| Parameter | Values |
| --- | --- |
| HCLT latent states | 64, 96, 128, 192, 256 |
| Fourier time features | 16, 32, 64, 128 |
| Learning rate | 0.0001, 0.0003, 0.001, 0.002, 0.003 |

This is the full 5 × 4 × 5 grid. The original h128/f32/lr0.001 setting runs first
for each dataset as a control. Remaining combinations use a deterministic mixed
order, alternating datasets. Every run uses the existing seed 0, batch size 64,
20,000-step budget, empirical initialization, AdamW/cosine schedule, clipping,
OTP conditional query distribution and velocity-MSE objective. No teacher is
retrained. Teacher weights, coupling, alpha, data splits and evaluation sample
budgets are inherited unchanged. Increasing the default maximum capacity from
128 to 256 explores larger students while bounding the per-run capacity.

The reusable controller is `scripts/run_otpfm_distill_sweep.py`. It uses the
existing `DistillationQueue`/`ParallelSweep`; each worker invokes the common
`main.py --config ...` distillation path. The gitignored local recipe is:

```bash
# Metadata preview; no Torch, GPU query, output writes, W&B or worker launch.
bash local_runs/sweep_otpfm_cite.sh --dry-run

# User command to start the sweep.
bash local_runs/sweep_otpfm_cite.sh
```

Defaults are GPUs 0–7, at most two workers per GPU and 16 concurrent workers,
four CPU threads per worker, at least 12 GiB free GPU memory for each new launch,
and 20 GiB disk headroom. Actual concurrency depends on available memory.
`GPUS`, `WORKERS_PER_GPU`, `MAX_PARALLEL`, `CPU_THREADS_PER_RUN`, `MIN_FREE_GB`,
`MIN_DISK_FREE_GB`, `WANDB_MODE`, `FLOW2PC_PYTHON` and `OTP_CITE_SWEEP_ROOT` can be
overridden. W&B uses project `flow2pc`; names start with `cite50d` or `cite100d`
and include latent/time-feature size, learning rate, batch size, steps, seed and
configuration hash. The existing offline fallback remains available.

Outputs use a separate namespace, preserving the completed baseline students:

```text
/data/lthung/flow2pc/distill/otpfm/cite-sweep-100/
  distillation_plan.json
  summary.csv
  _configs/DATASET/RUN.yaml
  _launcher/{status.json,logs/}
  DATASET/best_models.json
  DATASET/RUN/checkpoints/{best,last}.pt
  DATASET/RUN/eval/{metrics.json,media.json,comparison_marginals.png,...}
  DATASET/RUN/eval/population/evaluation.json
```

Generated run configs are stored with outputs, rather than adding hundreds of
YAMLs to Git. Repeating the command reuses completed configurations and resumes
compatible checkpoints. A partial output without a checkpoint requires the
existing `--retry-incomplete` archival option. A second controller cannot own
the same sweep root concurrently. The baseline batch has a different root.

Checkpoint selection and per-dataset top-ten ranking use validation
student–teacher velocity MSE only. The existing CITE observed-row validation
policy is retained. EMD does not select a winner. Each completed run saves and
logs teacher/student comparison images. The CSV and ranked records retain
teacher/student mean, held-out and final-time EMD, plus final paired ODE MSE,
so day-7 drift is visible alongside the selection score. These diagnostics use
the original full-dimensional raw-coordinate CITE metrics.

To refresh summaries/rankings without starting workers:

```bash
bash local_runs/sweep_otpfm_cite.sh --summarize-only
```

The controller also accepts explicit `--latents`, `--time-features`,
`--learning-rates`, `--datasets cite50|cite100`, and `--steps` overrides. These
change the generated plan; grid axes must contain unique valid values.
Different scientific settings receive different run paths.

Regression contracts are in `tests/test_otpfm_distill_sweep.py`. Only static
checks and metadata-only previews were performed while preparing this sweep;
no training, evaluation or tests were launched.
