# ALI-CFM transport W2 correction

GOM/Beijing pretraining now reports raw-coordinate W2 as its primary transport
metric and retains W1/EMD as a separate diagnostic. Per-time `teacher_w2` values
are averaged equally across the four temporal holdouts into
`teacher_heldout_w2`. The transport sweep ranks that field, rejects old W1-only
reports as incomplete, and records `transport_ode_w2_v1` in the metric protocol.

W2 uses squared Euclidean optimal-transport costs followed by a square root.
The two configured subsets are truncated to their first `min(N,M)` rows, as in
the existing common transport evaluator. Units are raw coordinates. Sampling is
the frozen instantaneous velocity ODE, scored at exact observation times. Native
OTP-FM finite-interval/nearby-time-minimum scores remain a distinct report; this
change does not claim those inference/time protocols are interchangeable.

No teacher/student architecture, training loss, optimizer, coupling, time map,
dataset preprocessing, or native method source is changed. The GOM/Beijing
YAML hyperparameters remain the previously retained W1-selected recipes, not
claimed W2 sweep winners. They can continue to supply the experiment checkpoints.

## Rescore the two retained teachers

```bash
bash local_runs/reevaluate_alicfm_transport_w2.sh
```

This is evaluation only, with two parallel CPU jobs. It preserves each old
`eval/` folder as `eval-w1-before-w2/`, verifies the teacher checkpoint hash, and
updates the existing `eval/` report/figures. Existing teacher predictions are
reused only if their exact old or current binding matches teacher, data,
preprocessing, source particles, time grid and numerical settings. Otherwise the
normal evaluator performs a new rollout. Cached-rollout reuse is explicitly
recorded in `metrics.json` under `execution`. Weights are never rewritten.

## Reevaluate the existing 1,024-checkpoint sweep in the background

```bash
bash local_runs/reevaluate_alicfm_transport_sweep_w2.sh --dry-run
bash local_runs/reevaluate_alicfm_transport_sweep_w2.sh
```

The second command detaches the existing controller with `nohup`. It uses GPUs
0–7, at most eight evaluation workers, one per GPU, W&B project `flow2pc` and
the existing `/data/lthung/flow2pc/alicfm-transport/sweep-512` checkpoint tree.
Valid cached trajectories avoid new GPU ODE solves; OT scoring runs on CPU.
`--evaluate-only` prevents training even if a checkpoint is missing. Completed
W2 evaluations are skipped. W1 metrics and original ranking files are preserved
before replacement; the derived summaries and `best_models.json` rank W2.

Controller log:
`/data/lthung/flow2pc/alicfm-transport/sweep-512/_launcher/reevaluate-w2.log`.
The matching `.pid` file records the controller PID. `ALICFM_SWEEP_ROOT`, `GPUS`,
`WANDB_MODE`, `CPU_THREADS_PER_RUN` and memory/disk limits can be overridden.
An existing controller lock still prevents duplicate jobs on this tree.

The sweep does not replace the retained pretrained-teacher folders or current
experiment teacher/student bindings. Selecting a different winner for a later
experiment is a separate checkpoint/config decision after reevaluation finishes.

## Refresh just the two portable score records after reevaluation

```bash
/data/nvidia/conda_envs/uffm/bin/python scripts/teacher_checkpoint_pack.py refresh-metrics \
  /data/lthung/flow2pc/teacher-checkpoints/ali-cfm/alicfm_gom.pt \
  /data/lthung/flow2pc/teacher-checkpoints/ali-cfm/alicfm_beijing.pt
```

This validates source hashes and corrected W2 reports, updates score metadata,
READMEs and the pack checksum index, and preserves previous versions in a hidden
backup folder. The exported weight fingerprints remain unchanged. It does not
evaluate or retrain anything.

## Optional future training

`local_runs/retrain_alicfm_transport_w2.sh` prepares a detached run of the two
retained recipes. `SCOPE=sweep` instead selects a fresh 512-per-dataset training
sweep. These are optional alternatives, not prerequisites for rescoring saved
checkpoints. Fresh training output defaults to a separate timestamped folder.
The reusable training sweep's default is now `sweep-w2-512`, preserving the old
W1-selected tree unless explicitly targeted for evaluation-only recovery.

## Verification status

Regression cases were added for the W2 formula/truncation, four-holdout
aggregation, W2 ranking when W1 ranking disagrees, rejection of stale W1 reports,
exact legacy trajectory-cache binding, and the evaluation-only training guard.
Tests and numerical reevaluation were not executed: the user's latest
instruction is to provide scripts only. Metadata-only dry runs and static
syntax/whitespace checks are separate from numerical validation.
