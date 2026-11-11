# Train all existing ALI-CFM and OTP-FM presets

From the repository root:

```bash
bash local_runs/pretrain_all_teachers.sh --dry-run
bash local_runs/pretrain_all_teachers.sh
```

This queues every top-level YAML in `configs/pretrain/alicfm/` and
`configs/pretrain/otpfm/`: currently 36 teachers (15 ALI-CFM, 21 OTP-FM).
It covers Arch, LiDAR, Sphere, knot, cell tracking, spatial transcriptomics,
CITE/Multi at 5/50/100 dimensions, EB, GOM, and Beijing. OTP-FM includes the
existing W2, W2Inf, MMD, and KL GOM/Beijing presets. The current direct OTP
student sampler supports W2/W2Inf; KL/MMD teacher pretraining is included here
but does not imply support for distilling those nonlinear conditional paths.

Each worker runs `main.py --config ...` with the existing method-specific
training and final evaluation. Seeds, losses, coupling, budgets, preprocessing,
splits, and evaluation settings remain those of the source YAML. Only output,
logging, assigned device, and missing synthetic data paths are adjusted.
The moved synthetic data resolve to
`/data/pqhung_group/pdtung/flow2pc/data/{arch,sphere,lidar}/seed42`.
`--prepared-root` can explicitly choose that parent directory.

The shared queue runs at most eight independent teachers, one per GPU 0–7,
and gives each freed GPU the next pending job. It waits when fewer than
12 GiB are free on a GPU or fewer than 20 GiB are free on the output filesystem.
It uses `nvidia-smi` rather than initializing CUDA in the controller.
W&B project is `flow2pc`; names include the model, dataset/recipe, and seed.
For an unavailable W&B service, explicitly use `--mode offline` to cover
native ALI presets as well as the methods with existing automatic fallbacks.

All outputs live under `/data/lthung/flow2pc/pretrained-teachers/`:

```text
alicfm/cite50/checkpoints/last.pt
alicfm/gom/checkpoints/last.pt
otpfm/cite50/checkpoints/last.pt
otpfm/gom_w2/checkpoints/best.pt
METHOD/PRESET/eval/media.json
summary.csv
checkpoints.json
pretrain_plan.json
_launcher/status.json
_launcher/logs/MODEL_PRESET.log
```

Use `checkpoints.json` for actual checkpoint paths: native OTP-FM transport
exports `best.pt`, while ALI-CFM and shared-data OTP-FM export `last.pt`.
Keep the whole run directory, including manifest and preprocessing metadata,
for later distillation. Metrics preserve their native definitions: OTP-FM
transport W2 is labeled separately from W1 EMD.

Completed matching runs are skipped. Partial/conflicting runs are preserved;
the launcher neither deletes nor silently restarts them. An individual worker
failure does not cancel the remaining queue. To retry selected presets in a
fresh directory:

```bash
bash local_runs/pretrain_all_teachers.sh \
  --output-root /data/lthung/flow2pc/pretrained-teachers-retry \
  --presets alicfm_cite50 otpfm_cite50
```

The `local_runs/` wrapper is intentionally gitignored. The equivalent reusable
entry point is:

```bash
/data/nvidia/conda_envs/uffm/bin/python scripts/run_pretrain_all.py
```

Preparing this launcher did not execute training, evaluation, or tests.
