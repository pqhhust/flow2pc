# MFM pretraining

The Arch, Sphere, LiDAR, CITE-5/50/100, EB-PCA5 and MULTI-5/50/100 entry configs in
this folder use the original MFM training pipeline. Other configs directly in
this folder retain the shared snapshot pipeline. Their comments
identify the parameter YAMLs in the external MFM checkout that informed each
adaptation; these presets do not claim to reproduce the original experiments.

For native MFM pretraining on the existing raw files under `data/single_cell/`, use:

```bash
python scripts/run_mfm_data.py pretrain cite100
python scripts/run_mfm_data.py pretrain multi100
```

The entry configs are `mfm_{arch,sphere,cite5,cite50,cite100,eb_pca5,multi5,multi50,multi100}.yaml`;
their native parameter YAMLs are under `configs/pretrain/mfm/native/` and
match the corresponding upstream OT-MFM settings. CITE additionally applies
the user-requested min-max transform after optional whitening.
Training imports the bundled
MFM implementation under `flow2pc.pretrain.models.mfm._source`, not the
external checkout. The launcher leaves the input file untouched and reserves
a fresh run directory each time;
the native code trains separately for each configured `t_exclude` setting. This is not
the prepared-snapshot protocol, and the snapshot distillation configs
do not automatically select either native checkpoint.
MULTI-50, EB-PCA5, Arch and Sphere now use their matching upstream YAML values.
EB reads the existing `data/single_cell/ebdata_v3.h5ad` without modifying it.
Arch and Sphere use the original seeded generators in the upstream loader; the
cached arrays under `data/arch` and `data/sphere` remain untouched. LiDAR uses
the source trainer and its original `rainier2-thin.las` input, copied from the
upstream repo into `data/lidar/` (SHA-256 recorded by its source manifest).

For example:

```bash
python main.py --config configs/pretrain/mfm/mfm_cite5.yaml --validate-only
```

Profiles without a matching upstream source YAML are deferred. The launcher
rejects them for native pretraining; it does not silently fall back to the
snapshot trainer. Previous snapshot configs are archived under
`configs/pretrain/mfm/snapshots/` for existing workflows only.
After training, temporal holdouts are evaluated from the selected teacher pair
and written to `eval/test_emd.json`; `eval/status.json` records whether
evaluation completed. An evaluation error leaves the trained teacher pair
available. This uses a 100-step Euler rollout from
the full preceding observed marginal and exact uniform-mass Euclidean W1
against the full held-out marginal. The rollout is batched, but OT uses every
point in both populations. The report records the counts
and provenance, plus initial-marginal baselines and distances in original
coordinates (`test_EMD_raw`). Its normalized values are not directly comparable
to runs with different dimensions or preprocessing.
For `arch`, evaluation also writes `arch_trajs.png`. For `sphere`, it writes
`sphere_trajs.png` and three additional views and records
`mean_distance_from_sphere`. For `lidar`, it writes `lidar_interpolant.png` in
the original scene coordinates; LiDAR has no temporal holdout or `test_EMD`.
When W&B tracking is enabled, the post-train step resumes the teacher's run
and logs these metrics and images. `eval/wandb_status.json` records upload
success or failure separately from teacher and local evaluation status.
To backfill already completed snapshot runs without retraining:

```bash
python scripts/evaluate_mfm_test_emd.py --all --device cpu
```

Add `--wandb` to append backfilled diagnostics to the existing tracked runs.

Supported native pretraining writes to a new
`outputs/pretrain/mfm/<dataset>/run-<timestamp>-<id>/`
directory on each invocation. Set
`--run-name <name>` in the launcher (or override `output.dir` with a non-default
path in `main.py`) to choose a specific run directory; an existing directory
is never reused. Automatic `both` is unavailable because native checkpoints
do not use the snapshot checkpoint-pair format.
Their corresponding students are `configs/distill/mfm/mfm_<dataset>_<student>.yaml`.

The matching source recipes are `arch/ot-mfm.yaml`, `lidar/mfm.yaml`,
`sphere/mfm.yaml`, and `single_cell/{5dims,50dims,100dims}/ot-mfm_{cite,multi}.yaml`
in `../pretrain/metric-flow-matching/configs`. `mfm_eb_pca5.yaml` uses
`single_cell/5dims/ot-mfm_eb.yaml` as its parameter source.

The archived snapshot backend supports one LAND `gamma_current` per run. Thus the CITE-5
and MULTI-5 presets use the first source gamma (0.125), rather than the source's
per-holdout gamma list. The CITE/MULTI 50/100-dimensional presets use the
source RBF metric with its `n_centers: 150`, `kappa: 1.5`, `rho: -2.75`,
`metric_epochs: 200`, and `alpha_metric: 1.0`. Their snapshot data protocol
still differs from the original experiment protocol.
