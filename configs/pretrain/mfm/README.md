# MFM pretraining

The configs in this folder use the shared snapshot pipeline. Their comments
identify the parameter YAMLs in the external MFM checkout that informed each
adaptation; these presets do not claim to reproduce the original experiments.

For example:

```bash
python main.py --config configs/pretrain/mfm/mfm_cite5.yaml --validate-only
```

The `mfm_<dataset>.yaml` files directly in this folder use the shared prepared
snapshot pipeline. They export a bound `checkpoints/{flow,geopath}.pt` pair.
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

Pretraining writes to `outputs/pretrain/mfm/<dataset>/default/`; the MFM launcher
uses the same layout and substitutes `--run-name` for `default`.
Their corresponding students are `configs/distill/mfm/mfm_<dataset>_<student>.yaml`.

The matching source recipes are `arch/ot-mfm.yaml`, `lidar/mfm.yaml`,
`sphere/mfm.yaml`, and `single_cell/{5dims,50dims,100dims}/ot-mfm_{cite,multi}.yaml`
in `../pretrain/metric-flow-matching/configs`. `mfm_eb_pca5.yaml` uses
`single_cell/5dims/ot-mfm_eb.yaml` as its parameter source. The other profiles
have no exact dataset counterpart in that config tree and retain the documented
starting recipes in their individual files.

The snapshot backend supports one LAND `gamma_current` per run. Thus the CITE-5
and MULTI-5 presets use the first source gamma (0.125), rather than the source's
per-holdout gamma list. The CITE/MULTI 50/100-dimensional presets use the
source RBF metric with its `n_centers: 150`, `kappa: 1.5`, `rho: -2.75`,
`metric_epochs: 200`, and `alpha_metric: 1.0`. Their snapshot data protocol
still differs from the original experiment protocol.
