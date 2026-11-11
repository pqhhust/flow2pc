# Selected ALI-CFM and OTP-FM teachers

Selection snapshots: real datasets 2026-11-05; Arch/Sphere/LiDAR 2026-11-06. Each preset copies the winning run’s seed,
data/preprocessing, model parameters, training budget and evaluation settings.
Only the output directory and W&B run name change for a new pretraining run.
Selected existing checkpoint paths, original resolved-config paths, hashes,
media manifests and per-dataset completed counts are in
[selected_teachers.json](selected_teachers.json). No experiments were run to
make this selection.

| Teacher | Dataset | Saved selection metric | EMD ↓ | Completed / planned | Pretrain preset |
| --- | --- | --- | ---: | ---: | --- |
| alicfm | beijing | Held-out-time W1 | 12.581962 | 512 / 512 | [YAML](../configs/pretrain/alicfm/alicfm_beijing.yaml) |
| alicfm | gom | Held-out-time W1 | 0.133903 | 512 / 512 | [YAML](../configs/pretrain/alicfm/alicfm_gom.yaml) |
| otpfm | cell_tracking | Observed-time W1 | 34.825586 | 28 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_cell_tracking.yaml) |
| otpfm | cite100 | Held-out-time W1 | 48.923553 | 32 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_cite100.yaml) |
| otpfm | cite5 | Held-out-time W1 | 0.867000 | 33 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_cite5.yaml) |
| otpfm | cite50 | Held-out-time W1 | 43.121042 | 32 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_cite50.yaml) |
| otpfm | eb | Held-out-time W1 | 0.424751 | 32 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_eb.yaml) |
| otpfm | knot | Observed-time W1 | 0.015208 | 32 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_knot.yaml) |
| otpfm | multi100 | Held-out-time W1 | 52.916254 | 31 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_multi100.yaml) |
| otpfm | multi5 | Held-out-time W1 | 0.912809 | 32 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_multi5.yaml) |
| otpfm | multi50 | Held-out-time W1 | 46.237329 | 32 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_multi50.yaml) |
| otpfm | st | Held-out-time W1 | 117.998783 | 31 / 100 | [YAML](../configs/pretrain/otpfm/otpfm_st.yaml) |

## Selection scope

ALI-CFM GOM/Beijing were selected from all 512 completed configurations per
dataset, minimizing mean raw-coordinate W1 EMD across the four saved temporal
holdouts. Those holdouts are tuning/validation populations, not an untouched
test set. Beijing remains in its original representation; no log transform or
prediction clipping was introduced.

OTP-FM single-cell/ST/knot/tracking teachers were selected from the 315 completed runs in their 1,000-run sweep.
The remaining 685 runs are marked failed in the saved summary; these presets
are best among the completed models, not evidence that all 100 settings per
dataset finished. Single-cell includes CITE and Multi at 5D, 50D and 100D,
plus EB. Their left-out-time W1 is the saved ranking rule, as it is for ST.
Knot and cell tracking use observed-time reconstruction W1 because their
protocols have no held-out times. Tracking evaluates the six README frames:
1, 25, 50, 75, 100 and 115. Metrics retain each sweep’s coordinate system,
population initialization, sample budget and solver; scores across datasets
or different evaluation protocols are not interchangeable.

The selected checkpoint files were verified against the SHA-256 recorded in
their manifests. Saved metrics and ranking files agree with the selected
rows. No checkpoint, data array, running sweep or existing output was changed.

## Selected synthetic teachers

The Arch/Sphere/LiDAR sweep ended with 548 completed, 928 failed and 60
blocked runs out of 1,536 planned configurations. Only completed models
were eligible. Ranking uses reserved endpoint-row validation W1 from
`eval/selection.json`; Arch/Sphere middle-time test EMD never selects a model.
LiDAR retains its saved standardized metric coordinates; Arch/Sphere use
identity coordinates. These remain two-observation baselines, with no added
manifold/terrain objective.

| Teacher | Dataset | Endpoint validation W1 ↓ | Middle-time test W1 ↓ | Complete / planned | Pretrain | Distill |
| --- | --- | ---: | ---: | ---: | --- | --- |
| alicfm | arch | 0.033440 | 0.631537 | 95 / 256 | [YAML](../configs/pretrain/alicfm/alicfm_arch.yaml) | [HCLT](../configs/distill/alicfm/alicfm_arch_hclt.yaml) |
| alicfm | sphere | 0.081391 | 0.659763 | 95 / 256 | [YAML](../configs/pretrain/alicfm/alicfm_sphere.yaml) | [HCLT](../configs/distill/alicfm/alicfm_sphere_hclt.yaml) |
| alicfm | lidar | 0.037562 | — | 87 / 256 | [YAML](../configs/pretrain/alicfm/alicfm_lidar.yaml) | [HCLT](../configs/distill/alicfm/alicfm_lidar_hclt.yaml) |
| otpfm | arch | 0.031111 | 0.627394 | 93 / 256 | [YAML](../configs/pretrain/otpfm/otpfm_arch.yaml) | [HCLT](../configs/distill/otpfm/otpfm_arch_hclt.yaml) |
| otpfm | sphere | 0.149838 | 0.760534 | 92 / 256 | [YAML](../configs/pretrain/otpfm/otpfm_sphere.yaml) | [HCLT](../configs/distill/otpfm/otpfm_sphere_hclt.yaml) |
| otpfm | lidar | 0.034300 | — | 86 / 256 | [YAML](../configs/pretrain/otpfm/otpfm_lidar.yaml) | [HCLT](../configs/distill/otpfm/otpfm_lidar_hclt.yaml) |

The six pretrain presets reproduce the winning saved settings in a new
output directory. The six distill presets directly reference the existing
winning checkpoints, so using them does not retrain a teacher. Student
architecture, optimizer and velocity-MSE objective remain the existing recipe;
selection uses reserved observed rows and leaves the middle time for testing.

## Using the selected teachers

Running a pretrain preset starts fresh training with the selected settings.
Its output is `/data/lthung/flow2pc/pretrain/METHOD/DATASET/best-config-seed42`.
All selected presets retain final evaluation/images and W&B project `flow2pc`.
An existing output directory is not automatically overwritten.

The two ALI-CFM student presets already load the existing selected teacher:

- [GOM HCLT](../configs/distill/alicfm/alicfm_gom_hclt.yaml)
- [Beijing HCLT](../configs/distill/alicfm/alicfm_beijing_hclt.yaml)

They optimize velocity MSE and select the student using the implemented
held-out-time velocity-MSE validation. Population metrics remain diagnostics.
Their output directories are under `/data/lthung/flow2pc/distill/alicfm/`.
If retraining a teacher, explicitly point `teacher.checkpoint` to the new
teacher rather than continuing to reference the archived sweep winner.

OTP-FM student presets now use the restored conditional-tuple sampler, with the
existing HCLT student and velocity-MSE training. Each selected teacher has a
`distill_config` in the index. See [the restored integration](OTPFM_DISTILLATION.md)
for commands, validation policy and supported direct W2Inf paths.

The Arch/Sphere/LiDAR sweep settings and its endpoint-only interpretation
remain as documented in [MFM_TEACHER_SWEEP.md](MFM_TEACHER_SWEEP.md).

All config moves preserve other presets’ scientific contents. Static syntax,
path and saved-setting comparisons are separate from runtime validation.
Regression cases are provided in `tests/test_config_layout.py`; no tests,
training, distillation or evaluation were run during this change.
