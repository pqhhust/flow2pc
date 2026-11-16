# Configuration layout

Runnable configurations use the same filenames across methods:

```text
configs/
  pretrain/
    alicfm/alicfm_<dataset>.yaml
    otpfm/otpfm_<dataset>.yaml
    mfm/mfm_<dataset>.yaml
    mmfm/mmfm_<dataset>.yaml
  distill/
    alicfm/alicfm_<dataset>_hclt.yaml
    otpfm/otpfm_<dataset>_hclt.yaml
    mfm/mfm_<dataset>_{hclt,gaussian}.yaml
    mmfm/mmfm_<dataset>_{hclt,gaussian}.yaml
  evaluate/
    mmfm/mmfm_<dataset>_{teacher,hclt,gaussian}.yaml
```

Pretraining creates a teacher, so its filename has no student suffix. Student
filenames explicitly distinguish HCLT and Gaussian PCs. Method variants use a
suffix on the dataset/recipe, such as `otpfm_gom_w2.yaml` or
`mfm_eb_t2_hclt.yaml`. Names describe existing supported configurations; this
layout does not imply every method supports every dataset or student type.

MFM additionally retains its original experiment wrappers under
`pretrain/mfm/native/` and their original-format checkpoint student recipes under
`distill/mfm/native/`. The primary MFM files use the shared prepared snapshots
and bound flow/geopath checkpoint pairs. The two paths have different data and
checkpoint contracts. Native parameter snapshots in `pretrain/mfm/upstream/`
retain their original filenames and contents because they are source reference
files, not Flow2PC stage configurations.

The organization is shared; method-specific YAML parameters, loss functions and
training loops remain method-specific. Existing launchers resolve these paths.

## Selected sweep teachers

The ALI-CFM GOM/Beijing presets, ten OTP-FM single-cell/knot/ST/tracking presets,
and six ALI/OTP Arch/Sphere/LiDAR presets contain the exact scientific settings
from the best completed sweep runs.
[Selected teacher results](../docs/SELECTED_TEACHERS.md) lists their scores;
[the index](../docs/selected_teachers.json) records original checkpoints, hashes,
resolved configurations and ranking scope.

Running a pretrain preset trains a new teacher in its configured output directory.
It does not reuse the selected checkpoint automatically. The ALI-CFM GOM/Beijing
distill presets already point to the existing selected checkpoint:

```bash
PYTHONPATH=src python main.py --config configs/pretrain/alicfm/alicfm_gom.yaml
PYTHONPATH=src python main.py --config configs/distill/alicfm/alicfm_gom_hclt.yaml
```

These are commands for the user to execute. Data and outputs remain external;
override `data.root`, `output.dir` or `teacher.checkpoint` through `--set` as needed.
The synthetic HCLT distill presets also directly reference the selected existing
checkpoints, for example `distill/alicfm/alicfm_arch_hclt.yaml` and
`distill/otpfm/otpfm_sphere_hclt.yaml`. Reserved endpoint rows select students;
Arch/Sphere middle populations remain test data.

The selected teacher presets log to W&B project `flow2pc` and retain final
evaluation/figures. Existing trained checkpoints and sweep outputs are preserved.

Research experiment YAMLs are grouped separately under
`experiments/<experiment>/{alicfm,otpfm}/`, with filenames such as
`alicfm_beijing_hclt.yaml`. They use `stage: evaluate` and an explicit experiment
name; they never start training. See [experiment protocols and launch commands](../docs/experiments/README.md).

OTP-FM distillation uses the restored `otpfm_conditional` sampler and the existing
HCLT student/velocity-MSE training. Presets load the ten selected shared-data
teachers; transport W2/W2Inf presets also load migrated reference teachers.
See [the recovered sampler and its supported protocols](../docs/OTPFM_DISTILLATION.md).
