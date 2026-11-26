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
`mfm_eb_phate2_hclt.yaml`. Names describe existing supported configurations; this
layout does not imply every method supports every dataset or student type.

MFM presets use shared prepared snapshots and bound flow/geopath checkpoint
pairs. Comments in those presets identify the original parameter YAMLs in the
external MFM checkout; those source YAMLs are not Flow2PC stage configurations.

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

For student hyperparameter sweeps against the current frozen ALI-CFM/OTP-FM
teachers, see [student sweeps](../docs/STUDENT_SWEEPS.md). These inherit the
presets here, generate run YAMLs under the external output root, and rank by
the dataset-specific velocity-MSE validation score, independently of EMD/UQ.

The 19 presets covered by the completed 1,900-run student sweep were manually
synchronized with its winners on 2026-11-21: eight ALI-CFM and eleven OTP-FM
recipes. They retain the exact saved teacher, student settings, validation and
evaluation protocol, W&B name, and existing winning output directory. Each YAML
records its source `best_config.yaml` and selected checkpoint step in comments.
For a new training replication, override `output.dir`; the default identifies
the already completed winning run. This synchronization does not switch the
evaluation to full populations or redirect existing experiment configs.

Research experiment YAMLs are grouped separately under
`experiments/<experiment>/{alicfm,otpfm}/`, with filenames such as
`alicfm_beijing_hclt.yaml`. They use `stage: evaluate` and an explicit experiment
name; they never start training. See [experiment protocols and launch commands](../docs/experiments/README.md).

E4 presets under `experiments/e4/{alicfm,otpfm}/` pin the Beijing/ST pilot
checkpoints and declare teacher-relative initial-law shifts. The dataset choice,
sampling law and unsupported HDR gate are explicit in
[the E4 protocol](../docs/experiments/E4_SHIFT.md).

OTP-FM distillation uses the restored `otpfm_conditional` sampler and the existing
HCLT student/velocity-MSE training. Presets load the ten selected shared-data
teachers; transport W2/W2Inf presets also load migrated reference teachers.
See [the recovered sampler and its supported protocols](../docs/OTPFM_DISTILLATION.md).

The optional ALI-CFM/OTP-FM initial-density pilot has separate ST configs under
`distill/{alicfm,otpfm}/source_anchor/`; selected experiment recipes remain unchanged.
See [the objective, validation and launch protocol](../docs/experiments/SOURCE_DENSITY_ANCHOR.md).
The 42-run ST refinement reuses these same configs and the existing pilot
controller, varying only NLL weight and learning rate:
`bash scripts/run_source_nll_refinement_8gpu.sh`. It does not replace selected
experiment recipes or train on student-generated samples.

Four confirmation recipes under `source_anchor/confirmation/` pin completed
seed-0 students for the repeated-seed check. The normal distillation YAML
format is retained. Use `bash scripts/run_source_nll_confirmation_7gpu.sh` to
reuse them and train seeds 1/2; see
[the protocol](../docs/experiments/SOURCE_NLL_CONFIRMATION.md).

The CITE-5D/50D/100D and knot source-NLL recipes share `source_anchor/` and
preserve each selected recipe's final benchmark settings. See the
[40-run pilot and separate query audits](../docs/experiments/SOURCE_NLL_MULTIDATA.md).
