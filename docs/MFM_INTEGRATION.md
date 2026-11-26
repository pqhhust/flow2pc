# MFM integration

The shared snapshot workflow is documented in [MFM local data](MFM_LOCAL_DATA.md).
It covers 20 non-AFHQ profiles with explicit splits/time grids, paired
checkpoint export and both retained PC students.

Method-owned code lives under `src/flow2pc/pretrain/models/mfm/`, following
the same organization as ALI-CFM:

- `trainer.py`: registry method and original pretraining command preparation/execution.
- `model.py`: original network construction and native weight restoration.
- `adapter.py`: frozen velocity interface and geopath sampling.
- `checkpoint_adapter.py`: paired checkpoint loading, metadata validation and freezing.
- `base.py`: loaded-teacher result contract.

The former `pretrain/mfm_native.py` is now `models/mfm/trainer.py`;
the MFM-only `pretrain/metric_backends.py` registry was removed in favor of
the method's checkpoint adapter. CLI/config callers use the new module paths.
Shared data, distillation and configuration helpers remain in their own layers.
The native CITE/MULTI paths now bundle the upstream MFM modules under
`models/mfm/_source/`; see `NOTICE.md` for provenance.

Use `scripts/run_mfm_data.py` to select a profile and stage:

```bash
python scripts/run_mfm_data.py pretrain cite100 --dry-run
python scripts/run_mfm_data.py pretrain multi5 --dry-run
python scripts/run_mfm_data.py prepare cite5
python scripts/run_mfm_data.py pretrain cite100
python scripts/run_mfm_data.py distill cite5 --student both --teacher-run <snapshot-run>
```

The launcher trains CITE-5/50/100 and MULTI-5/50/100 through the bundled original MFM code without a
backend switch. Prepared-snapshot distillation remains available for existing
snapshot checkpoint pairs; automatic `both` is unavailable. `--dry-run` prints
commands without starting jobs. Use
`FLOW2PC_PYTHON` to select the environment and `CUDA_VISIBLE_DEVICES` to
restrict GPUs.

Data, logging, output paths and flow/geopath checkpoints are read from the
existing configs. The launcher points student jobs at the flow/geopath pair
exported by the matching teacher run. Direct CLI invocations need that pair set
explicitly in `teacher.checkpoint` and `teacher.params.geopath_checkpoint`.

The MFM method is isolated from ALI-CFM's manifest-backed runner. New snapshot
pretraining uses the bundled MFM implementation by default; an explicit
`model.params.source_root` can select a separate checkout.
The native presets use the bundled implementation and Flow2PC-local
parameter YAML. The snapshot presets record their original parameter
sources in comments. Distillation loads
frozen flow and geopath checkpoints with the adapter copied from
`flow2pc-multi-marginal`; the loader uses the run's saved source path for
existing snapshot checkpoints. No source algorithm was replaced with a generic
TorchCFM loss.

Shared snapshot presets cover remaining profiles, including LiDAR, and other
non-image datasets. Their starting parameters came from the MFM checkout's
configuration files; the checked-in snapshot was copied on 2026-10-31 from
commit `e44e03ec6207d2b111c1d51a03dcc53a98f87b95` with local configuration
changes. Snapshot pretraining exports a bound `.pt` flow/geopath
pair. The distillation adapter checks the pair, records hashes, and freezes
both networks.

MFM data loading retains the source's ordinal unit-interval model times. It
records original observation labels separately as physical times where numeric,
plus the split indices, normalization and source file hashes. AFHQ image classes
are labels, not physical times. The MFM student loss is velocity MSE only;
source NLL has been retired. ALI-CFM also uses its existing frozen-teacher
velocity MSE objective.

Existing whitened MFM checkpoints were trained with the upstream scaler fitted
on all source frames, including temporal holdouts. The adapter records
`fit_scope: all_source_frames_as_in_upstream` and reuses those coordinates to
match the frozen teacher. This is a known exception to the proposed
training-only normalization rule; changing it requires retraining the teacher.

MMFM is integrated separately; see [its integration notes](MMFM_INTEGRATION.md).
`ConditionalGaussianPC`, MNIST latent code and source benchmark launchers were
not copied. The native CITE/MULTI paths use the existing raw files under
`flow2pc/data/single_cell` and no external source checkout. Snapshot runs and
legacy checkpoints may still refer to an external checkout. Runtime numerical
parity and checkpoint reload still need verification when test execution is authorized.

Student presets now enable final evaluation and W&B project `flow2pc`.
`best.pt` is restored for final evaluation, both direct-PC and ODE predictions
are retained, and shared metrics use exact EMD/W1 rather than sliced Wasserstein.
See [dataset-specific validation and benchmark conventions](EXPERIMENT_PROTOCOLS.md).
