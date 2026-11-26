# Compact teacher export

The approved export is `/data/lthung/flow2pc/teacher-checkpoints/`: 30
self-contained `.pt` files, matching per-teacher README files, one standalone
unpack helper, a checksum index, and a folder README. No raw data, images, logs,
intermediate checkpoints, or unused optimizer/RNG/discriminator state is copied.
Checkpoints and matching READMEs are grouped into `ali-cfm/` and `otp-fm/`;
the shared helper, index, and general README stay at the export root.

Selection is one ALI-CFM and one OTP-FM teacher for each of the 15 existing
dataset profiles: Arch, LiDAR, Sphere, knot, tracking, GOM, Beijing, CITE and
Multi at 5/50/100 dimensions, EB, and ST. OTP-FM GOM/Beijing use the W2 recipe,
which has the lowest saved native W2 score among the four completed recipes.
ALI-CFM tracking uses the successful historical normalized-coordinate teacher
at `../ali-cfm/changed_files/ali_cfm_model.pth`, rather than the recent failed
raw-pixel teacher. The remaining 29 exports come from `pretrained-teachers/`.

Each file embeds weights and the essential metadata needed to restore its
current loader interface. ALI-CFM retains the frozen velocity field **and**
interpolant. OTP-FM retains the full model state, EMA weights, potential buffers,
checkpoint correction strength, and saved OT chain indices where applicable.
Original dataset/time/preprocessing/config/source-provenance records are kept.
For source methods that export a final checkpoint only, that final checkpoint
is retained; no earlier best checkpoint is invented.

The `.pt` is an export bundle, rather than a drop-in raw state dict for the
existing loader. Unpacking restores the minimal directory layout without
modifying training code or running a model. The original weights are preserved
exactly. Slimming changes checkpoint file bytes, so the helper regenerates
consistent manifest and preprocessing bindings and records the original SHA-256.
These exports support inference/distillation, not resuming teacher optimization.

Copy the whole export directory to the drive. From the copied directory:

```bash
python unpack_teacher.py verify ali-cfm/*.pt otp-fm/*.pt
python unpack_teacher.py unpack ali-cfm/*.pt otp-fm/*.pt --output-dir /path/to/restored-teachers
```

The helper needs PyTorch and PyYAML. `restored_teachers.json` contains the
`teacher` config fragments, including explicit historical tracking metadata.
Keep each unpacked teacher's small directory together. The Flow2PC repository
still supplies architecture/adapter code, and datasets must be available
separately for distillation/evaluation. Original data paths remain recorded;
the export neither copies datasets nor silently changes their contracts.

The reusable exporter in this repository is:

```bash
/data/nvidia/conda_envs/uffm/bin/python scripts/teacher_checkpoint_pack.py export
```

It refuses to overwrite an existing destination, verifies each source checkpoint
hash/binding, stages the complete result, verifies the reloaded bundle's exact
tensor and metadata fingerprints, then publishes the new directory. Existing
training directories and running jobs are untouched.
