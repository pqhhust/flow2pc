#!/usr/bin/env python3
"""Export or unpack self-contained inference/distillation teachers; no inference.

The unpack/verify commands are standalone (Python, PyTorch and PyYAML only).
No model is instantiated, dataset loaded, GPU initialized, or training started.
"""

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

FORMAT = "flow2pc_portable_teacher_v1"
METHOD_FOLDERS = {"alicfm": "ali-cfm", "otpfm": "otp-fm"}
ROOT = Path(__file__).resolve().parents[1]
EXTRAS = ("resolved_config.yaml", "native_config.json", "sampling_contract.json", "ot_chain_indices.npy")
DATASETS = ("arch", "beijing", "cell_tracking", "cite5", "cite50", "cite100", "eb",
            "gom", "knot", "lidar", "multi5", "multi50", "multi100", "sphere", "st")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def identity(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def fingerprint(value):
    """Exact tensor bytes, dtypes, shapes, keys and primitive metadata."""
    result = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            result.update(str((str(item.dtype), tuple(item.shape))).encode())
            result.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b"dict")
            for key in sorted(item):
                result.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(str(type(item).__name__).encode())
            for part in item:
                visit(part)
        else:
            result.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())

    visit(value)
    return result.hexdigest()


def validate_bundle(bundle):
    if bundle.get("format") != FORMAT or bundle.get("method") not in ("alicfm", "otpfm"):
        raise ValueError("Unknown portable teacher format or method")
    if bundle.get("dataset") not in DATASETS:
        raise ValueError("Unknown portable teacher dataset")
    if fingerprint(bundle["weights"]) != bundle["integrity"]["weights"]:
        raise ValueError("Portable weights checksum mismatch")
    if set(bundle["files"]) != set(bundle["integrity"]["files"]):
        raise ValueError("Portable metadata inventory mismatch")
    for name, data in bundle["files"].items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Portable metadata filename must remain inside the run directory")
        if digest(data) != bundle["integrity"]["files"][name]:
            raise ValueError(f"Portable metadata checksum mismatch: {name}")
    if bundle["source_format"] == "framework":
        manifest = json.loads(bundle["files"]["manifest.json"])
        preprocessing = json.loads(bundle["files"]["preprocessing.json"])
        if (manifest["method"] != bundle["method"]
                or manifest["preprocessing_file"] != "preprocessing.json"
                or digest(bundle["files"]["preprocessing.json"]) != manifest["preprocessing_sha256"]
                or identity(preprocessing) != manifest["dataset"]["preprocessing_id"]):
            raise ValueError("Portable teacher preprocessing/manifest binding mismatch")
        required = {"cfm_model", "interpolant"} if bundle["method"] == "alicfm" else {"model_state_dict"}
        if set(bundle["weights"]) != required:
            raise ValueError("Portable teacher has missing or unexpected model states")
    elif bundle["source_format"] == "legacy":
        if bundle["method"] != "alicfm" or bundle["dataset"] != "cell_tracking":
            raise ValueError("Only the explicitly selected tracking legacy teacher is exported")
        if set(bundle["weights"]) != {"cfm_model", "interpolant", "min_max", "config"}:
            raise ValueError("Incomplete historical tracking teacher")
    else:
        raise ValueError("Unknown source artifact format")
    return bundle


def read_bundle(path):
    return validate_bundle(torch.load(path, map_location="cpu", weights_only=True))


def framework_bundle(directory, dataset):
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if json.loads((directory / "status.json").read_text()).get("status") != "complete":
        raise ValueError(f"Teacher did not complete: {directory}")
    relative = Path(manifest["checkpoint_relpath"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid original checkpoint path")
    checkpoint = directory / relative
    if digest(checkpoint.read_bytes()) != manifest["checkpoint_sha256"]:
        raise ValueError(f"Original checkpoint hash mismatch: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["manifest_identity"] != identity({k: v for k, v in manifest.items() if k != "checkpoint_sha256"}):
        raise ValueError(f"Original checkpoint metadata binding mismatch: {checkpoint}")
    method = manifest["method"]
    keys = ("cfm_model", "interpolant") if method == "alicfm" else ("model_state_dict",)
    files = {"manifest.json": manifest_bytes, "preprocessing.json": (directory / "preprocessing.json").read_bytes()}
    files.update({name: (directory / name).read_bytes() for name in EXTRAS if (directory / name).is_file()})
    metrics = json.loads((directory / "eval/metrics.json").read_text())
    return {
        "format": FORMAT, "method": method, "dataset": dataset, "source_format": "framework",
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": manifest["checkpoint_sha256"],
        "weights": {k: payload[k] for k in keys}, "files": files, "saved_metrics": metrics,
        "selection": manifest.get("selected_checkpoint", {"rule": "final configured CFM checkpoint"}),
        "omitted_payload_keys": sorted(set(payload) - set(keys) - {"manifest_identity"}),
    }


def historical_bundle(checkpoint, metadata_path):
    metadata = json.loads(metadata_path.read_text())
    source_hash = digest(checkpoint.read_bytes())
    if source_hash != metadata["checkpoint_sha256"]:
        raise ValueError("Historical tracking checkpoint differs from its pinned identity")
    # This explicitly selected local legacy checkpoint is trusted and contains
    # original training metadata. The exported format uses weights_only=True.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = json.loads(json.dumps(saved["config"], allow_nan=False))
    return {
        "format": FORMAT, "method": "alicfm", "dataset": "cell_tracking", "source_format": "legacy",
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": source_hash,
        "weights": {"cfm_model": saved["cfm_model"], "interpolant": saved["interpolant"],
                    "min_max": saved["min_max"], "config": config},
        "files": {"legacy_metadata.json": json_bytes(metadata)},
        "saved_metrics": {}, "selection": {"rule": "user-selected successful historical normalized-coordinate teacher"},
        "omitted_payload_keys": sorted(set(saved) - {"cfm_model", "interpolant", "min_max", "config"}),
    }


def checkpoint_readme(bundle, filename):
    files = bundle["files"]
    if bundle["source_format"] == "framework":
        manifest = json.loads(files["manifest.json"])
        data, model = manifest["dataset"], manifest["model_config"]
        preprocessing = json.loads(files["preprocessing.json"])
        seed, training = manifest["seed"], manifest["training_config"]
    else:
        metadata = json.loads(files["legacy_metadata.json"])
        data, model = metadata["data"], bundle["weights"]["config"]
        lo, hi = bundle["weights"]["min_max"]
        preprocessing = {"normalise": True, "whiten": False, "min": lo.tolist(), "max": hi.tolist()}
        seed, training = 42, bundle["weights"]["config"]
    scores = {k: v for k, v in bundle["saved_metrics"].items() if
              k in ("teacher_mean_emd", "teacher_heldout_emd", "teacher_final_emd", "teacher_heldout_w2", "teacher_train_w2")}
    protocol = bundle["saved_metrics"].get("protocol", {})
    data = {k: data[k] for k in ("name", "dim", "root", "physical_times", "model_times",
                                "observed_indices", "heldout_indices", "params", "source", "split_seed") if k in data}
    return f"""# {bundle['method']} / {bundle['dataset']}

Checkpoint: `{filename}`. Format: `{FORMAT}`. Teacher seed: `{seed}`.
Source: `{bundle['source_checkpoint']}`.
Original checkpoint SHA-256: `{bundle['source_checkpoint_sha256']}`.
Exact exported weight fingerprint: `{bundle['integrity']['weights']}`.

This is an inference/distillation export, not a training-resume checkpoint.
{"ALI-CFM includes its velocity field and generator/interpolant in this one file." if bundle['method'] == 'alicfm' else "OTP-FM includes the complete saved model state, including EMA and potential buffers."}
The archive embeds model/time/data contracts, preprocessing, configs, provenance,
and saved coupling indices when present. Dataset arrays and model source code
are not included. Logs, plots, discriminator weights and optimizer/RNG states
are omitted where unused by the frozen teacher loaders.

## Selection and saved results

```json
{json.dumps(bundle['selection'], indent=2)}
```

Some methods save only a final checkpoint; no earlier best step was invented.
{"For this OTP-FM transport dataset, W2 is the best of the four completed native recipes by their saved native W2 score." if bundle['method'] == 'otpfm' and bundle['dataset'] in ('gom', 'beijing') else "Selection is specific to this configured teacher; no best-over-training checkpoint was synthesized."}
W2 scores and W1 EMD are different metrics.

Saved evaluation (no new evaluation was run):

```json
{json.dumps({'scores': scores, 'protocol': protocol}, indent=2)}
```

## Load on another machine

From the export root (parent of `ali-cfm/` and `otp-fm/`), with PyTorch and PyYAML installed:

```bash
python unpack_teacher.py unpack {filename} --output-dir /path/to/restored-teachers
```

This writes a minimal folder compatible with the existing Flow2PC loaders and
prints the checkpoint path plus a `teacher` config fragment. It runs no model.
For this file, the restored folder is
`/path/to/restored-teachers/{bundle['method']}/{bundle['dataset']}`.
Keep the current Flow2PC repository available for architecture and adapter code.
Source-code hashes/provenance are retained and checked by the existing loaders.
Datasets remain separate: original locations below are provenance, not data
embedded in the `.pt`. They must be made available when distilling/evaluating.

## Dataset, time, preprocessing, model and training settings

Physical observation times and normalized model times are distinct. The saved
field uses velocity with respect to model time. Preserve the recorded transforms.
{"This historical tracking teacher uses normalized-coordinate CFM, its saved min/max bounds, global time in [0,1], and 115 frames. It replaces the failed recent raw-coordinate ALI-CFM tracking run." if bundle['source_format'] == 'legacy' else "The full saved dataset and preprocessing contracts remain embedded in the checkpoint."}

```json
{json.dumps({'data': data, 'preprocessing': preprocessing, 'model': model, 'training': training}, indent=2)}
```
"""


def unpack(bundle, destination, bundle_sha):
    destination = Path(destination) / bundle["method"] / bundle["dataset"]
    if destination.exists():
        raise FileExistsError(f"Preserving existing output: {destination}")
    destination.mkdir(parents=True)
    for name, data in bundle["files"].items():
        if name != "manifest.json":
            (destination / name).write_bytes(data)
    if bundle["source_format"] == "legacy":
        checkpoint = destination / "checkpoints/teacher.pt"
        checkpoint.parent.mkdir()
        torch.save(bundle["weights"], checkpoint)
        metadata = json.loads(bundle["files"]["legacy_metadata.json"])
        metadata.update(source_checkpoint_sha256=bundle["source_checkpoint_sha256"],
                        checkpoint_sha256=digest(checkpoint.read_bytes()))
        (destination / "legacy_metadata.json").write_bytes(json_bytes(metadata))
        source = {"name": bundle["method"], "format": "legacy", "checkpoint": str(checkpoint.resolve()),
                  "legacy_metadata": str((destination / "legacy_metadata.json").resolve())}
    else:
        manifest = json.loads(bundle["files"]["manifest.json"])
        relative = Path(manifest["checkpoint_relpath"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Restored checkpoint escapes its run directory")
        manifest.pop("checkpoint_sha256")
        manifest["portable_export"] = {"source_checkpoint_sha256": bundle["source_checkpoint_sha256"],
                                       "bundle_sha256": bundle_sha, "inference_only": True}
        payload = {**bundle["weights"], "manifest_identity": identity(manifest)}
        checkpoint = destination / relative
        checkpoint.parent.mkdir(parents=True)
        torch.save(payload, checkpoint)
        manifest["checkpoint_sha256"] = digest(checkpoint.read_bytes())
        (destination / "manifest.json").write_bytes(json_bytes(manifest))
        source = {"name": bundle["method"], "checkpoint": str(checkpoint.resolve())}
    # Artifact integrity only. No architecture is instantiated or executed.
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if fingerprint({k: restored[k] for k in bundle["weights"]}) != bundle["integrity"]["weights"]:
        raise ValueError("Unpacked weights differ from portable weights")
    return {"teacher": source}


def export(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing export: {args.output_dir}")
    rows = list(csv.DictReader((args.source_root / "summary.csv").open()))
    sources = []
    for row in rows:
        if row["status"] != "complete":
            continue
        method = row["method"]
        tag = row["preset"].removeprefix(method + "_")
        if method == "otpfm" and tag.startswith(("gom_", "beijing_")):
            if not tag.endswith("_w2"):
                continue
            tag = tag.removesuffix("_w2")
        sources.append((Path(row["output"]), tag))
    bundles = [framework_bundle(path, tag) for path, tag in sources]
    bundles.append(historical_bundle(args.historical_checkpoint, args.historical_metadata))
    expected = {(method, name) for method in ("alicfm", "otpfm") for name in DATASETS}
    if {(b["method"], b["dataset"]) for b in bundles} != expected or len(bundles) != 30:
        raise ValueError("Expected exactly one ALI-CFM and OTP-FM teacher for all 15 profiles")
    # Stage the complete export and rename only after all artifacts verify.
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".packing-teachers-", dir=args.output_dir.parent) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir()
        records = []
        for bundle in sorted(bundles, key=lambda b: (b["method"], b["dataset"])):
            bundle["created_utc"] = datetime.now(timezone.utc).isoformat()
            bundle["integrity"] = {"weights": fingerprint(bundle["weights"]),
                                   "files": {k: digest(v) for k, v in bundle["files"].items()}}
            validate_bundle(bundle)
            name = f"{METHOD_FOLDERS[bundle['method']]}/{bundle['method']}_{bundle['dataset']}.pt"
            path = staging / name
            path.parent.mkdir(exist_ok=True)
            torch.save(bundle, path)
            read_bundle(path)
            path.with_name(f"{path.stem}.README.md").write_text(checkpoint_readme(bundle, name))
            records.append({"file": name, "method": bundle["method"], "dataset": bundle["dataset"],
                            "sha256": digest(path.read_bytes()), "bytes": path.stat().st_size,
                            "source_checkpoint": bundle["source_checkpoint"],
                            "source_checkpoint_sha256": bundle["source_checkpoint_sha256"]})
            print(f"Packed {name}: {path.stat().st_size / 2**20:.2f} MiB", flush=True)
        shutil.copyfile(__file__, staging / "unpack_teacher.py")
        (staging / "index.json").write_bytes(json_bytes({"format": FORMAT, "models": records}))
        (staging / "README.md").write_text(
            "# Portable Flow2PC teachers\n\n30 self-contained .pt files: one ALI-CFM and one OTP-FM teacher for each of 15 profiles.\n"
            "OTP-FM GOM/Beijing retain the W2 recipe. ALI-CFM tracking is the earlier successful historical teacher.\n"
            "Each checkpoint has a matching README; essential metadata also live inside its .pt.\n"
            "ALI-CFM checkpoints/READMEs live in ali-cfm/; OTP-FM checkpoints/READMEs live in otp-fm/.\n"
            "No datasets, logs, images, optimizer states, or training-resume guarantees. Original runs are untouched.\n\n"
            "Verify files after transfer (requires PyTorch and PyYAML):\n\n```bash\npython unpack_teacher.py verify ali-cfm/*.pt otp-fm/*.pt\n```\n\n"
            "Unpack all teachers into loader-compatible folders without running models:\n\n"
            "```bash\npython unpack_teacher.py unpack ali-cfm/*.pt otp-fm/*.pt --output-dir /path/to/restored-teachers\n```\n\n"
            "Unpacking preserves weights exactly and regenerates consistent checkpoint/manifest hashes.\n"
            "It writes restored_teachers.json with teacher config fragments. Keep each restored folder together.\n"
            "The Flow2PC repository supplies model code. Original data paths are recorded in the metadata;\n"
            "datasets must be available separately for distillation/evaluation. These files do not copy the data.\n"
            "Some source methods export a final selected checkpoint rather than a best-over-training checkpoint;\n"
            "the per-teacher README records the actual selection rule. Historical tracking metrics are not\n"
            "invented for the current six-frame protocol. No new training, inference, evaluation, or tests were run.\n"
        )
        staging.rename(args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "models": len(records),
                      "checkpoint_bytes": sum(r["bytes"] for r in records)}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    command = commands.add_parser("export")
    command.add_argument("--source-root", type=Path, default=Path("/data/lthung/flow2pc/pretrained-teachers"))
    command.add_argument("--output-dir", type=Path, default=Path("/data/lthung/flow2pc/teacher-checkpoints"))
    command.add_argument("--historical-checkpoint", type=Path, default=Path("/data/pqhung_group/lthung/ali-cfm/changed_files/ali_cfm_model.pth"))
    command.add_argument("--historical-metadata", type=Path, default=ROOT / "configs/teacher_metadata/alicfm_cell_tracking.json")
    for action in ("verify", "unpack"):
        command = commands.add_parser(action)
        command.add_argument("checkpoints", type=Path, nargs="+")
        if action == "unpack":
            command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "export":
        export(args)
        return 0
    records = []
    for path in args.checkpoints:
        bundle = read_bundle(path)
        if args.action == "verify":
            print(f"Verified {path.name}: exact weights and embedded metadata")
        else:
            records.append(unpack(bundle, args.output_dir, digest(path.read_bytes())))
    if args.action == "unpack":
        (args.output_dir / "restored_teachers.json").write_bytes(json_bytes(records))
        print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
