"""Reconstruct the exact dataset representation bound to a trained teacher."""

from pathlib import Path

from flow2pc.data.registry import datasets


def resolve_data_root(contract, data_root=None):
    """Resolve a location; prepared relocations must retain the saved identity."""
    root = contract.get("root")
    if root is None:
        root = Path(contract["source_file"]).parent.parent
    prepared = (
        contract["name"] in ("arch", "sphere", "lidar")
        and contract.get("source", {}).get("prepared_identity") is not None
    )
    if data_root is not None:
        if not prepared:
            raise ValueError("teacher.data_root relocation is supported only for prepared Arch/Sphere/LiDAR")
        root = Path(data_root).expanduser().resolve()
    if prepared:
        # Metadata and file hashes only: this also runs in the Torch-free queue.
        from flow2pc.data.mfm_snapshots import read_manifest

        manifest = read_manifest(root)
        if (manifest["profile"] != contract["name"]
                or manifest["identity"] != contract["source_version"]
                or manifest["identity"] != contract["source"]["prepared_identity"]
                or manifest["arrays"] != contract["source"]["arrays"]):
            raise ValueError("Prepared dataset identity differs from the teacher checkpoint")
    return Path(root)


def restore_dataset(contract, preprocessing, *, data_root=None):
    root = resolve_data_root(contract, data_root)
    dataset = datasets.get(contract["name"])(
        name=contract["name"],
        root=Path(root),
        seed=contract["split_seed"],
        params=contract["params"],
        preprocessing=preprocessing,
    )
    expected = contract if data_root is None else {**contract, "root": str(root)}
    if dataset.contract() != expected:
        raise ValueError(
            "Dataset source, coordinates, time map, split or saved preprocessing changed"
        )
    return dataset
