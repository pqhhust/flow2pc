"""Metadata contracts for the MFM integration; no training is started."""

from pathlib import Path

import pytest
import yaml

from flow2pc.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_mfm_config_dispatch_keeps_ali_schema():
    ali = load_config(ROOT / "configs/pretrain/alicfm/alicfm_cite5.yaml")
    assert ali["stage"] == "pretrain"
    assert ali["model"]["name"] == "alicfm"
    stage, mfm = load_config(ROOT / "configs/distill/mfm/mfm_arch_hclt.yaml")
    assert stage == "distill"
    assert mfm.teacher["name"] == "mfm"
    assert "source_nll_weight" not in mfm.training


def test_native_mfm_pretraining_uses_paired_source_config(tmp_path):
    source = tmp_path / "source"
    (source / "mfm/train").mkdir(parents=True)
    (source / "mfm/train/main.py").write_text("")
    data = tmp_path / "data"
    data.mkdir()
    native = tmp_path / "native.yaml"
    native.write_text(yaml.safe_dump({"mfm": True, "data_type": "arch", "seeds": [17]}))
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump({
        "stage": "pretrain",
        "model": {"name": "mfm", "params": {
            "source_root": str(source), "config_path": str(native)}},
        "data": {"name": "arch", "root": str(data)},
        "training": {}, "seed": 17,
        "output": {"dir": str(tmp_path / "output")},
    }))
    stage, resolved = load_config(config)
    assert stage == "pretrain"
    assert resolved["native_config"] == str(native)
    assert resolved["command"][1:3] == ["-m", "mfm.train.main"]
    assert resolved["input_name"] is None
    assert not (tmp_path / "output").exists()

    native.write_text(yaml.safe_dump({"mfm": False, "data_type": "arch", "seeds": [17]}))
    with pytest.raises(ValueError, match="enable mfm"):
        load_config(config)


@pytest.mark.parametrize("name,family,filename", [
    ("cite5", "cite", "op_cite_inputs_0.h5ad"),
    ("cite50", "cite", "op_cite_inputs_0.h5ad"),
    ("cite100", "cite", "op_cite_inputs_0.h5ad"),
    ("multi5", "multi", "op_train_multi_targets_0.h5ad"),
    ("multi100", "multi", "op_train_multi_targets_0.h5ad"),
])
def test_bundled_native_entry_uses_local_code_and_data(tmp_path, name, family, filename):
    from flow2pc.pretrain.models.mfm.trainer import prepare
    from flow2pc.utils.paths import MFM_LOCAL_ROOT

    config = yaml.safe_load((ROOT / f"configs/pretrain/mfm/mfm_{name}.yaml").read_text())
    data = tmp_path / "data"
    data.mkdir()
    (data / filename).touch()
    config["data"]["root"] = str(data)
    config["output"]["dir"] = str(tmp_path / "run")
    resolved = prepare(config)
    assert resolved["source_root"] == str(MFM_LOCAL_ROOT.resolve())
    assert resolved["command"][1:3] == [
        "-m", "flow2pc.pretrain.models.mfm._source.mfm.train.main",
    ]
    assert resolved["data_root"] == str(data)
    assert config["data"]["name"] == family
    assert not (tmp_path / "run").exists()


def test_native_default_output_creates_a_new_folder_per_run(tmp_path, monkeypatch):
    from flow2pc.pretrain.models.mfm import trainer

    data = tmp_path / "data"
    data.mkdir()
    (data / "op_cite_inputs_0.h5ad").touch()
    resolved = {
        "output_dir": str(tmp_path / "cite100/default"),
        "data_root": str(data), "input_name": "op_cite_inputs_0.h5ad",
        "source_root": str(tmp_path), "command": ["python", "-m", "mfm", "--working_dir", str(tmp_path / "cite100/default")],
    }
    commands = []
    monkeypatch.setattr(trainer.subprocess, "run", lambda command, **kwargs: commands.append(command))
    first = trainer.run(resolved)
    second = trainer.run(resolved)
    assert first != second
    assert first.parent == second.parent == tmp_path / "cite100"
    assert commands[0][-1] == str(first) and commands[1][-1] == str(second)
    assert not (tmp_path / "cite100/default").exists()


def test_snapshot_preparation_defaults_to_bundled_mfm_source():
    from flow2pc.pretrain.models.mfm.trainer import prepare
    from flow2pc.utils.paths import MFM_LOCAL_ROOT

    config = yaml.safe_load((ROOT / "configs/pretrain/mfm/snapshots/mfm_cite100.yaml").read_text())
    resolved = prepare(config)
    assert resolved["config"]["model"]["params"]["source_root"] == str(MFM_LOCAL_ROOT.resolve())
    assert "source_root" not in config["model"]["params"]


def _resolved_input(tmp_path, dataset):
    from flow2pc.pretrain.models.mfm.trainer import DATA_TYPES, prepare

    source = tmp_path / "source"
    (source / "mfm/train").mkdir(parents=True)
    (source / "mfm/train/main.py").touch()
    data = tmp_path / "inputs"
    data.mkdir()
    for _, filename in DATA_TYPES.values():
        if filename == "afhq":
            (data / filename).mkdir()
            (data / filename / "image.jpg").write_bytes(b"original image")
        elif filename:
            (data / filename).write_bytes(b"original input")
    data_type, _ = DATA_TYPES[dataset]
    native = tmp_path / "native.yaml"
    native.write_text(yaml.safe_dump({
        "mfm": True, "data_type": data_type, "data_name": dataset, "seeds": [42],
    }))
    return prepare({
        "stage": "pretrain", "model": {"name": "mfm", "params": {
            "source_root": str(source), "config_path": str(native)}},
        "data": {"name": dataset, "root": str(data)},
        "training": {}, "output": {"dir": str(tmp_path / "output")},
    })


@pytest.mark.parametrize("dataset", ["arch", "sphere", "cite", "multi", "eb", "lidar", "afhq"])
@pytest.mark.parametrize("legacy", [False, True])
def test_run_exposes_only_selected_input(tmp_path, monkeypatch, dataset, legacy):
    from flow2pc.pretrain.models.mfm import trainer

    resolved = _resolved_input(tmp_path, dataset)
    output = Path(resolved["output_dir"])
    root = Path(resolved["data_root"])
    assert not output.exists()  # prepare must remain read-only
    original = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    if legacy:
        output.mkdir()
        (output / "data").symlink_to(root, target_is_directory=True)
    calls = []
    monkeypatch.setattr(trainer.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    for _ in range(2):
        assert trainer.run(resolved) == output
        data = output / "data"
        filename = resolved["input_name"]
        if filename:
            assert data.is_dir() and not data.is_symlink()
            assert {p.name for p in data.iterdir()} == {filename}
            assert (data / filename).is_symlink()
            assert (data / filename).resolve() == root / filename
        else:
            assert not data.exists()
        assert original == {
            p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
        }
    assert len(calls) == 2
    assert calls[0][0] == (resolved["command"],)
    assert calls[0][1]["cwd"] == resolved["source_root"]
    assert calls[0][1]["check"] is True


@pytest.mark.parametrize("conflict", ["root_link", "root_file", "input_link", "input_file"])
def test_run_preserves_conflicting_data(tmp_path, monkeypatch, conflict):
    from flow2pc.pretrain.models.mfm import trainer

    resolved = _resolved_input(tmp_path, "cite")
    output = Path(resolved["output_dir"])
    output.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep").write_bytes(b"keep")
    path = output / "data"
    if conflict.startswith("input"):
        path.mkdir()
        path = path / resolved["input_name"]
    if conflict.endswith("link"):
        path.symlink_to(foreign, target_is_directory=True)
    else:
        path.write_bytes(b"keep")
    monkeypatch.setattr(trainer.subprocess, "run", lambda *a, **kw: pytest.fail("training started"))
    with pytest.raises(ValueError):
        trainer.run(resolved)
    assert (foreign / "keep").read_bytes() == b"keep"
    if conflict.endswith("link"):
        assert path.is_symlink() and path.resolve() == foreign
    else:
        assert path.read_bytes() == b"keep"


def test_run_preserves_existing_directory_contents(tmp_path, monkeypatch):
    from flow2pc.pretrain.models.mfm import trainer

    resolved = _resolved_input(tmp_path, "cite")
    data = Path(resolved["output_dir"]) / "data"
    data.mkdir(parents=True)
    (data / "keep").write_bytes(b"keep")
    monkeypatch.setattr(trainer.subprocess, "run", lambda *a, **kw: None)
    trainer.run(resolved)
    assert (data / "keep").read_bytes() == b"keep"
