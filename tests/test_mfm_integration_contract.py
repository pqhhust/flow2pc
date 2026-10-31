"""Metadata contracts for the MFM integration; no training is started."""

from pathlib import Path

import pytest
import yaml

from flow2pc.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_mfm_config_dispatch_keeps_ali_schema():
    ali = load_config(ROOT / "configs/pretrain/ali_cfm/alicfm_cite5.yaml")
    assert ali["stage"] == "pretrain"
    assert ali["model"]["name"] == "alicfm"
    stage, mfm = load_config(ROOT / "configs/distill/mfm/arch/hclt.yaml")
    assert stage == "distill"
    assert mfm.teacher["name"] == "metric_flow"
    assert mfm.training["source_nll_weight"] == 1.0


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
    assert not (tmp_path / "output").exists()

    native.write_text(yaml.safe_dump({"mfm": False, "data_type": "arch", "seeds": [17]}))
    with pytest.raises(ValueError, match="enable mfm"):
        load_config(config)
