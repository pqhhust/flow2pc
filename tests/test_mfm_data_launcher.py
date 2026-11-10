"""Source-only tests for the local MFM launcher; not run automatically."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mfm_data_launcher", ROOT / "scripts/run_mfm_data.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def overrides(command):
    return dict((command[i + 1].split("=", 1)[0], json.loads(command[i + 1].split("=", 1)[1]))
                for i, value in enumerate(command) if value == "--set")


@pytest.mark.parametrize("name", list(launcher.PROFILES))
def test_every_profile_retains_distillation_configs(name):
    args = launcher.parser().parse_args(["distill", name, "--student", "both", "--dry-run"])
    commands = launcher.commands(args, name)
    assert len(commands) == 2
    run = args.output_root / "pretrain/mfm" / name / args.run_name
    for command in commands:
        config = yaml.safe_load(Path(command[3]).read_text())
        values = overrides(command)
        assert values["teacher.checkpoint"] == str(run / "checkpoints/flow.pt")
        assert values["teacher.params.geopath_checkpoint"] == str(run / "checkpoints/geopath.pt")
        assert config["evaluation"]["enabled"] is True
        assert config["evaluation"]["metric_space"] == "dataset"
        assert config["wandb"]["project"] == "flow2pc"
        assert overrides(command)["wandb.mode"] == "online"
        assert "source_nll_weight" not in config["distillation"]


def test_preview_never_spawns_or_creates_data(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run started an operation")

    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    monkeypatch.setattr(launcher, "link_sources", forbidden)
    assert launcher.main(["prepare", "all", "--dry-run", "--data-root", str(tmp_path / "data"),
                          "--output-root", str(tmp_path / "outputs")]) == 0
    assert list(tmp_path.iterdir()) == []


def test_pretrain_launcher_generates_a_new_run_name_each_invocation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: calls.append(command))
    for _ in range(2):
        assert launcher.main(["pretrain", "cite100", "--device", "cuda:0",
                              "--output-root", str(tmp_path / "outputs")]) == 0
    first, second = [overrides(command) for command in calls]
    assert first["data.root"] == second["data.root"]
    assert first["output.dir"] != second["output.dir"]
    assert Path(first["output.dir"]).parent == Path(second["output.dir"]).parent
    assert Path(first["output.dir"]).is_dir()
    assert Path(second["output.dir"]).is_dir()


@pytest.mark.parametrize("name", launcher.NATIVE_PROFILES)
def test_native_profiles_use_original_configs_and_flow2pc_raw_data(tmp_path, name):
    args = launcher.parser().parse_args(["pretrain", name,
                                          "--run-name", "native-1", "--dry-run",
                                          "--output-root", str(tmp_path / "outputs")])
    command, = launcher.commands(args, name)
    values = overrides(command)
    config = yaml.safe_load(Path(command[3]).read_text())
    native_config = config["model"]["params"]["config_path"]
    assert native_config == f"configs/pretrain/mfm/native/mfm_{name}_source.yaml"
    settings = yaml.safe_load((ROOT / native_config).read_text())
    assert settings["dim"] == launcher.PROFILES[name]["dim"]
    assert settings["t_exclude"] == [1, 2] and settings["mfm"] is True
    if name.endswith("5"):
        assert settings["whiten"] is True and settings["velocity_metric"] == "land"
        assert settings["gammas"] == [0.125, 0.25]
    else:
        assert settings["whiten"] is False and settings["velocity_metric"] == "rbf"
        assert settings["n_centers"] == 150 and settings["metric_epochs"] == 200
    assert config["data"]["name"] == settings["data_name"]
    assert "source_root" not in values
    assert values["data.root"] == str(ROOT / "data/single_cell")
    assert values["output.dir"] == str(tmp_path / f"outputs/pretrain/mfm/{name}/native-1")
    assert "wandb.mode" not in values and "training.device" not in values
    assert not (tmp_path / "outputs").exists()


def test_launcher_has_no_backend_switch_or_automatic_native_distillation():
    with pytest.raises(SystemExit):
        launcher.parser().parse_args(["pretrain", "cite100", "--backend", "snapshots"])
    with pytest.raises(SystemExit):
        launcher.main(["both", "cite100", "--dry-run"])
    with pytest.raises(SystemExit):
        launcher.main(["pretrain", "multi50", "--dry-run"])


def test_pretrain_failure_never_starts_distill(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, "link_sources", lambda *args, **kwargs: None)

    def execute(command, **kwargs):
        calls.append(command)
        if "--config" in command:
            raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(launcher.subprocess, "run", execute)
    assert launcher.main(["pretrain", "cite100", "--device", "cuda:0",
                          "--output-root", str(tmp_path / "outputs")]) == 9
    assert len(calls) == 1
    assert all("configs/distill" not in " ".join(command) for command in calls)
