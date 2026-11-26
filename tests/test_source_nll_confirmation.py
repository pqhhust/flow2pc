"""Confirmation contracts; execute only with explicit test authorization."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_reuses_seed_zero_and_changes_only_training_seed(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_source_nll_confirmation as confirm

    entries, configs, before = [], [], {}
    for index, original in enumerate(confirm.DEFAULT_CONFIGS):
        config = confirm.sweep.load_config(original)
        output = tmp_path / "reference" / f"recipe-{index}"
        output.mkdir(parents=True)
        config["output"]["dir"] = str(output)
        resolved = output / "resolved_config.yaml"
        resolved.write_text(yaml.safe_dump(config))
        before[resolved] = resolved.read_text()
        declared = tmp_path / f"recipe-{index}.yaml"
        declared.write_text(yaml.safe_dump(config))
        configs.append(declared)
        entries.append({"config": config})
    reference = tmp_path / "reference"
    (reference / "distillation_plan.json").write_text(json.dumps({"runs": entries}))
    def inspect(run, *args):
        run.status = "complete" if run.config["seed"] == 0 else "pending"
    monkeypatch.setattr(confirm.batch, "inspect_run", inspect)
    args = confirm.parser().parse_args(["--reference-root", str(reference),
                                       "--output-root", str(tmp_path / "confirmation")])
    args.configs = configs
    runs = confirm.build_runs(args)
    assert len(runs) == len({r.output for r in runs}) == 12
    assert sum(r.config["seed"] == 0 for r in runs) == 4
    for index, entry in enumerate(entries):
        for run in runs[index * 3:index * 3 + 3]:
            baseline = confirm.batch.scientific_config(entry["config"])
            actual = confirm.batch.scientific_config(run.config)
            baseline["seed"] = run.config["seed"]
            assert actual == baseline
            if run.config["seed"] == 0:
                assert run.output == Path(entry["config"]["output"]["dir"])
            else:
                assert run.output.is_relative_to(tmp_path / "confirmation")
                assert not run.output.exists()
    assert all(path.read_text() == text for path, text in before.items())


def test_confirmation_does_not_pool_mc_draws_as_independent_training_runs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_source_nll_confirmation as confirm

    tasks = []
    for training_seed, values in ((0, (0., 2.)), (1, (10., 12.))):
        for mc_seed, value in enumerate(values):
            task = confirm.diagnostic.Task(
                f"s{training_seed}-mc{mc_seed}", "alicfm", .003, tmp_path / "config.yaml",
                tmp_path / f"s{training_seed}.pt", {"student_training_seed": training_seed,
                    "student_learning_rate": .003, "seed": mc_seed}, tmp_path / "result")
            task.status = "complete"
            task.result = {"rows": [{"heldout": True, "time_index": 1, "direct_vs_observed_w1_raw": value}],
                           "velocity_support": [], "source_validation_nll_per_dimension": -.4}
            tasks.append(task)
    args = SimpleNamespace(output_root=tmp_path)
    confirm.report(args, tasks, [], "complete")
    checkpoint = list(csv.DictReader((tmp_path / "checkpoint_summary.csv").open()))
    recipes = list(csv.DictReader((tmp_path / "recipe_summary.csv").open()))
    assert [float(r["heldout_direct_w1_mc_mean"]) for r in checkpoint] == [1., 11.]
    assert float(checkpoint[0]["heldout_direct_w1_mc_std"]) == pytest.approx(2**.5)
    assert int(recipes[0]["training_seeds_complete"]) == 2
    assert float(recipes[0]["heldout_direct_w1_training_mean"]) == 6.
    assert float(recipes[0]["heldout_direct_w1_training_std"]) == pytest.approx(50**.5)
    tasks[0].status = "failed"
    confirm.report(args, tasks, [], "failed")
    recipes = list(csv.DictReader((tmp_path / "recipe_summary.csv").open()))
    assert int(recipes[0]["training_seeds_complete"]) == 1
    assert float(recipes[0]["heldout_direct_w1_training_mean"]) == 11.


def test_diagnostic_workers_use_each_tasks_mc_seed_and_direct_only_mode(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_pc_consistency_pilot as diagnostic

    args = diagnostic.parser().parse_args(["--output-root", str(tmp_path)])
    queue = diagnostic.Queue(args, [], repo=ROOT, inventory=lambda: {},
                             inspect=lambda *a: None, report=lambda *a: None, archive=lambda *a: None)
    monkeypatch.setattr(queue, "environment", lambda device: {})
    calls = []
    def start(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=100 + len(calls))
    monkeypatch.setattr(diagnostic.subprocess, "Popen", start)
    with (tmp_path / "lock").open("w") as lock:
        for seed in (101, 202):
            task = diagnostic.Task(str(seed), "alicfm", .003, tmp_path / "config.yaml", tmp_path / "best.pt",
                {"seed": seed, "direct_only": True, "metric_samples": 1024, "samples": 4096}, tmp_path / str(seed))
            queue.launch(task, "0", {"0": {}}, lock)
    for command, seed in zip(calls, (101, 202)):
        assert command[command.index("--seed") + 1] == str(seed)
        assert "--direct-only" in command
        assert command[command.index("--metric-samples") + 1] == "1024"
    for worker in queue.active:
        worker["log"].close()
