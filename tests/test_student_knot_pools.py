"""Student-only population contracts. Execute only with test authorization."""

import pytest
import torch

from flow2pc.data.student_pools import training_dataset
from flow2pc.data.synthetic.knot import Knot
from flow2pc.utils.config import load_config


def knot(tmp_path):
    return Knot(name="knot", root=tmp_path, seed=42,
                params={"size": 120, "samples_per_time": 10, "noise_std": .01,
                        "normalise": False, "whiten": False,
                        "validation_fraction": .1, "test_fraction": .1})


def test_student_knot_expansion_is_reproducible_without_changing_teacher(tmp_path):
    original = knot(tmp_path)
    contract = original.contract()
    points = [x.clone() for x in original.clouds]
    settings = {"student_data": {"name": "independent_knot", "samples_per_time": 1024, "seed": 20261126}}
    expanded, binding = training_dataset(original, settings)
    repeated, second_binding = training_dataset(original, settings)
    assert original.contract() == contract
    assert expanded.contract()["split_counts"][0] == {"val": 102, "test": 102, "train": 820}
    assert binding == second_binding and binding["dataset"] == expanded.contract()
    assert expanded.preprocessing == original.preprocessing
    assert expanded.metadata.model_times == original.metadata.model_times
    assert expanded.metadata.physical_times == original.metadata.physical_times
    for old, saved, new, again in zip(original.clouds, points, expanded.clouds, repeated.clouds):
        torch.testing.assert_close(old, saved)
        torch.testing.assert_close(new, again)
    # Middle branch retains the lifted knot; independent populations are not trajectories.
    assert expanded.clouds[60][:, 1].mean() > 1.9
    assert training_dataset(original, {}) == (original, None)


@pytest.mark.parametrize("count,seed", [(10, 1), (1024, 42), (1024, -1)])
def test_invalid_expansion_is_rejected(tmp_path, count, seed):
    with pytest.raises(ValueError, match="more points"):
        training_dataset(knot(tmp_path), {"student_data": {
            "name": "independent_knot", "samples_per_time": count, "seed": seed}})


def test_full_dimension_grid_preserves_selected_models_and_st_identities(monkeypatch, tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import run_source_nll_pilot as pilot

    args = pilot.parser().parse_args([
        "--datasets", "cite5", "cite50", "cite100", "knot",
        "--weights", "0", "0.003", "0.01", "0.03", "0.1", "--output-root", str(tmp_path / "grid")])
    runs = pilot.build_runs(args)
    assert len(runs) == len({r.output for r in runs}) == 40
    for run in runs:
        method = run.config["teacher"]["name"]
        baseline = load_config(root / f"configs/distill/{method}/{method}_{run.dataset}_hclt.yaml")
        assert run.config["teacher"] == baseline["teacher"]
        assert run.config["student"] == baseline["student"]
        assert run.config["evaluation"] == baseline["evaluation"]
        for key in ("sampler", "steps", "lr", "batch_size"):
            assert run.config["distillation"][key] == baseline["distillation"][key]
        assert run.config["distillation"]["validation_population_metrics"] is False
        assert run.config["logging"]["project"] == "flow2pc"
        assert not run.output.exists()
    # Multi-dataset support must not rename any completed ST pilot runs.
    args = pilot.parser().parse_args(["--output-root", str(tmp_path)])
    st_runs = pilot.build_runs(args)
    assert len(st_runs) == 8
    for run in st_runs:
        method = run.config["teacher"]["name"]
        baseline = load_config(root / f"configs/distill/{method}/source_anchor/{method}_st_hclt.yaml")
        baseline["distillation"]["source_nll_weight"] = run.config["distillation"]["source_nll_weight"]
        assert pilot.batch.scientific_config(run.config) == pilot.batch.scientific_config(baseline)
