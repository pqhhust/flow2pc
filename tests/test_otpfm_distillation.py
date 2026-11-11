"""Recovered OTP queries; execute only when the user authorizes tests."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from flow2pc.distill.otpfm_sampling import (
    ConditionalPath, OTPFMConditionalSampler, correction_alpha, ot_chains,
)
from flow2pc.distill.sampling import velocity_mse
from flow2pc.pretrain.models.otpfm._source.factory import create_model, create_potentials
from flow2pc.pretrain.models.otpfm._source.training import Curriculum
from flow2pc.pretrain.models.otpfm.adapter import FrozenOTPFM

ROOT = Path(__file__).resolve().parents[1]


def native_config(knots=(0.5,), *, dim=2, coupling=False, solver="direct"):
    return {
        "tks": list(knots), "potential": "w2inf", "strength": 3.0, "width": 0.2,
        "hidden_dim": 8, "num_hidden_layers": 2, "x_emb_dim": 4, "t_emb_dim": 4,
        "layernorm": False, "ema_decay": 0.99, "solver_type": solver,
        "consistency_loss": "imf", "lossfn": "mse", "ot_coupling": coupling,
        "train_times": [0, 2, 4] if knots else [0, 1],
        "otp_alpha_mean_scale": 0.5, "otp_alpha_slope": 6.0,
    }


@pytest.mark.parametrize("dim", (1, 2, 5))
@pytest.mark.parametrize("lambda_type", ("gaussian", "box"))
def test_lu_cache_matches_native_full_strength_solver_and_external_alpha(dim, lambda_type):
    cfg = native_config((0.25, 0.75), dim=dim)
    cfg["lambda_type"] = lambda_type
    model = create_model(cfg, dim, create_potentials(cfg), "cpu").eval().requires_grad_(False)
    tuples = torch.randn(8, 4, dim)
    sampler = ConditionalPath(model, 0.73)
    cache = sampler.build_cache(tuples)
    x0, base = tuples[:, 0], tuples[:, -1] - tuples[:, 0]
    with torch.no_grad():
        _, forces, _ = model.solve_X_tks(x0, base, tuples[:, 1:-1].transpose(0, 1),
                                        list(model.potentials.values()), list(model.potentials), 0.73)
    torch.testing.assert_close(cache["forces"], forces, rtol=2e-5, atol=2e-5)
    times = torch.linspace(0.1, 0.9, len(tuples))
    values, _ = sampler.sample_cache(cache, torch.arange(len(tuples)), times)
    h = torch.stack([model.x_time_dep(p, times[:, None]) for p in model.potentials.values()])
    torch.testing.assert_close(values, x0 + times[:, None] * base + 0.73 * (forces * h).sum(0), rtol=2e-5, atol=2e-5)
    for time in (0.0, 1.0):
        actual, _ = sampler.sample_cache(cache, torch.arange(len(tuples)), times.new_full(times.shape, time))
        torch.testing.assert_close(actual, tuples[:, 0 if time == 0 else -1], rtol=1e-5, atol=2e-5)


def test_auto_selected_native_direct_solver_is_supported():
    cfg = native_config(solver=None)
    model = create_model(cfg, 2, create_potentials(cfg), "cpu").eval().requires_grad_(False)
    ConditionalPath(model, 0.9).build_cache(torch.randn(4, 3, 2))


@pytest.mark.parametrize("lambda_type", ("gaussian", "box"))
def test_conditional_derivative_is_full_vector_time_derivative(lambda_type):
    cfg = native_config()
    cfg["lambda_type"] = lambda_type
    model = create_model(cfg, 2, create_potentials(cfg), "cpu").eval().requires_grad_(False)
    sampler = ConditionalPath(model, 0.8)
    cache = sampler.build_cache(torch.randn(4, 3, 2))
    rows, times = torch.arange(4), torch.linspace(0.17, 0.81, 4)
    _, derivative = sampler.sample_cache(cache, rows, times)
    _, jvp = torch.autograd.functional.jvp(
        lambda t: sampler.sample_cache(cache, rows, t)[0], times, torch.ones_like(times),
    )
    torch.testing.assert_close(derivative, jvp, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_all_118_box_factors_match_native_at_support_boundaries(dtype, monkeypatch):
    cfg = native_config(tuple(i / 119 for i in range(1, 119)))
    cfg.update(lambda_type="box", width=0.25 / 119, strength=1000.0)
    model = create_model(cfg, 2, create_potentials(cfg), "cpu").to(dtype=dtype)
    model.eval().requires_grad_(False)
    path = ConditionalPath(model, 0.9998764572799472)
    boundaries = torch.tensor([
        value for p in model.potentials.values()
        for value in (p.lambda_fn.tk - p.lambda_fn.width,
                      p.lambda_fn.tk + p.lambda_fn.width)
    ], dtype=dtype)
    times = torch.cat((
        torch.linspace(0.0, 1.0, 17, dtype=dtype), boundaries,
        torch.nextafter(boundaries, torch.full_like(boundaries, -torch.inf)),
        torch.nextafter(boundaries, torch.full_like(boundaries, torch.inf)),
    ))[:, None]
    expected_h = torch.stack([model.x_time_dep(p, times) for p in model.potentials.values()])
    expected_j = torch.stack([model.v_time_dep(p, times) for p in model.potentials.values()])
    monkeypatch.setattr(model, "x_time_dep", lambda *a: pytest.fail("per-potential position loop"))
    monkeypatch.setattr(model, "v_time_dep", lambda *a: pytest.fail("per-potential derivative loop"))
    h, j = path._time_factors(times, True)
    torch.testing.assert_close(h, expected_h, rtol=0, atol=0)
    torch.testing.assert_close(j, expected_j, rtol=0, atol=0)


def test_empty_potential_case_is_the_native_straight_conditional_path():
    cfg = native_config(())
    model = create_model(cfg, 2, create_potentials(cfg), "cpu").eval().requires_grad_(False)
    path = ConditionalPath(model, 0.0)
    tuples, times = torch.randn(4, 2, 2), torch.linspace(0.0, 1.0, 4)
    values, velocity = path.sample_cache(path.build_cache(tuples), torch.arange(4), times)
    torch.testing.assert_close(values, tuples[:, 0] + times[:, None] * (tuples[:, -1] - tuples[:, 0]))
    torch.testing.assert_close(velocity, tuples[:, -1] - tuples[:, 0])


def test_alpha_recovery_uses_last_teacher_update_not_a_student_curriculum():
    manifest = {"model_config": {"native_config": native_config()},
                "training_config": {"actual_updates": 10005},
                "selected_checkpoint": {"rule": "final_epoch", "global_step": 10005}}
    actual, source = correction_alpha(manifest)
    expected = Curriculum(10005, slope=12.0, midpoint=0.25)(10004)
    assert actual == expected
    assert source == "native_final_update_curriculum"
    with pytest.raises(ValueError, match="disagrees"):
        correction_alpha(manifest, 1.0)
    manifest["selected_checkpoint"] = {"rule": "unknown"}
    with pytest.raises(ValueError, match="no recorded sampling alpha"):
        correction_alpha(manifest)


def test_observed_only_independent_queries_target_diagonal_ema_not_path_derivative(tmp_path, monkeypatch):
    cfg = native_config()
    model = create_model(cfg, 2, create_potentials(cfg), "cpu")
    manifest = {"method": "otpfm", "model_config": {"native_config": cfg},
                "selected_checkpoint": {"sampling_alpha": 0.73}}
    teacher = FrozenOTPFM(model, manifest, {})
    dataset = SimpleNamespace(dim=2, observed=[0, 2, 4],
                              metadata=SimpleNamespace(model_times=(0.0, 0.25, 0.5, 0.75, 1.0)))
    sampler = OTPFMConditionalSampler(teacher, dataset, {"name": "otpfm_conditional"}, tmp_path / "checkpoints/last.pt")

    class Heldout:
        def __len__(self):
            raise AssertionError("Sampler accessed a held-out marginal")

    pools = [torch.randn(12, 2), Heldout(), torch.randn(12, 2), Heldout(), torch.randn(12, 2)]
    monkeypatch.setattr(teacher, "interpolate", lambda *args: pytest.fail("ALI generator fallback"))
    monkeypatch.setattr(teacher, "sample", lambda *args: pytest.fail("ODE or native finite-step query sampling"))
    monkeypatch.setattr(model, "v_time_dep", lambda *args: pytest.fail("unused path derivative"))
    before = copy.deepcopy(teacher.state_dict())
    batch = sampler.sample_batch(pools, 6, torch.Generator().manual_seed(3))
    torch.testing.assert_close(batch.target_velocity, teacher.velocity(batch.x_t, batch.model_t))
    assert not batch.x_t.requires_grad and not batch.target_velocity.requires_grad
    student = torch.nn.Linear(2, 2)
    student.velocity = lambda x, t: student(x)
    loss = velocity_mse(student, teacher, batch)
    loss.backward()
    assert student.weight.grad is not None
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert all(torch.equal(before[k], v) for k, v in teacher.state_dict().items())
    with pytest.raises(ValueError, match="coupling disagrees"):
        OTPFMConditionalSampler(teacher, dataset, {"name": "otpfm_conditional", "params": {"coupling": "native_ot"}}, tmp_path / "checkpoints/last.pt")


def test_performance_only_resume_requires_matching_science_and_known_sources():
    from flow2pc.distill.otpfm_compat import compatible_binding, compatible_implementation
    from flow2pc.utils.checkpoint import sha256

    old = "71c2e9a50a34310901bf277e7da1f3d9297e13ca83a21a812f66a8a9a3f84c14"
    current = sha256(ROOT / "src/flow2pc/distill/otpfm_sampling.py")
    saved = {"teacher_sha256": "teacher-a", "settings": {"lr": 0.001},
             "sampler": {"name": "otpfm_conditional_v1", "implementation_sha256": old,
                         "alpha": 0.73, "coupling": "independent"}}
    resumed = copy.deepcopy(saved)
    resumed["sampler"]["implementation_sha256"] = current
    assert compatible_implementation(old, current)
    assert compatible_binding(saved, resumed)
    for key, value in (("alpha", 0.8), ("coupling", "native_ot"),
                       ("implementation_sha256", "unreviewed-source")):
        changed = copy.deepcopy(resumed)
        changed["sampler"][key] = value
        assert not compatible_binding(saved, changed)
    changed = copy.deepcopy(resumed)
    changed["settings"]["lr"] = 0.002
    assert not compatible_binding(saved, changed)
    changed = copy.deepcopy(resumed)
    changed["teacher_sha256"] = "teacher-b"
    assert not compatible_binding(saved, changed)
    assert not compatible_implementation(old, "unreviewed-source")


def test_native_unequal_size_ot_chains_retain_duplicate_indices():
    pools = [torch.arange(5, dtype=torch.float32)[:, None],
             torch.tensor([[0.0], [4.0]]), torch.tensor([[0.0], [2.0], [4.0]])]
    chain = ot_chains(pools, [0, 1, 2])
    assert chain.shape == (5, 3)
    assert torch.equal(chain[:, 0], torch.arange(5))
    assert len(torch.unique(chain[:, 1])) < len(chain)
    assert (chain[:, 1] < 2).all() and (chain[:, 2] < 3).all()


def test_otp_configs_preserve_existing_student_and_training_settings():
    files = list((ROOT / "configs/distill/otpfm").glob("*.yaml"))
    assert len(files) == 17
    for file in files:
        config = yaml.safe_load(file.read_text())
        tag = file.stem.removeprefix("otpfm_").removesuffix("_hclt")
        dataset = tag.rsplit("_", 1)[0] if tag.endswith(("_w2", "_w2inf")) else tag
        existing = yaml.safe_load((ROOT / f"configs/distill/alicfm/alicfm_{dataset}_hclt.yaml").read_text())
        assert config["student"] == existing["student"]
        settings, original = dict(config["distillation"]), dict(existing["distillation"])
        assert settings.pop("sampler")["name"] == "otpfm_conditional"
        original.pop("sampler")
        original.setdefault("validation_scope", "dataset")
        assert settings == original
        assert config["teacher"]["name"] == "otpfm"
        assert config["logging"]["project"] == "flow2pc"


def test_selected_student_evaluation_restores_shared_weights_and_rejects_wrong_teacher(tmp_path):
    from flow2pc.distill.student import build_student
    from flow2pc.evaluation.entrypoint import _load_student

    params = {"num_latents": 4, "time_embedding_dim": 4, "time_feature_seed": 0}
    original = build_student([-1, 0], params)
    model = {"name": "hclt", "parents": [-1, 0], **params, "fourier_zero_center": False}
    dataset = SimpleNamespace(contract=lambda: {"name": "gom", "dim": 2})
    teacher = SimpleNamespace(manifest={"checkpoint_sha256": "teacher-fixture"})
    payload = {"binding": {"teacher_sha256": "teacher-fixture", "dataset": dataset.contract()},
               "model_config": model, "state_dict": original.state_dict(), "step": 7, "best_step": 7}
    path = tmp_path / "best.pt"
    torch.save(payload, path)
    restored, step = _load_student(path, teacher, dataset, "cpu")
    assert step == 7 and not restored.training
    assert all(torch.equal(original.state_dict()[k], value) for k, value in restored.state_dict().items())
    teacher.manifest["checkpoint_sha256"] = "another-teacher"
    with pytest.raises(ValueError, match="another teacher"):
        _load_student(path, teacher, dataset, "cpu")


def test_population_ode_uses_existing_teacher_ode_hook_without_clipping_time():
    from flow2pc.evaluation.metric_flow import _teacher_ode_field

    class Teacher:
        def velocity(self, x, t):
            if ((t < 0) | (t > 1)).any():
                raise ValueError("requires global model time in [0,1]")
            return x + 2 * t

        def ode_velocity(self, t, x):
            return x + 2 * t

    teacher = Teacher()
    values, time = torch.ones(3, 2), torch.tensor(1.125)
    with pytest.raises(ValueError, match="global model time"):
        teacher.velocity(values, time)
    field = _teacher_ode_field(teacher)
    torch.testing.assert_close(field(values, time), values + 2.25)
    time = torch.tensor(-0.125)
    torch.testing.assert_close(field(values, time), values - 0.25)


def test_population_ode_preserves_teachers_without_separate_ode_interface():
    from flow2pc.evaluation.metric_flow import _teacher_ode_field

    teacher = SimpleNamespace(velocity=lambda x, t: x + t)
    field = _teacher_ode_field(teacher)
    values, time = torch.ones(3, 2), torch.tensor(0.4)
    torch.testing.assert_close(field(values, time), teacher.velocity(values, time))


def test_shared_population_evaluator_routes_adaptive_stage_to_ode_hook(monkeypatch):
    from flow2pc.evaluation import metric_flow

    class Teacher:
        def velocity(self, x, t):
            raise AssertionError("Population rollout used the strict training query")

        def ode_velocity(self, t, x):
            assert float(t) == 1.125  # Simulated adaptive internal-stage overshoot.
            return torch.ones_like(x)

    def rollout(velocity, initial, times, settings):
        velocity(initial, times.new_tensor(1.125))
        return torch.stack([initial, initial + 1])

    monkeypatch.setattr(metric_flow, "_rollout", rollout)
    clouds = [torch.zeros(4, 2), torch.ones(4, 2)]
    data = SimpleNamespace(
        source_info={"profile": "arch"}, dim=2, normalization={},
        times=torch.tensor([0.0, 1.0]), observed_indices=[0, 1], heldout_indices=[],
        marginals=clouds, train_marginals=clouds, validation_marginals=clouds,
    )
    report = metric_flow.evaluate(None, Teacher(), data, {
        "samples": 4, "metric_space": "raw", "source_split": "all", "reference_split": "all",
    }, "cpu")
    assert report["teacher_ode_emd_mean"] == pytest.approx(0.0)
