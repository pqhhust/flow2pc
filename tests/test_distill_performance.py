"""Performance-only regressions; run only with explicit test authorization."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flow2pc.distill.performance import StepTimer, population_diagnostic_indices
from flow2pc.evaluation import metric_flow
from flow2pc.evaluation.teacher_cache import FrozenTeacherDiagnosticCache


class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.speed = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.calls = 0
        self.eval()

    def velocity(self, x, t):
        self.calls += 1
        return self.speed.expand_as(x)


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.speed = torch.nn.Parameter(torch.tensor(.5))

    def velocity(self, x, t):
        return self.speed.expand_as(x)

    def sample(self, count, time):
        return torch.randn(count, 2) + self.speed * time


def test_teacher_reuse_matches_uncached_metrics_and_recomputes_student(monkeypatch):
    monkeypatch.setattr(metric_flow, "emd", lambda real, pred, **kw:
                        float((real.mean(0) - pred.mean(0)).norm()))
    clouds = [torch.zeros(8, 2), torch.ones(8, 2)]
    data = SimpleNamespace(marginals=clouds, train_marginals=clouds,
                           validation_marginals=clouds, observed_indices=[0, 1],
                           heldout_indices=[], times=torch.tensor([0., 1.]),
                           source_info={}, normalization={}, dim=2)
    settings = {"samples": 4, "ode_substeps": 1}
    teacher, student = Teacher(), Student()
    cache = FrozenTeacherDiagnosticCache()
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    first = metric_flow.evaluate(student, teacher, data, settings, "cpu", teacher_cache=cache)
    calls = teacher.calls
    with torch.no_grad():
        student.speed.add_(.2)
    second = metric_flow.evaluate(student, teacher, data, settings, "cpu", teacher_cache=cache)
    assert teacher.calls == calls
    assert second["teacher_ode_emd_mean"] == first["teacher_ode_emd_mean"]
    assert second["student_ode_emd_mean"] != first["student_ode_emd_mean"]
    uncached = metric_flow.evaluate(student, teacher, data, settings, "cpu")
    assert second == uncached
    assert teacher.calls > calls
    assert cache.hits > 0
    assert student.training
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[2:] == np.random.get_state()[2:]
    assert python_state == random.getstate()


def test_teacher_cache_invalidates_on_inputs_numerics_weights_and_buffers():
    teacher, cache = Teacher(), FrozenTeacherDiagnosticCache()
    teacher.register_buffer("offset", torch.tensor(0.))
    initial, times = torch.zeros(2, 2), torch.tensor([0., 1.])
    calls = []

    def compute():
        calls.append(1)
        return torch.stack((initial, initial + teacher.speed + teacher.offset))

    settings = {"solver": "dopri5", "tolerance": 1e-5}
    cache.rollout(teacher, initial, times, settings, compute)
    cache.rollout(teacher, initial, times, settings, compute)
    assert len(calls) == 1
    cache.rollout(teacher, initial + 1, times, settings, compute)
    cache.rollout(teacher, initial, times, {**settings, "tolerance": 1e-6}, compute)
    cache.rollout(teacher, initial, times / 2, settings, compute)
    assert len(calls) == 4
    with torch.no_grad():
        teacher.speed.add_(1)
    updated = cache.rollout(teacher, initial, times, settings, compute)
    assert len(calls) == 5 and updated[-1, 0, 0] == 2
    teacher.offset.add_(1)
    updated = cache.rollout(teacher, initial, times, settings, compute)
    assert len(calls) == 6 and updated[-1, 0, 0] == 3
    teacher.train()
    with pytest.raises(ValueError, match="frozen"):
        cache.rollout(teacher, initial, times, settings, compute)


def test_preceding_observed_teacher_rollouts_are_reused_with_identical_scores(monkeypatch):
    monkeypatch.setattr(metric_flow, "emd", lambda real, pred, **kw:
                        float((real.mean(0) - pred.mean(0)).norm()))
    clouds = [torch.zeros(8, 2), torch.full((8, 2), .5), torch.ones(8, 2)]
    data = SimpleNamespace(marginals=clouds, train_marginals=clouds,
                           validation_marginals=clouds, test_marginals=clouds,
                           observed_indices=[0, 2], heldout_indices=[1],
                           times=torch.tensor([0., .5, 1.]),
                           source_info={"profile": "cite50"}, normalization={}, dim=2)
    teacher, student, cache = Teacher(), Student(), FrozenTeacherDiagnosticCache()
    settings = {"samples": 4, "ode_substeps": 1, "time_indices": [1]}
    expected = metric_flow.evaluate(student, teacher, data, settings, "cpu")
    first = metric_flow.evaluate(student, teacher, data, settings, "cpu", teacher_cache=cache)
    calls = teacher.calls
    second = metric_flow.evaluate(student, teacher, data, settings, "cpu", teacher_cache=cache)
    assert first == second == expected
    assert teacher.calls == calls
    assert second["by_time"]["1"]["source_index"] == 0
    assert cache.hits >= 3  # Full path, local benchmark path, teacher EMD.


def test_failed_teacher_rollout_is_not_cached():
    teacher, cache = Teacher(), FrozenTeacherDiagnosticCache()
    initial, times = torch.zeros(2, 2), torch.tensor([0., 1.])
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        cache.rollout(teacher, initial, times, {}, lambda: initial * float("nan"))
    actual = cache.rollout(teacher, initial, times, {}, lambda: initial + 1)
    torch.testing.assert_close(actual, initial + 1)
    assert cache.hits == 0 and cache.misses == 2


def test_tracking_population_frames_change_neither_velocity_scope_nor_other_datasets():
    tracking = SimpleNamespace(metadata=SimpleNamespace(name="cell_tracking"),
                               observed_indices=list(range(115)), heldout_indices=[])
    frames = [0, 24, 49, 74, 99, 114]
    assert population_diagnostic_indices(tracking, {}, {"time_indices": frames}) == frames
    assert tracking.observed_indices == list(range(115))
    gom = SimpleNamespace(metadata=SimpleNamespace(name="gom"),
                          observed_indices=[0, 2, 4], heldout_indices=[1, 3])
    assert population_diagnostic_indices(gom, {}, {"time_indices": [0, 4]}) == [1, 3]
    knot = SimpleNamespace(metadata=SimpleNamespace(name="knot"),
                           observed_indices=list(range(120)), heldout_indices=[])
    assert population_diagnostic_indices(knot, {}, {"time_indices": [0, 119]}) == list(range(120))
    with pytest.raises(ValueError, match="Tracking"):
        population_diagnostic_indices(tracking, {}, {"time_indices": [0, 115]})


def test_cpu_timing_does_not_draw_random_numbers_or_sync_cuda(monkeypatch):
    from flow2pc.distill import performance

    clock = iter((10., 10.5))
    monkeypatch.setattr(performance, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: pytest.fail("CPU timer touched CUDA"))
    state = torch.get_rng_state()
    timer = StepTimer("cpu", enabled=True)
    with timer.measure("query_sampling"):
        pass
    assert timer.metrics() == {"timing/query_sampling_seconds": .5}
    assert torch.equal(state, torch.get_rng_state())
    disabled = StepTimer("cuda:0", enabled=False)
    with disabled.measure("unused"):
        pass
    assert disabled.metrics() == {}
