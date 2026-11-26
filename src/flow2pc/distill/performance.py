"""Execution diagnostics; no student, optimizer, or query-distribution changes."""

from contextlib import contextmanager
from time import perf_counter

import torch

from flow2pc.utils.experiment_protocols import dataset_tag, validation_scope


class StepTimer:
    """Synchronized wall timings on logged updates only, never every update.

    CUDA synchronization is deliberate for attribution. These are elapsed times
    under the current GPU workload, not exclusive GPU kernel execution times.
    The timer draws no random numbers and never reads or modifies model state.
    """

    def __init__(self, device, *, enabled):
        self.device, self.enabled = torch.device(device), enabled
        self.seconds = {}

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        self._sync()
        started = perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.seconds[name] = self.seconds.get(name, 0.0) + perf_counter() - started

    def metrics(self):
        return {f"timing/{name}_seconds": value for name, value in self.seconds.items()}


def population_diagnostic_indices(data, settings, evaluation):
    """Keep velocity selection intact; tracking population plots use its frames."""
    if validation_scope(data, settings) == "heldout_times":
        return list(data.heldout_indices)
    observed = list(data.observed_indices)
    if dataset_tag(data) != "cell_tracking" or "time_indices" not in evaluation:
        return observed
    chosen = list(evaluation["time_indices"])
    if (not chosen or chosen != sorted(set(chosen))
            or any(type(i) is not int or i not in observed for i in chosen)):
        raise ValueError("Tracking diagnostic frames must be unique sorted observed time indices")
    return chosen
