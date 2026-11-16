"""In-memory reuse of deterministic frozen-teacher population diagnostics.

Student predictions and direct PC samples are always recomputed. Cache keys
include exact inputs and numerical settings. Changing teacher tensors invalidates
all entries. Nothing is written to disk or reused between independent runs.
"""

from collections import OrderedDict
from hashlib import sha256

import torch


def tensor_key(value):
    value = value.detach().cpu().contiguous()
    return (str(value.dtype), tuple(value.shape),
            sha256(value.numpy().tobytes()).hexdigest())


class FrozenTeacherDiagnosticCache:
    def __init__(self, max_entries=256):
        if max_entries < 1:
            raise ValueError("Teacher cache needs at least one entry")
        self.max_entries = max_entries
        self.teacher = self.signature = None
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def _bind(self, teacher):
        if teacher.training or any(p.requires_grad for p in teacher.parameters()):
            raise ValueError("Diagnostic reuse requires a frozen evaluation-mode teacher")
        tensors = (*teacher.parameters(), *teacher.buffers())
        signature = tuple((id(t), t._version, t.data_ptr(), str(t.device),
                           str(t.dtype), tuple(t.shape)) for t in tensors)
        if teacher is not self.teacher or signature != self.signature:
            self.entries.clear()
            self.teacher, self.signature = teacher, signature

    def _get(self, teacher, key, compute):
        self._bind(teacher)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        value = compute()  # Failed/nonfinite results are never inserted.
        self.entries[key] = value
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
        return value

    def rollout(self, teacher, initial, times, settings, compute):
        numerical = (
            settings.get("solver", "rk4"), int(settings.get("batch_size", 128)),
            int(settings.get("ode_substeps", 16)), settings.get("tolerance", 1e-5),
            int(settings.get("max_nfe", 20000)),
        )
        key = ("rollout", tensor_key(initial), tensor_key(times), numerical)

        def checked():
            path = compute().detach().cpu()
            if not torch.isfinite(path).all():
                raise FloatingPointError("Nonfinite cached teacher ODE samples")
            return path

        return self._get(teacher, key, checked)

    def distances(self, teacher, real, predicted, transport, compute):
        key = ("distances", tensor_key(real), tensor_key(predicted), bool(transport))
        return dict(self._get(teacher, key, compute))
