"""Optional source-density anchoring; student and teacher mathematics are unchanged."""

import math

import torch

from flow2pc.distill.sampling import velocity_mse


def source_nll_per_dimension(student, values, model_time=0.0):
    """Mean -log q(x,t)/D in saved model coordinates, without PC sampling."""
    parameter = next(student.parameters())
    values = values.to(parameter)
    if (values.ndim != 2 or not len(values) or values.shape[1] != student.dim
            or not torch.isfinite(values).all()):
        raise ValueError("Source NLL requires finite source cells with the student dimension")
    time = values.new_tensor(model_time)
    if time.ndim != 0 or not torch.isfinite(time):
        raise ValueError("Source NLL requires one finite model-normalized time")
    log_prob = student.log_prob(values, time)
    if log_prob.shape != (len(values),) or not torch.isfinite(log_prob).all():
        raise FloatingPointError("Nonfinite or invalid source PC log probabilities")
    return -log_prob.mean() / student.dim


def anchored_velocity_loss(student, teacher, batch, *, weight=0.0, source=None, model_time=0.0):
    """Return total, velocity MSE and optional per-coordinate source NLL.

    Weight zero follows the existing velocity-only path exactly: no density
    evaluation, PC sampling, source draw or additional stochastic operation.
    """
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("source_nll_weight must be finite and nonnegative")
    velocity = velocity_mse(student, teacher, batch)
    if weight == 0:
        return velocity, velocity, None
    if source is None:
        raise ValueError("Positive source_nll_weight requires source training cells")
    nll = source_nll_per_dimension(student, source, model_time)
    total = velocity + weight * nll
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite anchored distillation objective")
    return total, velocity, nll


@torch.no_grad()
def validate_source_density(student, source, model_time, batch_size):
    """Score fixed reserved source cells; never fit/tune on temporal test data."""
    if not len(source):
        raise ValueError("Source-density validation requires reserved initial-time cells")
    total = sum(float(source_nll_per_dimension(student, batch, model_time)) * len(batch)
                for batch in source.split(batch_size))
    return total / len(source)

