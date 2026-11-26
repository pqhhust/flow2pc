"""Conditional Gaussian KDE, explicitly an evaluation-only smoothed law.

For fit particles z_j in training-standardized coordinates, the joint KDE is
sum_j m_j N(z; z_j, h^2 I). Conditioning gives observed-space kernel weights
and Gaussian hidden components. Every density, threshold draw and score draw
refers to this SAME normalized conditional mixture, not an unsmoothed bank.
"""

import numpy as np
from scipy.special import logsumexp

from flow2pc.experiments.e3_1.metrics import finite_matrix
from flow2pc.experiments.e3_2.config import validate_dimension


def masks_for_dimension(masks, dimension):
    validate_dimension(masks, dimension)
    result = []
    for observed in masks:
        hidden = [index for index in range(dimension) if index not in observed]
        result.append((observed, hidden))
    return result


class ConditionalKDE:
    def __init__(self, fit, observed, bandwidth):
        fit = finite_matrix(fit)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Conditional bandwidth must be finite and positive")
        self.observed, self.hidden = masks_for_dimension([observed], fit.shape[1])[0]
        self.centers, counts = np.unique(fit, axis=0, return_counts=True)
        self.log_mass = np.log(counts / len(fit))
        self.bandwidth = float(bandwidth)

    def condition(self, observed_values):
        values = np.asarray(observed_values, dtype=np.float64)
        if values.shape != (len(self.observed),) or not np.isfinite(values).all():
            raise ValueError("Conditioning values must match observed coordinates")
        differences = (self.centers[:, self.observed] - values) / self.bandwidth
        log_weights = self.log_mass - 0.5 * np.square(differences).sum(1)
        log_weights -= logsumexp(log_weights)
        return ConditionalMixture(self.centers[:, self.hidden], log_weights, self.bandwidth)


class ConditionalMixture:
    def __init__(self, centers, log_weights, bandwidth):
        self.centers, self.log_weights, self.bandwidth = centers, log_weights, bandwidth
        self.weights = np.exp(log_weights)
        self.effective_components = float(1 / np.square(self.weights).sum())

    def sample(self, count, rng):
        indices = rng.choice(len(self.centers), size=count, p=self.weights)
        return self.centers[indices] + self.bandwidth * rng.standard_normal((count, self.centers.shape[1]))

    def log_prob(self, values, chunk=256):
        values = finite_matrix(values)
        if values.shape[1] != self.centers.shape[1]:
            raise ValueError("Conditional density hidden dimension mismatch")
        # cdist avoids [draws, components, dimensions] temporary arrays.
        from scipy.spatial.distance import cdist

        normalizer = values.shape[1] * np.log(self.bandwidth * np.sqrt(2 * np.pi))
        blocks = []
        for start in range(0, len(values), chunk):
            distances = cdist(values[start:start + chunk] / self.bandwidth,
                              self.centers / self.bandwidth, metric="sqeuclidean")
            blocks.append(logsumexp(self.log_weights - .5 * distances, axis=1) - normalizer)
        result = np.concatenate(blocks)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite conditional KDE density")
        return result
