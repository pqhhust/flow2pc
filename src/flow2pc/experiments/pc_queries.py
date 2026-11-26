"""Frozen, same-time PC queries in explicit affine scoring coordinates.

This adapter changes no student parameters or training code. Gaussian/HCLT
marginals integrate unobserved leaves to one. Conditional HCLT sampling uses
upward evidence messages followed by joint posterior ancestral sampling.
It never samples hidden latent variables independently from their marginals.
"""

from collections import deque

import numpy as np
import torch

from flow2pc.pc.gaussian_pc import TimeDependentGaussianPC
from flow2pc.pc.hclt_pc import TimeDependentHCLTPC


class DirectPC:
    def __init__(self, model, preprocessing, center, scale, chunk=256):
        if not isinstance(model, (TimeDependentGaussianPC, TimeDependentHCLTPC)):
            raise ValueError("Direct queries support Gaussian and HCLT PCs only")
        if getattr(model, "leaf_distribution", "gaussian") == "categorical":
            raise ValueError("Continuous E3 queries require Gaussian PC leaves")
        self.model, self.dim, self.chunk = model, model.dim, chunk
        self.device = next(model.parameters()).device
        self.center = self.tensor(center)
        self.scale = self.tensor(scale)
        self.raw_shift = self.tensor(preprocessing["shift"])
        self.raw_scale = self.tensor(preprocessing["scale"])
        for value in (self.center, self.scale, self.raw_shift, self.raw_scale):
            if value.shape != (self.dim,) or not torch.isfinite(value).all():
                raise ValueError("Direct PC queries require finite affine coordinate metadata")
        if (self.scale <= 0).any() or (self.raw_scale <= 0).any():
            raise ValueError("Direct PC coordinate scales must be positive")
        self.transitions = {}
        if isinstance(model, TimeDependentHCLTPC):
            with torch.no_grad():
                for node in range(self.dim):
                    if node == model.root:
                        continue
                    edge = int(model.edge_by_node[node])
                    if model.sum_layer == "dense":
                        log_weights = model.transition_logits[edge].double().log_softmax(-1)
                    else:
                        # Materialize the existing normalized Monarch operator
                        # once, without modifying its factors. Rows are parents.
                        operator = torch.eye(model.num_latents, device=self.device, dtype=torch.float64)
                        for layer in range(3):
                            if layer:
                                operator = operator.index_select(-1, model.monarch_permutation)
                            blocked = operator.reshape(model.num_latents, model.num_blocks, model.block_size)
                            weights = model.monarch_logits[edge, layer].double().softmax(-1)
                            operator = torch.einsum("noi,bni->bno", weights, blocked).flatten(1)
                        log_weights = operator.T.log()
                    self.transitions[node] = log_weights.detach()

    def tensor(self, values):
        return torch.as_tensor(values, device=self.device, dtype=torch.float64)

    @torch.no_grad()
    def at(self, time):
        t = torch.as_tensor(time, device=self.device, dtype=next(self.model.parameters()).dtype)
        if t.ndim:
            raise ValueError("A direct query needs one scalar model-normalized time")
        mean, _, log_scale, _ = self.model.leaf_parameters(t)
        if isinstance(self.model, TimeDependentHCLTPC):
            if self.model.leaf_distribution == "gaussian":
                mean, log_scale = mean[..., None], log_scale[..., None]
            offset = ((self.raw_shift - self.center) / self.scale)[:, None, None]
            factor = (self.raw_scale / self.scale)[:, None, None]
        else:
            offset = ((self.raw_shift - self.center) / self.scale)[None]
            factor = (self.raw_scale / self.scale)[None]
        mean = mean.double() * factor + offset
        log_scale = log_scale.double() + factor.log()
        if not torch.isfinite(mean).all() or not torch.isfinite(log_scale).all():
            raise FloatingPointError("Nonfinite direct PC leaf parameters")
        return PCDistribution(self, mean, log_scale)


class PCDistribution:
    """Exact circuit law (up to arithmetic), optionally conditioned at one time."""

    effective_components = None  # KDE effective component count is inapplicable.

    def __init__(self, adapter, mean, log_scale, observed=(), evidence=None):
        self.adapter, self.model = adapter, adapter.model
        self.mean, self.log_scale = mean, log_scale
        self.observed = list(observed)
        self.hidden = [i for i in range(adapter.dim) if i not in self.observed]
        self.evidence = adapter.tensor(evidence) if evidence is not None else None
        self.hclt = isinstance(self.model, TimeDependentHCLTPC)
        self.log_normalizer = None
        if self.observed:
            self.log_normalizer = self._marginal(self.evidence[None], self.observed)[0]
            if not torch.isfinite(self.log_normalizer):
                raise FloatingPointError("Nonfinite PC evidence density")
        self.posterior = None

    @torch.no_grad()
    def condition(self, observed, values):
        if (self.observed or not observed or list(observed) != sorted(set(observed))
                or any(type(i) is not int or not 0 <= i < self.adapter.dim for i in observed)
                or len(observed) >= self.adapter.dim):
            raise ValueError("Invalid direct PC observed-coordinate mask")
        values = self.adapter.tensor(values)
        if values.shape != (len(observed),) or not torch.isfinite(values).all():
            raise ValueError("PC evidence must match the observed mask")
        return PCDistribution(self.adapter, self.mean, self.log_scale, observed, values)

    def _emissions(self, values, coordinates):
        if self.hclt:
            result = self.mean.new_zeros((len(values), self.adapter.dim, self.model.num_latents))
            for j, node in enumerate(coordinates):
                z = (values[:, j, None, None] - self.mean[node]) * (-self.log_scale[node]).exp()
                component = -.5 * (z.square() + np.log(2 * np.pi)) - self.log_scale[node]
                if self.model.leaf_distribution == "gaussian-mixture":
                    weights = self.model.leaf_mixture_logits[node].double().log_softmax(-1)
                    result[:, node] = torch.logsumexp(component + weights, -1)
                else:
                    result[:, node] = component[..., 0]
            return result
        z = (values[:, None] - self.mean[:, coordinates]) * (-self.log_scale[:, coordinates]).exp()
        return (-.5 * (z.square() + np.log(2 * np.pi)) - self.log_scale[:, coordinates]).sum(-1)

    def _upward(self, emissions):
        scores = {}
        for nodes in self.model.levels:
            for node in nodes:
                score = emissions[:, node]
                for child in self.model.tree_children[node]:
                    score = score + torch.logsumexp(
                        self.adapter.transitions[child][None] + scores[child][:, None], -1)
                scores[node] = score
        root = scores[self.model.root] + self.model.root_logits.double().log_softmax(-1)
        return torch.logsumexp(root, -1), scores

    def _marginal(self, values, coordinates):
        emissions = self._emissions(values, coordinates)
        if self.hclt:
            return self._upward(emissions)[0]
        return torch.logsumexp(emissions + self.model.mixture_logits.double().log_softmax(-1), -1)

    @torch.no_grad()
    def marginal_log_prob(self, values, coordinates):
        if self.observed:
            raise ValueError("Use log_prob for an already conditioned law")
        coordinates = list(coordinates)
        if (len(set(coordinates)) != len(coordinates)
                or any(type(i) is not int or not 0 <= i < self.adapter.dim for i in coordinates)):
            raise ValueError("Invalid PC marginal coordinates")
        return self._evaluate(values, coordinates)

    def _evaluate(self, values, coordinates):
        values = self.adapter.tensor(values)
        if values.ndim != 2 or values.shape[1] != len(coordinates) or not len(values) or not torch.isfinite(values).all():
            raise ValueError("Direct PC density values have invalid shape or entries")
        result = []
        for batch in values.split(min(self.adapter.chunk, 32)):
            if self.observed:
                full = batch.new_empty((len(batch), self.adapter.dim))
                full[:, self.hidden] = batch
                full[:, self.observed] = self.evidence
                score = self._marginal(full, list(range(self.adapter.dim))) - self.log_normalizer
            else:
                score = self._marginal(batch, coordinates)
            result.append(score.cpu().numpy())
        result = np.concatenate(result)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite direct PC density")
        return result

    @torch.no_grad()
    def log_prob(self, values, chunk=None):
        return self._evaluate(values, self.hidden)

    @torch.no_grad()
    def rectangle_probability(self, lower, upper):
        """P(lower <= X_hidden <= upper | evidence), with no particle draws."""
        lower, upper = self.adapter.tensor(lower), self.adapter.tensor(upper)
        if (lower.shape != (len(self.hidden),) or upper.shape != lower.shape
                or torch.isnan(lower).any() or torch.isnan(upper).any() or (lower >= upper).any()):
            raise ValueError("Rectangle bounds must match hidden coordinates and have positive width")
        evidence = self.evidence[None] if self.observed else self.mean.new_empty((1, 0))
        emissions = self._emissions(evidence, self.observed)
        for j, node in enumerate(self.hidden):
            mean = self.mean[node] if self.hclt else self.mean[:, node]
            log_scale = self.log_scale[node] if self.hclt else self.log_scale[:, node]
            lo = (lower[j] - mean) * (-log_scale).exp()
            hi = (upper[j] - mean) * (-log_scale).exp()
            # Use the nearer tail to avoid subtracting two CDFs near one.
            a = torch.where(lo >= 0, torch.special.log_ndtr(-lo), torch.special.log_ndtr(hi))
            b = torch.where(lo >= 0, torch.special.log_ndtr(-hi), torch.special.log_ndtr(lo))
            log_mass = a + torch.log(-torch.expm1(b - a))
            if self.hclt:
                if self.model.leaf_distribution == "gaussian-mixture":
                    log_mass = torch.logsumexp(log_mass + self.model.leaf_mixture_logits[node].double().log_softmax(-1), -1)
                else:
                    log_mass = log_mass[:, 0]
                emissions[:, node] = log_mass
            else:
                emissions += log_mass[None]
        numerator = (self._upward(emissions)[0][0] if self.hclt else
                     torch.logsumexp(emissions[0] + self.model.mixture_logits.double().log_softmax(-1), -1))
        log_mass = numerator - (self.log_normalizer if self.observed else 0.)
        if torch.isnan(log_mass) or log_mass > 1e-8:
            raise FloatingPointError("Invalid PC rectangle probability")
        return float(log_mass.clamp_max(0).exp().cpu())

    @torch.no_grad()
    def sample(self, count, rng):
        if type(count) is not int or count < 1:
            raise ValueError("Direct PC sample count must be positive")
        generator = torch.Generator(device=self.adapter.device).manual_seed(int(rng.integers(0, 2**63 - 1)))
        # Posterior messages depend on evidence, never on the hidden target.
        if self.posterior is None:
            evidence = self.evidence[None] if self.observed else self.mean.new_empty((1, 0))
            emissions = self._emissions(evidence, self.observed)
            if self.hclt:
                _, self.posterior = self._upward(emissions)
            else:
                self.posterior = emissions[0] + self.model.mixture_logits.double().log_softmax(-1)
        pieces = []
        for size in range(0, count, self.adapter.chunk):
            n = min(count - size, self.adapter.chunk)
            if self.hclt:
                states = torch.empty((n, self.adapter.dim), device=self.adapter.device, dtype=torch.long)
                root = self.model.root
                root_logits = self.posterior[root][0] + self.model.root_logits.double().log_softmax(-1)
                states[:, root] = torch.multinomial(root_logits.softmax(-1), n, True, generator=generator)
                queue = deque([root])
                while queue:
                    parent = queue.popleft()
                    for child in self.model.tree_children[parent]:
                        logits = self.adapter.transitions[child][states[:, parent]] + self.posterior[child]
                        states[:, child] = torch.multinomial(logits.softmax(-1), 1, generator=generator)[:, 0]
                        queue.append(child)
                nodes = torch.as_tensor(self.hidden, device=self.adapter.device)
                mean, log_scale = self.mean[nodes, states[:, nodes]], self.log_scale[nodes, states[:, nodes]]
                if self.model.leaf_distribution == "gaussian-mixture":
                    logits = self.model.leaf_mixture_logits[nodes, states[:, nodes]].double()
                    components = torch.multinomial(logits.softmax(-1).flatten(0, 1), 1, generator=generator).reshape(n, -1, 1)
                    mean, log_scale = mean.gather(-1, components)[..., 0], log_scale.gather(-1, components)[..., 0]
                else:
                    mean, log_scale = mean[..., 0], log_scale[..., 0]
            else:
                states = torch.multinomial(self.posterior.softmax(-1), n, True, generator=generator)
                mean, log_scale = self.mean[states][:, self.hidden], self.log_scale[states][:, self.hidden]
            noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
            pieces.append((mean + log_scale.exp() * noise).cpu().numpy())
        result = np.concatenate(pieces)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite direct PC draws")
        return result


class PCDensityHDR:
    """Exact PC density ranking; only the HDR mass threshold is Monte Carlo."""

    bandwidth = 0.0  # No smoothing; also permits existing region plots.
    at_search_boundary = None

    def __init__(self, distribution, hdr_scale):
        self.distribution = distribution
        self.hdr_scale = np.asarray(hdr_scale, dtype=np.float64)

    def score(self, values):
        return self.distribution.log_prob(np.asarray(values) * self.hdr_scale) + np.log(self.hdr_scale).sum()

    def coverage(self, threshold_bank, real, levels):
        from flow2pc.experiments.e3_1.metrics import EstimatedHDR

        return EstimatedHDR.coverage(self, threshold_bank, real, levels)
