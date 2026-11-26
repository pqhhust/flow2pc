"""OTP conditional queries recovered from f2c6942:repc/otpfm.py.

Only constructs teacher-side states. It does not define a student, optimizer,
boundary fit, or loss. Alpha multiplies the correction AFTER the full-strength
linear solve, exactly as in native forward_with_loss and the successful port.
"""

import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from flow2pc.distill.sampling import DistillationBatch, draw, require_frozen_teacher
from flow2pc.utils.checkpoint import sha256


def correction_alpha(manifest, explicit=None):
    """Recover the selected checkpoint's scale; never start a new curriculum."""
    selected = manifest.get("selected_checkpoint", {})
    native = manifest["model_config"]["native_config"]
    value = selected.get("sampling_alpha")
    provenance = "selected_checkpoint.sampling_alpha"
    if value is None and selected.get("rule") == "final_epoch":
        total = manifest.get("training_config", {}).get("actual_updates")
        total = total or manifest.get("sampling_contract", {}).get("actual_updates")
        step = selected.get("global_step")
        if type(total) is int and type(step) is int and 0 < step <= total:
            # Old shared-data exports did not save alpha. Their original
            # MarginalTrainer used this exact native schedule and last update.
            from flow2pc.pretrain.models.otpfm._source.training import Curriculum

            schedule = Curriculum(
                total_iterations=total, schedule="sigmoid",
                slope=2 * native.get("otp_alpha_slope", 6.0),
                midpoint=0.5 * native["otp_alpha_mean_scale"],
            )
            value, provenance = float(schedule(step - 1)), "native_final_update_curriculum"
    if not native.get("tks"):
        value, provenance = 0.0, "no_interior_potentials"
    if value is None:
        if explicit is None:
            raise ValueError("OTP checkpoint has no recorded sampling alpha; provide sampler.params.alpha from its original training record")
        value, provenance = explicit, "explicit_original_training_record"
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("OTP sampling alpha must be finite and in [0,1]")
    if explicit is not None and not math.isclose(float(explicit), float(value), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Explicit OTP alpha disagrees with the selected checkpoint")
    return float(value), provenance


class ConditionalPath:
    """Historical W2Inf LU cache, also reusable for fresh independent tuples."""

    def __init__(self, model, alpha):
        from flow2pc.pretrain.models.otpfm._source.lambdas import BoxLambda, GaussianLambda
        from flow2pc.pretrain.models.otpfm._source.potentials import W2InfPotential
        from flow2pc.pretrain.models.otpfm._source.solvers import DirectSolver

        self.model, self.alpha = model, alpha
        self.potentials = list(model.potentials.values())
        self.parameter = next(model.parameters())
        self.knots = self.parameter.new_tensor(list(model.potentials))
        self.strengths = self.parameter.new_tensor([p.strength for p in self.potentials])
        self.k = len(self.potentials)
        # Native recipes may leave solver_type=None and auto-select DirectSolver.
        if self.k and (not isinstance(model.solver, DirectSolver) or not all(isinstance(p, W2InfPotential) for p in self.potentials)):
            raise NotImplementedError("Recovered OTP conditional cache supports native direct W2Inf potentials; nonlinear KL/MMD paths need their own native solver integration")
        self.checked_endpoints = False
        self.condition_number = 1.0
        self.box_constants = None
        self.gaussian_constants = None
        if self.k and all(type(p.lambda_fn) is BoxLambda for p in self.potentials):
            # Form scalar constants in Python, just as the native lambda does,
            # then cast once. Subtracting rounded float32 knots/widths instead
            # would move support boundaries relative to the original formulas.
            lambdas = [p.lambda_fn for p in self.potentials]
            self.box_constants = tuple(
                self.parameter.new_tensor(values).reshape(self.k, 1, 1)
                for values in (
                    [fn.tk - fn.width for fn in lambdas],
                    [fn.tk + fn.width for fn in lambdas],
                    [fn.width for fn in lambdas],
                    [fn.double_integrated_at_1() for fn in lambdas],
                )
            )
        if self.k and all(type(p.lambda_fn) is GaussianLambda for p in self.potentials):
            lambdas = [p.lambda_fn for p in self.potentials]
            knots, widths, z0, at_one = (
                self.parameter.new_tensor(values).reshape(self.k, 1, 1)
                for values in (
                    [fn.tk for fn in lambdas],
                    [fn.width for fn in lambdas],
                    # Native z0 uses Python division BEFORE the dtype cast.
                    [-fn.tk / fn.width for fn in lambdas],
                    [fn.double_integrated_at_1() for fn in lambdas],
                )
            )
            sqrt_two = self.parameter.new_tensor(2.0).sqrt()
            sqrt_two_pi = self.parameter.new_tensor(2 * torch.pi).sqrt()
            phi0 = 0.5 * (1.0 + torch.erf(z0 / sqrt_two))
            pdf0 = torch.exp(-0.5 * z0 ** 2) / sqrt_two_pi
            self.gaussian_constants = (
                knots, widths, at_one, sqrt_two, sqrt_two_pi, phi0, pdf0,
            )
        if self.k:
            h = torch.stack([
                model.x_time_dep(p, self.knots[:, None]).reshape(self.k)
                for p in self.potentials
            ], 1)
            self.a = h * self.strengths[None]
            self.system = torch.eye(self.k, device=h.device, dtype=h.dtype) - self.a
            self.lu, self.pivots = torch.linalg.lu_factor(self.system)
            self.condition_number = float(torch.linalg.cond(self.system))
            if not math.isfinite(self.condition_number):
                raise FloatingPointError("Singular OTP conditional linear system")

    @torch.no_grad()
    def build_cache(self, tuples):
        tuples = tuples.to(self.parameter)
        if tuples.ndim != 3 or tuples.shape[1:] != (self.k + 2, self.model.d) or not len(tuples) or not torch.isfinite(tuples).all():
            raise ValueError("OTP queries require finite [batch, observed_times, dimension] tuples")
        n, dim = len(tuples), tuples.shape[-1]
        x0, base = tuples[:, 0], tuples[:, -1] - tuples[:, 0]
        residual = 0.0
        forces = tuples.new_empty((0, n, dim))
        # Native training disables corrections at alpha <= 1e-3.
        if self.k and self.alpha > 1e-3:
            middle = tuples[:, 1:-1].transpose(0, 1)
            bases = x0[None] + self.knots[:, None, None] * base[None]
            rhs = bases - torch.einsum("ij,jnd->ind", self.a, middle)
            z = torch.linalg.lu_solve(self.lu, self.pivots, rhs.reshape(self.k, -1)).reshape(self.k, n, dim)
            residual = float((torch.einsum("ij,jnd->ind", self.system, z) - rhs).norm() / rhs.norm().clamp_min(1e-12))
            if residual > 1e-5 or not torch.isfinite(z).all():
                raise FloatingPointError(f"OTP conditional linear solve failed: relative residual {residual}")
            forces = self.strengths[:, None, None] * (z - middle)
        cache = {"x0": x0, "base": base, "forces": forces,
                 "relative_linear_residual": residual}
        if not self.checked_endpoints:
            rows = torch.arange(min(n, 8), device=x0.device)
            for endpoint in (0.0, 1.0):
                values, _ = self.sample_cache(
                    cache, rows, x0.new_full((len(rows),), endpoint), derivative=False
                )
                expected = tuples[rows, 0 if endpoint == 0 else -1]
                if not torch.allclose(values, expected, rtol=1e-5, atol=2e-5):
                    raise FloatingPointError("OTP conditional path fails an endpoint identity")
            self.checked_endpoints = True
        return cache

    def _time_factors(self, t, derivative):
        if self.gaussian_constants is not None:
            knots, widths, at_one, sqrt_two, sqrt_two_pi, phi0, pdf0 = self.gaussian_constants
            time = t[None]
            z = (time - knots) / widths
            integrated = 0.5 * (1.0 + torch.erf(z / sqrt_two)) - phi0
            pdf = torch.exp(-0.5 * z ** 2) / sqrt_two_pi
            double_integrated = (time - knots) * integrated + widths * (pdf - pdf0)
            return (double_integrated - at_one * time,
                    integrated - at_one if derivative else None)
        if self.box_constants is None:
            h = torch.stack([self.model.x_time_dep(p, t) for p in self.potentials])
            j = (torch.stack([self.model.v_time_dep(p, t) for p in self.potentials])
                 if derivative else None)
            return h, j
        # Evaluate all native box formulas together. Unlike boolean-indexed
        # assignment guarded by mask.any(), where() never branches on a CUDA
        # scalar in Python: 118 knots no longer cause 236 per-batch waits.
        left, right, width, at_one = self.box_constants
        time = t[None]
        middle = ((time - left) ** 2) / (4.0 * width)
        tail = width + (time - right)
        double_integrated = torch.where(
            (time > left) & (time <= right), middle,
            torch.where(time > right, tail, torch.zeros_like(middle)),
        )
        h = double_integrated - at_one * time
        j = None
        if derivative:
            overlap = (torch.minimum(time, right) - left.clamp_min(0.0)).clamp_min(0.0)
            j = overlap / (2.0 * width) - at_one
        return h, j

    def sample_cache(self, cache, rows, times, *, derivative=True):
        """Conditional derivative is diagnostic; it is never the student target."""
        t = times.reshape(-1, 1).to(cache["x0"])
        rows = rows.to(device=cache["x0"].device, dtype=torch.long)
        if len(rows) != len(t) or not (torch.isfinite(t) & (t >= 0) & (t <= 1)).all():
            raise ValueError("One finite global time in [0,1] is required per tuple row")
        base = cache["base"][rows]
        values = cache["x0"][rows] + t * base
        velocity = base if derivative else None
        if len(cache["forces"]):
            h, j = self._time_factors(t, derivative)
            forces = cache["forces"][:, rows]
            values = values + self.alpha * (forces * h).sum(0)
            if derivative:
                velocity = velocity + self.alpha * (forces * j).sum(0)
        finite = torch.isfinite(values).all()
        if velocity is not None:
            finite = finite & torch.isfinite(velocity).all()
        if not finite:
            raise FloatingPointError("Nonfinite OTP conditional path")
        return values, velocity


def ot_chains(pools, observed):
    """Native squared-cost POT row-argmax chains, including duplicate mappings."""
    import ot

    rows = torch.empty((len(pools[observed[0]]), len(observed)), dtype=torch.long)
    rows[:, 0] = torch.arange(len(rows))
    for column, (left, right) in enumerate(zip(observed[:-1], observed[1:])):
        source, target = pools[left].numpy(), pools[right].numpy()
        cost = ot.dist(source, target, metric="sqeuclidean")
        plan, info = ot.emd(np.ones(len(source)) / len(source), np.ones(len(target)) / len(target),
                            cost, numItermax=10_000_000, log=True)
        if info.get("warning"):
            raise RuntimeError(f"Native OTP coupling failed: {info['warning']}")
        rows[:, column + 1] = torch.from_numpy(plan.argmax(1))[rows[:, column]]
    return rows


class OTPFMConditionalSampler:
    """Keep the checkpoint's coupling and query only observed training pools."""

    def __init__(self, teacher, dataset, settings, checkpoint):
        require_frozen_teacher(teacher)
        if settings["name"] != "otpfm_conditional":
            raise ValueError("OTP-FM requires sampler.name: otpfm_conditional; it has no ALI endpoint generator")
        params = settings.get("params", {})
        if set(params) - {"alpha", "coupling"}:
            raise ValueError("OTP sampler params support only original alpha and coupling")
        if "alpha" in params and (type(params["alpha"]) not in (int, float) or not math.isfinite(params["alpha"]) or not 0 <= params["alpha"] <= 1):
            raise ValueError("Explicit OTP alpha must be finite and in [0,1]")
        self.teacher, self.dataset = teacher, dataset
        self.observed = list(dataset.observed)
        self.required_indices = set(self.observed)
        native = teacher.manifest["model_config"]["native_config"]
        if type(native.get("ot_coupling")) is not bool:
            raise ValueError("OTP checkpoint must record its original coupling")
        self.coupling = "native_ot" if native["ot_coupling"] else "independent"
        if params.get("coupling", self.coupling) != self.coupling:
            raise ValueError("OTP query coupling disagrees with teacher training")
        if (native.get("train_times", self.observed) != self.observed
                or self.observed[0] != 0 or self.observed[-1] != len(dataset.metadata.model_times) - 1):
            raise ValueError("OTP observed tuple ordering differs from dataset metadata")
        if list(teacher.model.potentials) != native["tks"] or len(native["tks"]) != len(self.observed) - 2:
            raise ValueError("OTP potential knots and observed tuple size disagree")
        self.alpha, alpha_source = correction_alpha(teacher.manifest, params.get("alpha"))
        self.path = ConditionalPath(teacher.model, self.alpha)
        self.checkpoint_dir = Path(checkpoint).resolve().parent.parent
        self.caches = {}
        self.contract = {
            "name": "otpfm_conditional_v1", "historical_source": "f2c6942:repc/otpfm.py",
            "implementation_sha256": sha256(__file__), "coupling": self.coupling,
            "alpha": self.alpha, "alpha_source": alpha_source,
            "alpha_placement": "outside_full_strength_solve",
            "potential_times": list(native["tks"]),
            "observed_indices": self.observed,
            "model_times": list(dataset.metadata.model_times),
            "training_populations": "dataset_train_rows_only",
            "time_sampling": "uniform_global_model_time",
            "target": "frozen_EMA_v_func_ema(x,t,t)",
            "conditional_path_derivative_is_target": False,
            "condition_number": self.path.condition_number,
        }
        self.fresh_knot = teacher.manifest.get("sampling_contract", {}).get("knot_training") == "fresh independent analytic noise at all knots"
        if self.fresh_knot:
            from flow2pc.data.synthetic.knot import knot_means

            means, _ = knot_means(dataset.params["size"])
            self.knot_means = torch.as_tensor(means, dtype=torch.float32)[self.observed]
            self.contract["training_populations"] = "fresh independent analytic knot noise, matching teacher training"
            self.contract["noise_std"] = dataset.params["noise_std"]
        chain_file = self.checkpoint_dir / "ot_chain_indices.npy"
        if self.coupling == "native_ot" and chain_file.is_file():
            self.contract["saved_chain_sha256"] = sha256(chain_file)

    def _ot_cache(self, pools):
        key = tuple(id(pools[i]) for i in self.observed)
        if key not in self.caches:
            if any(not len(pools[i]) for i in self.observed):
                raise ValueError("OTP coupling requires every observed query population")
            chain_file = self.checkpoint_dir / "ot_chain_indices.npy"
            full_counts = {i: len(self.dataset.clouds[i]) for i in self.observed}
            if chain_file.is_file() and all(len(pools[i]) == full_counts[i] for i in self.observed):
                indices = torch.from_numpy(np.load(chain_file, allow_pickle=False))
                if indices.dtype != torch.int64 or indices.shape != (len(pools[self.observed[0]]), len(self.observed)):
                    raise ValueError("Saved native OTP chain has invalid shape or dtype")
                for column, index in enumerate(self.observed):
                    if (indices[:, column] < 0).any() or (indices[:, column] >= len(pools[index])).any():
                        raise ValueError("Saved native OTP chain has out-of-range rows")
                if not torch.equal(indices[:, 0], torch.arange(len(indices))):
                    raise ValueError("Saved native OTP chain does not start at its source rows")
            else:
                indices = ot_chains(pools, self.observed)
            tuples = torch.stack([pools[i][indices[:, column]] for column, i in enumerate(self.observed)], 1)
            self.caches[key] = self.path.build_cache(tuples)
        return self.caches[key]

    @torch.no_grad()
    def sample_batch(self, pools, count, rng, *, validation=False, timer=None):
        measure = timer.measure if timer is not None else lambda name: nullcontext()
        with measure("query_sampling"):
            if self.coupling == "native_ot":
                cache = self._ot_cache(pools)
                rows = torch.randint(len(cache["x0"]), (count,), generator=rng)
            else:
                if self.fresh_knot and not validation:
                    tuples = self.knot_means[None] + torch.randn(count, len(self.observed), self.dataset.dim, generator=rng) * self.dataset.params["noise_std"]
                else:
                    tuples = torch.stack([draw(pools[i], count, rng) for i in self.observed], 1)
                cache = self.path.build_cache(tuples)
                rows = torch.arange(count)
            times = torch.rand(count, generator=rng).to(cache["x0"])
            values, _ = self.path.sample_cache(cache, rows, times, derivative=False)
        # Crucially, use the learned diagonal EMA field, never the derivative
        # of an individual conditional tuple path as student supervision.
        with measure("teacher_target"):
            target = self.teacher.velocity(values, times)
        return DistillationBatch(values, times, target_velocity=target)
