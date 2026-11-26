"""Direct-query parity, not training/evaluation of any saved research model."""

import pytest
import torch

from flow2pc.evaluation.pc_query_consistency import query_consistency
from flow2pc.pc.hclt_pc import TimeDependentHCLTPC


@pytest.mark.parametrize("dimension,sum_layer", [(2, "dense"), (5, "dense"), (5, "monarch"), (50, "dense"), (100, "dense")])
def test_direct_queries_match_native_full_dimension_law(dimension, sum_layer):
    with torch.random.fork_rng():
        torch.manual_seed(234)
        model = TimeDependentHCLTPC([-1] + list(range(dimension - 1)), num_latents=4,
                                    time_order=2, sum_layer=sum_layer,
                                    fourier_zero_center=False).double().eval()
        with torch.no_grad():
            model.mean_coefficients.normal_(0, .4)
            model.log_scale_coefficients.normal_(0, .1)
            model.root_logits.normal_(0, .6)
            if model.transition_logits is not None:
                model.transition_logits.normal_(0, .6)
            else:
                model.monarch_logits.normal_(0, .6)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        time = torch.tensor(.7, dtype=torch.float64)
        states = model.sample(64, time, torch.Generator().manual_seed(404))
        preprocessing = {"shift": [2. + i / 10 for i in range(dimension)],
                         "scale": [1.5 + i / 20 for i in range(dimension)]}
        result = query_consistency(model, states, .7, preprocessing, seed=101, batch_size=16)
        assert result["passed"]
        assert result["full_state_dimension"] == dimension
        assert result["joint_density"]["max_abs_log_density_error"] < 1e-8
        assert all(r["density"]["passed"] and r["sampling"]["passed"] for r in result["conditionals"])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        assert all(p.grad is None for p in model.parameters())
