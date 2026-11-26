"""MMFM teacher query validation."""

import pytest
import torch
from torch import nn

from flow2pc.pretrain.models.mmfm.adapter import LoadedMMFM, model_velocity


def teacher(model=None):
    return LoadedMMFM(model or nn.Linear(3, 2), {"model_config": {"data_dim": 2}}, "")


def test_valid_velocity_matches_native():
    loaded = teacher()
    x, t = torch.randn(4, 2), torch.linspace(0, 1, 4)
    torch.testing.assert_close(loaded.velocity(x, t), model_velocity(loaded.model, x, t), rtol=0, atol=0)


@pytest.mark.parametrize("x,t", [
    (torch.empty(0, 2), 0.5), (torch.ones(2, 3), 0.5),
    (torch.ones(2, 2, dtype=torch.float64), 0.5),
    (torch.empty(2, 2, device="meta"), 0.5),
    (torch.tensor([[float('nan'), 0.]]), 0.5),
    (torch.ones(2, 2), float('inf')), (torch.ones(2, 2), torch.ones(2, 1)),
])
def test_invalid_queries_fail(x, t):
    with pytest.raises(ValueError):
        teacher().velocity(x, t)


def test_context_and_invalid_outputs_fail():
    with pytest.raises(ValueError, match="context"):
        teacher().velocity(torch.ones(2, 2), 0.5, context={})
    with pytest.raises(ValueError, match="invalid velocity"):
        teacher(nn.Linear(3, 1)).velocity(torch.ones(2, 2), 0.5)
    model = nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.fill_(float('nan'))
    with pytest.raises(ValueError, match="invalid velocity"):
        teacher(model).velocity(torch.ones(2, 2), 0.5)
