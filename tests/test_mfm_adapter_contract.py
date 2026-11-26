"""MFM velocity interface and input contract."""

import pytest
import torch
from torch import nn

from flow2pc.pretrain.models.mfm.adapter import MetricVelocity
from flow2pc.pretrain.models.mfm.base import LoadedTeacher


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, time, values):
        return self.linear(values) + time[:, None]


def test_mfm_velocity_preserves_network_result():
    field = Field().eval()
    teacher = LoadedTeacher(MetricVelocity(field, (2,)), {})
    values = torch.randn(4, 2)
    time = torch.linspace(0, 1, 4)
    torch.testing.assert_close(teacher.velocity(values, time), field(time, values), rtol=0, atol=0)
    torch.testing.assert_close(teacher.velocity(values, time, context=None), field(time, values), rtol=0, atol=0)
    with pytest.raises(ValueError, match="does not accept context"):
        teacher.velocity(values, time, context={})


def test_mfm_conditional_velocity_requires_and_forwards_context():
    context = {"label": 1}
    teacher = LoadedTeacher(lambda t, x, c: (t, x, c), {}, conditional=True)
    with pytest.raises(ValueError, match="requires context"):
        teacher.velocity("x", "t")
    assert teacher.velocity("x", "t", context=context) == ("t", "x", context)


@pytest.mark.parametrize("values", [torch.ones(2, 2, dtype=torch.float64),
                                   torch.empty(2, 2, device="meta")])
def test_mfm_rejects_mismatched_dtype_or_device(values):
    teacher = MetricVelocity(Field(), (2,))
    with pytest.raises(ValueError, match="device and dtype"):
        teacher(0.5, values)
