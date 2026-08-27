import pytest
import torch
from torch import nn

from moe import SwiGLUExpert


def test_projection_specifications():
    d_model = 8
    expert_ffn_dim = 4

    expert = SwiGLUExpert(d_model, expert_ffn_dim)

    assert isinstance(expert, nn.Module)

    assert expert.gate_proj.in_features == d_model
    assert expert.gate_proj.out_features == expert_ffn_dim

    assert expert.up_proj.in_features == d_model
    assert expert.up_proj.out_features == expert_ffn_dim

    assert expert.down_proj.in_features == expert_ffn_dim
    assert expert.down_proj.out_features == d_model

    assert expert.gate_proj.bias is None
    assert expert.up_proj.bias is None
    assert expert.down_proj.bias is None

    assert len(list(expert.parameters())) == 3


@pytest.mark.parametrize(
    "input_shape",
    [
        (8, 8),
        (2, 3, 8),
        (2, 4, 5, 8),
    ],
)
def test_arbitrary_leading_dimensions(input_shape):
    expert = SwiGLUExpert(d_model=8, expert_ffn_dim=4)
    x = torch.randn(*input_shape)

    with torch.no_grad():
        output = expert(x)

    assert output.shape == x.shape


def test_non_contiguous_input():
    expert = SwiGLUExpert(d_model=8, expert_ffn_dim=4)
    x = torch.randn(2, 3, 8)
    non_contiguous_x = x.transpose(0, 1)

    assert not non_contiguous_x.is_contiguous()

    with torch.no_grad():
        output = expert(non_contiguous_x)
        contiguous_output = expert(non_contiguous_x.contiguous())

    torch.testing.assert_close(output, contiguous_output)


def test_gradient_check():
    expert = SwiGLUExpert(d_model=3, expert_ffn_dim=4).double()
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(expert, (x,))
