import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, expert_ffn_dim: int) -> None:
        """
        Gate: d_model -> expert_ffn_dim
        Up: d_model -> expert_ffn_dim
        Down: expert_ffn_dim -> d_model
        """
        super().__init__()

        # Gate: d_model -> expert_ffn_dim
        self.gate_proj = nn.Linear(
            in_features=d_model, out_features=expert_ffn_dim, bias=False
        )

        # Up: d_model -> expert_ffn_dim
        self.up_proj = nn.Linear(
            in_features=d_model, out_features=expert_ffn_dim, bias=False
        )

        # Down: expert_ffn_dim -> d_model
        self.down_proj = nn.Linear(
            in_features=expert_ffn_dim, out_features=d_model, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(x))

        values = self.up_proj(x)

        return self.down_proj(gated * values)


class TopKRouter(nn.Module):
    def __init__(self, n_routed_experts: int, d_model: int, routed_top_k: int) -> None:
        super().__init__()

        self.n_routed_experts = n_routed_experts
        self.d_model = d_model
        self.routed_top_k = routed_top_k

        self.weights = nn.Parameter(torch.randn((n_routed_experts, d_model)))

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Receives tokens shaped [T, d_model]

        logits = F.linear(tokens, self.weights)

        router_probs = F.softmax(logits.float(), dim=-1)

        topk_weights, topk_indices = torch.topk(
            input=router_probs, k=self.routed_top_k, dim=-1
        )

        return logits, router_probs, topk_weights, topk_indices
