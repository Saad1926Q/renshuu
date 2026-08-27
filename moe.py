from dataclasses import dataclass

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


@dataclass
class MoEOutput:
    hidden_states: Tensor
    router_logits: Tensor
    topk_indices: Tensor
    topk_weights: Tensor
    expert_counts: Tensor


class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        n_routed_experts: int,
        n_shared_experts: int,
        d_model: int,
        routed_top_k: int,
        expert_ffn_dim: int,
    ) -> None:
        super().__init__()

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.d_model = d_model
        self.routed_top_k = routed_top_k
        self.expert_ffn_dim = expert_ffn_dim

        # Router weight shape: [n_routed_experts, d_model].
        self.router = TopKRouter(n_routed_experts, d_model, routed_top_k)

        # Contains n_shared_experts modules;
        self.shared_experts = nn.ModuleList(
            [SwiGLUExpert(d_model, expert_ffn_dim) for _ in range(n_shared_experts)]
        )

        # One gate matrix per routed expert: [n_routed_experts, expert_ffn_dim, d_model].
        self.routed_gate = nn.Parameter(
            torch.randn(n_routed_experts, expert_ffn_dim, d_model)
        )

        # One up matrix per routed expert: [n_routed_experts, expert_ffn_dim, d_model].
        self.routed_up = nn.Parameter(
            torch.randn(n_routed_experts, expert_ffn_dim, d_model)
        )

        # One down matrix per routed expert: [n_routed_experts, d_model, expert_ffn_dim].
        self.routed_down = nn.Parameter(
            torch.randn(n_routed_experts, d_model, expert_ffn_dim)
        )

    def forward(self, x: Tensor) -> MoEOutput:
        # input: [batch, sequence, d_model].

        tokens = x.flatten(0, 1)  # [batch, sequence, d_model] -> [tokens, d_model]

        # logits/probs: [tokens, n_routed_experts].
        # top-k weights/indices: [tokens, routed_top_k].
        logits, router_probs, topk_weights, topk_indices = self.router(tokens)

        # [tokens, routed_top_k] -> [tokens, routed_top_k, n_routed_experts].
        assignment_mask = F.one_hot(topk_indices, num_classes=self.n_routed_experts)

        # [tokens, routed_top_k, n_routed_experts] -> [n_routed_experts, routed_top_k, tokens].
        assignment_mask = assignment_mask.permute(2, 1, 0)

        shared_output = torch.zeros_like(tokens)

        for expert in self.shared_experts:
            shared_output = shared_output + expert(tokens)

        routed_output = torch.zeros_like(tokens)

        for expert_id in range(self.n_routed_experts):
            topk_slot, token_id = torch.where(assignment_mask[expert_id])

            if token_id.numel() == 0:
                continue

            expert_tokens = tokens[token_id]

            gate_weights = self.routed_gate[expert_id]  #  [expert_ffn_dim, d_model]
            up_weights = self.routed_up[expert_id]  # [expert_ffn_dim, d_model]
            down_weights = self.routed_down[expert_id]  # [d_model, expert_ffn_dim]

            gate_output = F.linear(expert_tokens, gate_weights)
            gate_output = F.silu(gate_output)

            up_output = F.linear(expert_tokens, up_weights)

            expert_output = F.linear(
                gate_output * up_output,
                down_weights,
            )

            # [num_assignments] -> [num_assignments, 1], matching expert dtype.
            routing_weights = topk_weights[
                token_id,
                topk_slot,
            ].to(expert_output.dtype).unsqueeze(-1)

            # [num_assignments, d_model] * [num_assignments, 1] -> [num_assignments, d_model].
            weighted_output = expert_output * routing_weights

            # index_add_ mutates routed_output in place; its returned tensor is unused.
            _ = routed_output.index_add_(
                0,
                token_id,
                weighted_output,
            )

        combined_tokens = shared_output + routed_output

        expert_counts = torch.bincount(
            topk_indices.flatten(),
            minlength=self.n_routed_experts,
        )

        hidden_states = combined_tokens.reshape_as(x)

        return MoEOutput(
            hidden_states=hidden_states,
            router_logits=logits,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            expert_counts=expert_counts,
        )
