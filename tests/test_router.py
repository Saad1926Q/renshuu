import torch

from moe import TopKRouter

T = 5  # number of tokens
D = 8  # model dimension
E = 4  # number of routed experts
K = 2  # experts selected per token


def test_router_output_shapes():
    router = TopKRouter(
        n_routed_experts=E,
        d_model=D,
        routed_top_k=K,
    )
    tokens = torch.randn(T, D)

    router_logits, router_probs, topk_weights, topk_indices = router(tokens)

    assert router_logits.shape == (T, E)
    assert router_probs.shape == (T, E)
    assert topk_weights.shape == (T, K)
    assert topk_indices.shape == (T, K)


def test_router_probability_rows_sum_to_one():
    router = TopKRouter(E, D, K)
    tokens = torch.randn(T, D)

    _, router_probs, _, _ = router(tokens)

    torch.testing.assert_close(
        router_probs.sum(dim=-1),
        torch.ones(T),
        rtol=0.0,
        atol=1e-6,
    )


def test_uniform_router():
    router = TopKRouter(E, D, K)
    tokens = torch.randn(T, D)

    with torch.no_grad():
        router.weights.zero_()

    _, router_probs, topk_weights, topk_indices = router(tokens)

    torch.testing.assert_close(
        router_probs,
        torch.full((T, E), 1 / E),
    )

    torch.testing.assert_close(
        topk_weights,
        torch.full((T, K), 1 / E),
    )

    assert ((topk_indices >= 0) & (topk_indices < E)).all()
