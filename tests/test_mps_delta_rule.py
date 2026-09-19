"""The matmul-only gated delta rule must match transformers' reference. Runs on CPU."""

import pytest
import torch

from reflex.mps import chunk_gated_delta_rule, unit_lower_inverse

reference = pytest.importorskip(
    "transformers.models.qwen3_5.modeling_qwen3_5"
).torch_chunk_gated_delta_rule


def test_unit_lower_inverse():
    torch.manual_seed(0)
    # entries are products of unit keys in the real system, so keep them well below 1
    system = 0.1 * torch.randn(2, 3, 64, 64, dtype=torch.float64)
    unit = system.tril(-1) + torch.eye(64, dtype=torch.float64)
    assert torch.allclose(unit_lower_inverse(system) @ unit, torch.eye(64, dtype=torch.float64))


@pytest.mark.parametrize("seq_len", [1, 63, 64, 200])
@pytest.mark.parametrize("with_state", [False, True])
def test_matches_reference(seq_len, with_state):
    torch.manual_seed(seq_len)
    B, H, Dk, Dv = 2, 4, 16, 24
    q, k = torch.randn(B, seq_len, H, Dk), torch.randn(B, seq_len, H, Dk)
    v = torch.randn(B, seq_len, H, Dv)
    g, beta = -torch.rand(B, seq_len, H), torch.rand(B, seq_len, H)
    state = torch.randn(B, H, Dk, Dv) if with_state else None
    kw = {
        "g": g,
        "beta": beta,
        "initial_state": state,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
    }
    want_out, want_state = reference(q, k, v, **kw)
    got_out, got_state = chunk_gated_delta_rule(q, k, v, **kw)
    assert torch.allclose(got_out, want_out, atol=1e-5)
    assert torch.allclose(got_state, want_state, atol=1e-5)


def test_nearly_parallel_keys():
    """Real activations: neighbouring keys are almost parallel and beta is near 1, so the
    triangular system has entries near 1. A Neumann-series inverse overflows here."""
    torch.manual_seed(0)
    B, T, H, D = 1, 256, 4, 32
    base = torch.randn(B, 1, H, D)
    q, k = torch.randn(B, T, H, D), base + 0.05 * torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)
    g, beta = -0.01 * torch.rand(B, T, H), 1 - 0.01 * torch.rand(B, T, H)
    kw = {"g": g, "beta": beta, "output_final_state": True, "use_qk_l2norm_in_kernel": True}
    want_out, want_state = reference(q, k, v, **kw)
    got_out, got_state = chunk_gated_delta_rule(q, k, v, **kw)
    assert torch.allclose(got_out, want_out, atol=1e-4)
    assert torch.allclose(got_state, want_state, atol=1e-4)
