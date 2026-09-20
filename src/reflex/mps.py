"""Apple Silicon (MPS) support for hybrid models such as Qwen3.5.

The gated delta rule's fast kernels (flash-linear-attention) are Triton, so on a Mac
transformers runs its reference PyTorch implementation. That reference solves two
unit-lower-triangular systems per layer with `torch.linalg.solve_triangular`, which on MPS
costs ~15 ms per call regardless of size and was 80% of a request. The systems are only
chunk_size x chunk_size, so we build the inverse with a handful of batched matmuls instead.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping

import torch
import torch.nn.functional as F

log = logging.getLogger("reflex.mps")

HIGH_WATERMARK = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
LOW_WATERMARK = "PYTORCH_MPS_LOW_WATERMARK_RATIO"
DEFAULT_HIGH_WATERMARK = 0.7
DEFAULT_LOW_WATERMARK = 0.6


def memory_watermarks(env: Mapping[str, str]) -> dict[str, str]:
    """The MPS allocator watermarks to set, given what the user already set in `env`.

    torch lets MPS allocate 1.7x the recommended working set by default, which is more than
    physical RAM on most Macs: a model that is too large then swaps the machine into a
    watchdog reboot instead of failing. We default the cap to 0.7 so it raises an
    out-of-memory error instead.

    The two ratios are only valid as a pair (torch needs 0 <= high <= 2, where 0 means no
    limit, and low <= high unless high is 0), so each default is chosen around a value the
    user set rather than independently: a user's low of 1.0 raises our high to 1.0 instead of
    colliding with 0.7. Returns only the variables that are not already set.
    """

    def ratio(name: str) -> float | None:
        if name not in env:
            return None
        try:
            return float(env[name])
        except ValueError:
            raise ValueError(f"{name}={env[name]!r} is not a number") from None

    user_high, user_low = ratio(HIGH_WATERMARK), ratio(LOW_WATERMARK)
    high = user_high if user_high is not None else max(DEFAULT_HIGH_WATERMARK, user_low or 0.0)
    if user_low is not None:
        low = user_low
    else:
        low = DEFAULT_LOW_WATERMARK if high == 0 else min(DEFAULT_LOW_WATERMARK, high)
    if not 0 <= high <= 2 or low < 0 or (high != 0 and low > high):
        raise ValueError(
            f"invalid MPS memory watermarks: {HIGH_WATERMARK}={high:g}, {LOW_WATERMARK}={low:g}. "
            "torch needs 0 <= high <= 2 (0 disables the limit) and low <= high."
        )
    # repr, not ":g": a default derived from the user's value has to survive the round trip
    # exactly, or 0.7000001 is written back as 0.7 and the pair is no longer valid.
    chosen = {}
    if user_high is None:
        chosen[HIGH_WATERMARK] = repr(high)
    if user_low is None:
        chosen[LOW_WATERMARK] = repr(low)
    return chosen


def is_out_of_memory(error: BaseException) -> bool:
    """MPS reports an allocation failure as a plain RuntimeError ("MPS backend out of memory
    ..."), not as torch.OutOfMemoryError, so it has to be recognised by its message."""
    return isinstance(error, RuntimeError) and "MPS backend out of memory" in str(error)


def unit_lower_inverse(system: torch.Tensor) -> torch.Tensor:
    """(I + L)^-1 for the strictly lower part L of `system` [..., n, n], n a power of two.

    Recursive block inversion: [[A, 0], [C, B]]^-1 = [[A^-1, 0], [-B^-1 C A^-1, B^-1]], merging
    diagonal blocks of size 1, 2, 4, ... so it is log2(n) rounds of batched matmuls. Every
    intermediate is the true inverse of a sub-block, which keeps it as well conditioned as
    forward substitution. (The shorter Neumann product (I - L)(I + L^2)(I + L^4)... is not:
    neighbouring keys are nearly parallel, L has entries near 1, and its powers overflow.)
    """
    n = system.shape[-1]
    inverse = system.new_ones(*system.shape[:-2], n, 1, 1)  # n diagonal blocks of size 1
    size = 1
    while size < n:
        pairs = n // (2 * size)
        # the [2*size, 2*size] diagonal blocks of `system`, then their lower-left quarter C
        blocks = system.unflatten(-1, (pairs, 2 * size)).unflatten(-3, (pairs, 2 * size))
        lower_left = blocks.diagonal(dim1=-4, dim2=-2).movedim(-1, -3)[..., size:, :size]
        a_inv, b_inv = inverse[..., 0::2, :, :], inverse[..., 1::2, :, :]
        top = torch.cat([a_inv, torch.zeros_like(a_inv)], -1)
        bottom = torch.cat([-(b_inv @ lower_left @ a_inv), b_inv], -1)
        inverse = torch.cat([top, bottom], -2)
        size *= 2
    return inverse.squeeze(-3)


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop-in for transformers' `torch_chunk_gated_delta_rule` (same maths, same signature);
    only the triangular solves are replaced. Shapes: q/k [B, T, H, Dk], v [B, T, H, Dv],
    g/beta [B, T, H]."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    query = query * query.shape[-1] ** -0.5

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    num_chunks = (sequence_length + pad_size) // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)

    cum_decay = decay.cumsum(dim=3)
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(upper, float("-inf")).exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    rhs = torch.cat([v_beta, decayed_k_beta], -1)
    if chunk_size & (chunk_size - 1) == 0:
        solved = unit_lower_inverse(ut_system) @ rhs
    else:
        solved = torch.linalg.solve_triangular(ut_system, rhs, upper=False, unitriangular=True)
    new_values, k_cumdecay = solved[..., :v_head_dim], solved[..., v_head_dim:]

    if initial_state is None:
        state = new_values.new_zeros(batch_size, num_v_heads, k_head_dim, v_head_dim)
    else:
        state = initial_state.to(new_values)
    out = torch.zeros_like(new_values)

    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = query[:, :, i] @ state + intra_chunk_attn[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    out = out.reshape(batch_size, num_v_heads, -1, v_head_dim)[:, :, :sequence_length]
    out = out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return out, (state if output_final_state else None)


def patch_delta_rule(model) -> int:
    """Point every modeling module used by `model` at the matmul-only delta rule.
    Returns the number of modules patched (0 for pure-attention models)."""
    patched = set()
    for module in model.modules():
        mod = sys.modules.get(type(module).__module__)
        if mod is not None and mod not in patched and hasattr(mod, "torch_chunk_gated_delta_rule"):
            mod.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
            patched.add(mod)
    if patched:
        log.info("mps: using matmul-only gated delta rule in %s", [m.__name__ for m in patched])
    return len(patched)
