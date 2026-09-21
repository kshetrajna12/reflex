"""Kernel-path detection, with transformers' fallback wrapper rebuilt in miniature.

The real decorator resolves its implementation once, at import time, and closes over
both the callable it chose and a flag saying whether that callable is the reference.
`fake_wrapper` reproduces exactly that closure shape, so the detector is tested against
the structure it reads in production without importing a model.
"""

from __future__ import annotations

import functools
import sys
import types

import pytest

from reflex import kernels


def fake_wrapper(reference, implementation=None):
    """A stand-in for `use_kernel_func_from_hub_with_fallback`'s wrapped function."""
    implementation = reference if implementation is None else implementation
    is_new_implementation = implementation is not reference

    @functools.wraps(reference)
    def wrapped(*args, **kwargs):
        if not is_new_implementation:
            pass  # the real wrapper warns here; the flag must stay in the closure
        return implementation(*args, **kwargs)

    return wrapped


def _reference():  # pragma: no cover - never called
    pass


def _fast():  # pragma: no cover - never called
    pass


_fast.__module__ = "fla.ops.gated_delta_rule.chunk"


def test_resolved_implementation_reads_the_closure():
    assert kernels.resolved_implementation(fake_wrapper(_reference)) == (__name__, False)
    assert kernels.resolved_implementation(fake_wrapper(_reference, _fast)) == (
        "fla.ops.gated_delta_rule.chunk",
        True,
    )


def test_resolved_implementation_sees_through_an_outer_decorator():
    """The hub-kernel decorator goes on top of the fallback wrapper."""

    def hub_decorate(fn):
        @functools.wraps(fn)
        def outer(*args, **kwargs):
            return fn(*args, **kwargs)

        return outer

    assert kernels.resolved_implementation(hub_decorate(fake_wrapper(_reference, _fast))) == (
        "fla.ops.gated_delta_rule.chunk",
        True,
    )


def test_plain_functions_are_not_kernel_wrappers():
    assert kernels.resolved_implementation(_reference) is None
    assert kernels.resolved_implementation(len) is None
    assert kernels.resolved_implementation(lambda: (_fast, 1)) is None


class FakeModel:
    """A model whose class lives in a fake `modeling_x` module."""

    def __init__(self, module):
        self.__class__ = type("FakeModel", (FakeModel,), {"__module__": module.__name__})
        self.config = types.SimpleNamespace(model_type="qwen3_5", _attn_implementation="sdpa")

    def parameters(self):
        return iter(())


@pytest.fixture
def modeling(monkeypatch):
    """A fake modeling module with one fast op and one fallen-back op."""
    module = types.ModuleType("reflex_test_modeling_qwen3_5")
    module.torch_chunk_gated_delta_rule = fake_wrapper(_reference, _fast)
    module.causal_conv1d_fn = fake_wrapper(_reference)
    module.not_a_kernel = _reference
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.fixture
def report(modeling, monkeypatch):
    monkeypatch.setattr(
        kernels,
        "_version",
        lambda package: {"fla": "0.5.2", "causal_conv1d": None, "kernels": None}[package],
    )
    return kernels.kernel_report(FakeModel(modeling))


def test_kernel_functions_split_fast_from_fallback(modeling):
    found = kernels.kernel_functions(FakeModel(modeling))
    assert set(found) == {"torch_chunk_gated_delta_rule", "causal_conv1d_fn"}
    assert found["torch_chunk_gated_delta_rule"]["fast"] is True
    assert found["causal_conv1d_fn"]["fast"] is False


def test_report_names_the_fallback_and_the_environment(report):
    assert report["fallbacks"] == ["causal_conv1d_fn"]
    assert report["model_type"] == "qwen3_5"
    assert report["attn_implementation"] == "sdpa"
    assert report["packages"] == {
        "flash-linear-attention": "0.5.2",
        "causal-conv1d": None,
        "kernels": None,
    }


def test_describe_is_one_line_and_names_both_sides(report):
    line = kernels.describe(report)
    assert "\n" not in line
    assert "REFERENCE FALLBACK: causal_conv1d_fn" in line
    assert "fast kernels: torch_chunk_gated_delta_rule" in line
    assert "flash-linear-attention==0.5.2" in line


def test_require_fast_kernels_exits_and_names_the_missing_package(report):
    with pytest.raises(SystemExit) as excinfo:
        kernels.require_fast_kernels(report)
    message = str(excinfo.value)
    assert "causal_conv1d_fn" in message
    assert "install causal-conv1d" in message
    # `kernels` is optional: its absence alone is never the reason to refuse.
    assert "install kernels" not in message


def test_require_fast_kernels_passes_when_everything_is_fast(report):
    report["fallbacks"] = []
    report["functions"]["causal_conv1d_fn"]["fast"] = True
    assert kernels.require_fast_kernels(report) is None


def test_a_backbone_without_kernel_ops_is_not_a_fallback(monkeypatch):
    module = types.ModuleType("reflex_test_modeling_qwen3")
    module.eager_attention_forward = _reference
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(kernels, "_version", lambda package: None)
    report = kernels.kernel_report(FakeModel(module))
    assert report["functions"] == {}
    assert kernels.require_fast_kernels(report) is None
    assert "no kernel-backed ops" in kernels.describe(report)


def test_version_returns_none_for_a_package_that_cannot_be_imported():
    assert kernels._version("reflex_definitely_not_a_package") is None
