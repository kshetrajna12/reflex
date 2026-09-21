"""Which kernels the loaded model actually runs on.

Hybrid linear-attention backbones (Qwen3.5, Qwen3-Next, Mamba) do their real work in
two places transformers does not implement itself: the gated delta rule, which wants
``flash-linear-attention``'s Triton kernels, and the short causal convolution, which
wants ``causal-conv1d``'s compiled CUDA kernel. When a package is missing transformers
silently substitutes a readable PyTorch reference, logs one warning, and keeps going.
The answers stay correct and the model gets slower -- by more than an order of magnitude
for the delta rule on an H100, which is exactly the kind of thing that is invisible
until someone benchmarks you on a pod that was built without those packages.

This module reads the resolution back out of the model that is already loaded, rather
than re-deriving it from imports::

    from reflex.kernels import kernel_report, describe, fallbacks
    log.info("kernels: %s", describe(kernel_report(model)))

`reflex-serve` prints that line at startup, and `--require-fast-kernels` turns any
reference fallback into a refusal to serve. Use it for benchmark runs, where a silent
fallback is the difference between a good number and a bad one.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import inspect
import sys
from typing import Any

# The distribution that ships each import name, where the two differ.
_DISTRIBUTION = {"fla": "flash-linear-attention", "causal_conv1d": "causal-conv1d"}

# Import names whose absence puts a hybrid backbone on a reference path. Reported even
# when the loaded model does not use them, so the line reads the same everywhere.
_KERNEL_PACKAGES = ("fla", "causal_conv1d", "kernels")


def _version(package: str) -> str | None:
    """The installed version of `package`, or None if it cannot be imported."""
    try:
        importlib.import_module(package)
    except Exception:  # noqa: BLE001 - a kernel package that fails to import for any
        return None  # reason (no GPU, wrong arch, broken build) is simply unavailable
    try:
        return md.version(_DISTRIBUTION.get(package, package))
    except md.PackageNotFoundError:  # importable but not installed as a distribution
        return "unknown"


def resolved_implementation(fn: Any) -> tuple[str, bool] | None:
    """`(module of the implementation, is it the fast one)` for a transformers kernel
    wrapper, or None if `fn` is not one.

    `transformers.integrations.hub_kernels.use_kernel_func_from_hub_with_fallback`
    resolves its implementation once, at import time, and closes over both the callable
    it picked and a flag saying whether that callable is the reference. Reading those
    two cells is the only way to know what a loaded model is running without timing it.
    """
    seen: set[int] = set()
    stack = [fn]
    while stack:
        f = stack.pop()
        if f is None or id(f) in seen:
            continue
        seen.add(id(f))
        try:
            nonlocals = inspect.getclosurevars(f).nonlocals
        except TypeError:
            continue
        if "implementation" in nonlocals and "is_new_implementation" in nonlocals:
            impl = nonlocals["implementation"]
            return getattr(impl, "__module__", "?"), bool(nonlocals["is_new_implementation"])
        # The wrapper may itself be wrapped (the hub-kernel decorator goes on top).
        stack.append(getattr(f, "__wrapped__", None))
        stack.extend(v for v in nonlocals.values() if inspect.isfunction(v))
    return None


def _modeling_module(model: Any):
    """The `modeling_<arch>` module the model class was defined in."""
    return inspect.getmodule(type(model)) or sys.modules.get(type(model).__module__)


def kernel_functions(model: Any) -> dict[str, dict[str, Any]]:
    """Every kernel-backed function in the model's modeling module, by name, with the
    implementation each one resolved to. Empty for a plain attention backbone, which
    has no such functions to begin with."""
    module = _modeling_module(model)
    found: dict[str, dict[str, Any]] = {}
    for name, value in vars(module or object()).items():
        if not callable(value):
            continue
        resolved = resolved_implementation(value)
        if resolved is None:
            continue
        where, fast = resolved
        found[name] = {"implementation": where, "fast": fast}
    return dict(sorted(found.items()))


def kernel_report(model: Any) -> dict[str, Any]:
    """Everything worth printing about how this model will run: the kernel path, the
    attention implementation, dtype and device."""
    try:
        param = next(model.parameters())
        dtype, device = str(param.dtype).removeprefix("torch."), str(param.device)
    except (StopIteration, AttributeError):  # a backend that holds no weights here
        dtype, device = "n/a", "n/a"
    config = getattr(model, "config", None)
    functions = kernel_functions(model)
    return {
        "model_type": getattr(config, "model_type", "unknown"),
        "attn_implementation": getattr(config, "_attn_implementation", "unknown"),
        "dtype": dtype,
        "device": device,
        "hub_kernels": bool(getattr(model, "_use_kernels", False)),
        "packages": {_DISTRIBUTION.get(p, p): _version(p) for p in _KERNEL_PACKAGES},
        "functions": functions,
        "fallbacks": [name for name, info in functions.items() if not info["fast"]],
    }


def fallbacks(report: dict[str, Any]) -> list[str]:
    """The functions running on a reference PyTorch implementation."""
    return list(report.get("fallbacks", ()))


def _missing_distributions(report: dict[str, Any]) -> list[str]:
    return [
        name
        for name, version in report.get("packages", {}).items()
        if version is None and name != "kernels"
    ]


def describe(report: dict[str, Any]) -> str:
    """The one-line startup summary."""
    parts = [
        report["model_type"],
        f"attn={report['attn_implementation']}",
        f"{report['dtype']} on {report['device']}",
    ]
    if report["functions"]:
        fast = [n for n, i in report["functions"].items() if i["fast"]]
        parts.append(f"fast kernels: {', '.join(fast) if fast else 'none'}")
        if report["fallbacks"]:
            parts.append(f"REFERENCE FALLBACK: {', '.join(report['fallbacks'])}")
    else:
        parts.append("no kernel-backed ops")
    installed = [f"{n}=={v}" for n, v in report["packages"].items() if v]
    parts.append(f"installed: {', '.join(installed) if installed else 'none'}")
    return " | ".join(parts)


def require_fast_kernels(report: dict[str, Any]) -> None:
    """Raise SystemExit unless every kernel-backed op resolved to a fast kernel.

    What `--require-fast-kernels` is for: a benchmark run should fail loudly on a pod
    built without the kernel packages rather than quietly post a slow number.
    """
    slow = fallbacks(report)
    if not slow:
        return
    missing = _missing_distributions(report)
    hint = (
        f"install {' and '.join(missing)} in this environment"
        if missing
        else "the packages are installed but transformers did not resolve them"
    )
    raise SystemExit(
        "--require-fast-kernels: "
        f"{', '.join(slow)} {'is' if len(slow) == 1 else 'are'} running on the reference "
        f"PyTorch implementation, which is correct but much slower. To fix: {hint}. "
        "To serve anyway, drop --require-fast-kernels.\n"
        f"  {describe(report)}"
    )
