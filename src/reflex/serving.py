"""The recommended serving configuration, as data.

`serving/stable.json` in the repo names what reflex should be served with: the base
model, an adapter (a hub id, or null for the frozen model), a calibration file, the
prompt style and any prompt-text overrides. The git tag `stable` points at the commit
that file describes, so a deployer can track the tag and read the file instead of
hard-coding flags that go stale when a better configuration is found.

    from reflex.serving import load_stable, engine_kwargs
    engine = Engine.load(**engine_kwargs(load_stable()))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# src/reflex/serving.py -> repo root / serving / stable.json
STABLE_MANIFEST = Path(__file__).resolve().parents[2] / "serving" / "stable.json"

ENGINE_KEYS = ("adapter", "calibration", "prompt_style", "prompt_texts", "permutations")


def load_stable(path: str | Path | None = None) -> dict[str, Any]:
    """The manifest as a dict, or a frozen-model default when the file is absent
    (an installed wheel without the repo, say)."""
    p = Path(path) if path else STABLE_MANIFEST
    if not p.exists():
        return {
            "model": "Qwen/Qwen3.5-4B",
            "adapter": None,
            "calibration": None,
            "prompt_style": "markdown",
            "prompt_texts": None,
            "permutations": 1,
        }
    with open(p) as f:
        return json.load(f)


def engine_kwargs(manifest: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Keyword arguments for `Engine.load` from a manifest; explicit non-None overrides
    win (so an env var or CLI flag can still pin an experiment)."""
    kw: dict[str, Any] = {
        "model_id": manifest.get("model", "Qwen/Qwen3.5-4B"),
        "adapter_path": manifest.get("adapter"),
        "calibration_path": manifest.get("calibration"),
        "prompt_style": manifest.get("prompt_style") or "markdown",
        "prompt_texts": manifest.get("prompt_texts"),
        "default_permutations": int(manifest.get("permutations") or 1),
    }
    for k, v in overrides.items():
        if v is not None:
            kw[k] = v
    return kw
