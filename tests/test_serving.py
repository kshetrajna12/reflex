import json

from reflex.serving import STABLE_MANIFEST, engine_kwargs, load_stable


def test_repo_manifest_is_consistent():
    m = load_stable()
    assert STABLE_MANIFEST.exists()
    assert m["model"].startswith("Qwen/")
    assert set(m) >= {"adapter", "calibration", "prompt_style", "prompt_texts", "permutations"}


def test_engine_kwargs_defaults_and_overrides(tmp_path):
    p = tmp_path / "stable.json"
    p.write_text(
        json.dumps(
            {
                "model": "m",
                "adapter": "org/ad",
                "calibration": None,
                "prompt_style": "markdown",
                "prompt_texts": None,
                "permutations": 2,
            }
        )
    )
    kw = engine_kwargs(load_stable(p))
    assert kw == {
        "model_id": "m",
        "adapter_path": "org/ad",
        "calibration_path": None,
        "prior_path": None,
        "prompt_style": "markdown",
        "prompt_texts": None,
        "default_permutations": 2,
    }
    kw = engine_kwargs(load_stable(p), adapter_path="other", calibration_path=None)
    assert kw["adapter_path"] == "other" and kw["calibration_path"] is None


def test_missing_manifest_means_frozen_model(tmp_path):
    m = load_stable(tmp_path / "nope.json")
    assert m["adapter"] is None and m["calibration"] is None
