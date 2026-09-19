"""Publish a trained adapter (plus its calibration.json) to the Hugging Face hub, so anyone
can serve it with `reflex-serve --adapter <repo>`.

    reflex-publish-adapter runs/lora-mix2 --repo you/reflex-qwen3.5-4b-lora [--private]

Needs a hub token with write access (`hf auth login`, or env HF_TOKEN).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def model_card(base: str, src: Path) -> str:
    cal = (
        json.load(open(src / "calibration.json"))["temperature"]
        if (src / "calibration.json").exists()
        else None
    )
    report = (src / "eval_report.txt").read_text() if (src / "eval_report.txt").exists() else ""
    return f"""---
base_model: {base}
library_name: peft
license: mit
tags: [reflex, decision-model, system-one, lora, calibration]
---

# reflex LoRA for {base}

A LoRA adapter for [reflex](https://github.com/kshetrajna12/reflex), an open System One
decision model: state + typed questions in, calibrated probabilities out, over the
TypeSafe-compatible `POST /v1/systemone`.

Trained with `reflex-calibrate` (proper scoring rule on the label-restricted logits) on
the public mix built by `reflex-data`. `calibration.json` holds the fitted temperature
and is picked up automatically.

Serve it:

    uv run reflex-serve --model {base} --adapter <this repo> --host 0.0.0.0 --port 8000

Calibration temperature: {cal}

## Held-out evaluation at training time

```
{report}
```
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir")
    ap.add_argument("--repo", required=True, help="hub repo id, e.g. you/reflex-qwen3.5-4b-lora")
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    src = Path(args.adapter_dir)
    if not (src / "adapter_config.json").exists():
        raise SystemExit(f"{src} is not an adapter dir (no adapter_config.json)")
    (src / "README.md").write_text(model_card(args.base, src))
    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(src),
        repo_id=args.repo,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "calibration.json",
            "eval_report.txt",
            "README.md",
        ],
        commit_message="reflex adapter",
    )
    print(f"published https://huggingface.co/{args.repo}")
    print(f"serve with: reflex-serve --model {args.base} --adapter {args.repo}")


if __name__ == "__main__":
    main()
