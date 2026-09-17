"""System One over pixels: typed judgments about an image, in one pass.

uv run python examples/image_triage.py path/or/url/to/image.jpg
uv run python examples/image_triage.py photo.jpg --model Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import json
import time

QUESTIONS = {
    "subject": {
        "type": "choice",
        "instructions": "What is the main subject of the photo?",
        "criteria": {
            "person": None,
            "animal": None,
            "food": None,
            "vehicle": None,
            "landscape": "outdoor scenery, nature, cityscape",
            "document": "text, screenshot, receipt",
            "product": "an object for sale, packaging",
            "other": None,
        },
    },
    "contains_text": {"type": "noul", "instructions": "Does the image contain readable text?"},
    "safe_for_work": {
        "type": "noul",
        "instructions": "Is this image appropriate to show in a workplace?",
    },
    "quality": {
        "type": "score",
        "instructions": "How good is the technical image quality?",
        "criteria": [
            "Unusable: extremely blurry, dark, or corrupted",
            "Poor: noticeable blur, noise, or bad exposure",
            "Acceptable: minor flaws",
            "Good: sharp and well exposed",
        ],
    },
    "matches_caption": {
        "type": "noul",
        "instructions": "Does the `caption` in the state accurately describe the photo?",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--caption", default="a photo")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--http", action="store_true", help="use a running reflex-serve instead")
    args = ap.parse_args()

    state = {"photo": {"type": "image", "source": args.image}, "caption": args.caption}
    if args.http:
        from reflex.client import Reflex

        c = Reflex()
        t0 = time.perf_counter()
        out = c.systemone(state, QUESTIONS)
        t1 = time.perf_counter()
        out2 = c.systemone(state, QUESTIONS)
        t2 = time.perf_counter()
    else:
        from reflex import Engine, SystemOneRequest

        eng = Engine.load(args.model)
        req = SystemOneRequest(state=state, questions=QUESTIONS)
        t0 = time.perf_counter()
        out = eng.answer(req).model_dump()
        t1 = time.perf_counter()
        out2 = eng.answer(req).model_dump()
        t2 = time.perf_counter()
    assert out2["answers"] == out["answers"]
    print(json.dumps(out["answers"], indent=2))
    print(f"usage: {out['usage']}")
    print(
        f"cold: {(t1 - t0) * 1000:.0f} ms   warm (image + state cached): {(t2 - t1) * 1000:.0f} ms"
    )


if __name__ == "__main__":
    main()
