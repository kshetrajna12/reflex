"""Image state on a Qwen3-VL model: packed/cached branches must match the naive
full-prompt forward, and the answers must be sensible. Needs GPU + model download."""

import numpy as np
import pytest
import torch

from reflex.images import IMAGE_PLACEHOLDER, split_images, to_pil
from reflex.prompt import build_branches
from reflex.readout import softmax
from reflex.schema import SystemOneRequest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")

# attention-only VL (packed strategy) and hybrid linear-attention VL (batched strategy)
MODELS = ["Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3.5-0.8B"]


@pytest.fixture(scope="module")
def image_path(tmp_path_factory):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((60, 60, 260, 260), fill="red")
    d.rectangle((280, 280, 420, 420), fill="blue")
    p = tmp_path_factory.mktemp("img") / "shapes.png"
    img.save(p)
    return str(p)


@pytest.fixture(scope="module", params=MODELS)
def engine(request):
    from reflex.engine import Engine

    return Engine.load(request.param, dtype=torch.float32, max_pack_tokens=4096)


def make_request(image_path):
    return SystemOneRequest(
        state={
            "photo": {"type": "image", "source": image_path},
            "note": "A synthetic test image with simple shapes.",
        },
        questions={
            "red_shape": {
                "type": "choice",
                "instructions": "What shape is the red object in the photo?",
                "criteria": {"circle": None, "square": None, "triangle": None},
            },
            "has_blue": {"type": "noul", "instructions": "Is there a blue shape in the photo?"},
            "clutter": {
                "type": "score",
                "instructions": "How cluttered is the photo?",
                "criteria": [
                    "empty or one or two simple shapes",
                    "several objects",
                    "very busy scene",
                ],
            },
        },
    )


def naive_probs(engine, state, branch):
    st, blobs = split_images(state)
    text = engine.fmt.prefix(st) + branch.text
    images = [to_pil(b) for b in blobs] or None
    inputs = engine.processor(text=[text], images=images, return_tensors="pt")
    inputs = {k: v.to(engine.device) for k, v in inputs.items()}
    inner = engine._inner()
    inner.rope_deltas = None
    with torch.inference_mode():
        logits = engine.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    return softmax(engine.restrict(logits, branch).cpu().numpy())


def test_image_state_matches_naive(engine, image_path):
    req = make_request(image_path)
    resp = engine.answer(req)
    assert resp.usage.images == 1
    assert resp.usage.state_tokens > 100  # image tokens are in the prefix

    branches = []
    for qid, q in req.questions.items():
        branches.extend(build_branches(qid, q, engine.fmt))
    entry, hit = engine.encode_state(req.state)
    assert hit
    packed = engine._forward_branches(entry, [engine._encode(b.text) for b in branches])
    for b, row in zip(branches, packed):
        p_packed = softmax(engine.restrict(row, b).cpu().numpy())
        p_naive = naive_probs(engine, req.state, b)
        np.testing.assert_allclose(p_packed, p_naive, atol=2e-3), (b.qid, p_packed, p_naive)

    a = resp.answers
    assert a["red_shape"].choice == "circle", a["red_shape"]
    assert a["has_blue"].noul > 0.5, a["has_blue"]
    assert a["clutter"].score < 1.0, a["clutter"]


def test_text_only_state_on_vl_model(engine):
    req = SystemOneRequest(
        state="The export button crashes the settings page in Safari but works in Chrome.",
        questions={
            "browser_specific": {"type": "noul", "instructions": "Is the bug browser-specific?"}
        },
    )
    resp = engine.answer(req)
    assert resp.usage.images == 0
    br = build_branches("browser_specific", req.questions["browser_specific"], engine.fmt)[0]
    p_naive = naive_probs(engine, req.state, br)
    np.testing.assert_allclose(resp.answers["browser_specific"].noul, p_naive[0], atol=2e-3)


def test_placeholder_split():
    st, blobs = split_images(
        {"a": {"type": "image", "source": "data:image/png;base64,aGk="}, "b": 1}
    )
    assert st == {"a": IMAGE_PLACEHOLDER, "b": 1} and blobs == [b"hi"]
