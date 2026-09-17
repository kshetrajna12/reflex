"""Images inside `state`.

Anywhere in the state JSON an object of the form

    {"type": "image", "source": "<file path | http(s) URL | data:image/...;base64,...>"}

is replaced by the model's image placeholder in the rendered text, and the decoded
image is handed to the vision encoder. Images are encoded once with the rest of the
state prefix, so they are cached and shared across all question branches exactly like
text. Questions stay text-only. The whole thing is "System One for pixels": the model
looks at the picture and answers typed questions about it.
"""

from __future__ import annotations

import base64
import hashlib
import io
from typing import Any

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"  # Qwen-VL convention


def is_image_ref(x: Any) -> bool:
    return isinstance(x, dict) and x.get("type") == "image" and isinstance(x.get("source"), str)


def has_images(state: Any) -> bool:
    if is_image_ref(state):
        return True
    if isinstance(state, dict):
        return any(has_images(v) for v in state.values())
    if isinstance(state, list):
        return any(has_images(v) for v in state)
    return False


def split_images(state: Any) -> tuple[Any, list[bytes]]:
    """Return (state with placeholders in place of image refs, image bytes in document order)."""
    blobs: list[bytes] = []

    def walk(x):
        if is_image_ref(x):
            blobs.append(load_bytes(x["source"]))
            return IMAGE_PLACEHOLDER
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(state), blobs


def load_bytes(source: str) -> bytes:
    if source.startswith("data:"):
        header, _, payload = source.partition(",")
        if ";base64" not in header:
            raise ValueError("only base64 data: URIs are supported")
        return base64.b64decode(payload)
    if source.startswith(("http://", "https://")):
        import httpx

        r = httpx.get(source, timeout=30, follow_redirects=True)
        r.raise_for_status()
        return r.content
    with open(source, "rb") as f:
        return f.read()


def to_pil(blob: bytes):
    from PIL import Image, UnidentifiedImageError

    try:
        return Image.open(io.BytesIO(blob)).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"image source is not a decodable image ({len(blob)} bytes)") from e


def digest(blobs: list[bytes]) -> str:
    h = hashlib.sha256()
    for b in blobs:
        h.update(hashlib.sha256(b).digest())
    return h.hexdigest()
