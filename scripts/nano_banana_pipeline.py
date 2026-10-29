#!/usr/bin/env python3
"""Draw paper_v4/figures/pipeline.png with Nano Banana 2.

Reads GEMINI_API_KEY and NANO_BANANA_MODEL from the repo .env.
The default model is gemini-3.1-flash-image, which is Nano Banana 2.
Set NANO_BANANA_MODEL=gemini-nano-banana-2.1 to use Nano Banana 2.1.

  python scripts/nano_banana_pipeline.py
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
OUT_PATH = ROOT / "paper_v4" / "figures" / "pipeline.png"
DEFAULT_MODEL = "gemini-3.1-flash-image"

# One overview figure. Illustrated ACL-style diagram, not a box flowchart.
NANO_BANANA_STYLE = """
Draw exactly one landscape figure, about twice as wide as tall, for an NLP paper.
Soft ACL diagram: powder-blue, mint, peach, and lavender panels, thin gray lines,
round icon badges, dark gray labels, white background, lots of space.
Flat vector. No photos, no 3D, no shadows, no logos, no watermark, no extra captions.
Do not use black rectangles. Do not repeat any label. Exactly these four panels and nothing else.

Panel 1, left, titled Data.
Document icon, label HotpotQA.
Document icon, label Natural Questions.
Stack icon, label 83,120 passages.

Panel 2, titled Episode.
Five round badges in a circle, clockwise, with arrowheads:
Observation, then Policy, then Action, then Tool, then New observation, back to Observation.
A small arc above the circle is labeled loop.
Policy is a tiny three-node network, not a game controller.
Under the circle, five small colored chips with readable labels:
retrieve, rewrite, rerank, verify, stop.

Panel 3, titled Reward. One path only, left to right, arrowheads point right:
receipt icon labeled Price card, right arrow, coin labeled Reward, right arrow, one loop icon labeled REINFORCE.
From REINFORCE only, one arrow points up into Policy.
There is no second price card and no second REINFORCE.
Action is not in this panel.

Panel 4, right, titled Test.
Stack labeled 300 test questions, right arrow, a small scatter plot titled Frontier.
Y axis label: Exact match. X axis label: Dollars. A few pastel dots. No axis numbers.
""".strip()


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def crop_margin(path: Path) -> None:
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return
    image = Image.open(path).convert("RGB")
    pixels = np.asarray(image)
    ink = np.any(pixels < 245, axis=2)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return
    pad = 16
    top = max(0, int(rows[0]) - pad)
    bottom = min(pixels.shape[0], int(rows[-1]) + pad + 1)
    left = max(0, int(cols[0]) - pad)
    right = min(pixels.shape[1], int(cols[-1]) + pad + 1)
    image.crop((left, top, right, bottom)).save(path, "PNG")


def generate(api_key: str, model: str, prompt: str, aspect_ratio: str | None = None) -> bytes:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    generation: dict = {"responseModalities": ["IMAGE"]}
    if aspect_ratio:
        generation["imageConfig"] = {"aspectRatio": aspect_ratio}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation,
    }
    context = ssl.create_default_context()
    try:
        import certifi
        context.load_verify_locations(certifi.where())
    except ImportError:
        pass
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180, context=context) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:800]
        raise SystemExit(f"Gemini image request failed ({exc.code}): {detail}") from exc

    for candidate in payload.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    raise SystemExit(f"No image in the Gemini response: {json.dumps(payload)[:800]}")


def main() -> None:
    env = load_env(ENV_PATH)
    api_key = os.environ.get("GEMINI_API_KEY") or env.get("GEMINI_API_KEY", "")
    model = os.environ.get("NANO_BANANA_MODEL") or env.get("NANO_BANANA_MODEL", DEFAULT_MODEL)
    if not api_key:
        raise SystemExit(
            "Set GEMINI_API_KEY in .env. "
            "The key comes from https://aistudio.google.com/apikey"
        )
    image = generate(api_key, model, NANO_BANANA_STYLE)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(image)
    crop_margin(OUT_PATH)
    print(f"wrote {OUT_PATH} with {model}")


if __name__ == "__main__":
    main()
