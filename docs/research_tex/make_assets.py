#!/usr/bin/env python3
"""Draw the guide figure and the result tables with Nano Banana.

Writes docs/research_tex/figures/. Every cell is copied from docs/research.html.
Pass image filenames to redraw a subset:

  python docs/research_tex/make_assets.py
  python docs/research_tex/make_assets.py table_learned.png
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from nano_banana_pipeline import DEFAULT_MODEL, ENV_PATH, crop_margin, generate, load_env

OUT = Path(__file__).resolve().parent / "figures"

TABLE_STYLE = """
Draw one publication table, nothing else. No title, no caption, no page number, no watermark.
White background. Soft powder-blue header row, white body, hairline gray rules, dark gray text.
Sans-serif. Flat. No photographs, no 3D, no icons, no extra rows, no extra columns.
Copy every string below exactly, including slashes, minus signs, and decimals.
Heavier weight means a bold font. Never surround a cell with asterisks or stars.
""".strip()

FIGURE = (
    "overview.png",
    "16:9",
    """
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

Panel 3, titled Checkpoint.
A horizontal ruler of about twelve short vertical marks. The middle mark is filled, and the only word under it is Kept.
Under Kept, a document icon labeled best.pt.
A closed lock to the right is labeled 300, with the small words chosen first beside the lock.
Do not write the words epoch, tick, or axis anywhere in this panel.
There is no second lock and no test score inside this panel.

Panel 4, right, titled Exam.
Four chips in a row: rewrite, rewrite, retrieve, stop.
Under the chips, the label 100/300.
No axis, no scatter plot, no dollar amounts.
""".strip(),
)

TABLES = [
    (
        "table_frozen.png",
        "16:9",
        TABLE_STYLE
        + """

Five columns. The top-left header cell is empty white. Do not write the word blank.
Other headers: One-shot | Rule | Max tools | Six-step
HotpotQA: 60/150 | 56/150 | 62/150 | 60/150
Natural Questions: 40/150 | 41/150 | 43/150 | 40/150
Correct: 100/300 | 97/300 | 105/300 | 100/300
The next row label is a dollar sign, a times sign, and 10 with a superscript minus 4. Do not write a caret.
That row: 1.72 | 5.04 | 7.47 | 4.62
Reward: 0.581 | 0.534 | 0.503 | 0.498
Make only these five values heavier: 62/150, 43/150, 105/300, 1.72, 0.581.
Do not draw stars or asterisks.
""",
    ),
    (
        "table_terms.png",
        "16:9",
        TABLE_STYLE
        + """

Exactly 4 body rows and 1 header row. Six columns. Each row label appears once.
Top-left header cell is empty.
Header cells 2 to 6: Qans with ans as a subscript, Qground with ground as a subscript, Qcal with cal as a subscript, dollar sign times 10 to the minus 4, Reward.
Body row 1 label One-shot. Cells: 0.366, 0.646, -0.137, 1.72, 0.581. The last two cells are bold.
Body row 2 label Rule. Cells: 0.360, 0.652, -0.146, 5.04, 0.534. None bold.
Body row 3 label Max tools. Cells: 0.385, 0.658, -0.127, 7.47, 0.503. The first three cells are bold.
Body row 4 label Six-step. Cells: 0.366, 0.646, -0.137, 4.62, 0.498. None bold.
The One-shot calibration cell contains only -0.137. Do not also write -0.127 in that cell.
Do not repeat the word One-shot.
""",
    ),
    (
        "table_learned.png",
        "16:9",
        TABLE_STYLE
        + """

Seven columns. Headers: Seed | Epoch | HotpotQA | Natural Questions | Correct | Steps | a dollar sign, a times sign, and 10 with superscript minus 4
42 | 17 | 52/150 | 48/150 | 100/300 | 4 | 2.02
43 | 14 | 52/150 | 48/150 | 100/300 | 4 | 2.02
44 | 11 | 52/150 | 48/150 | 100/300 | 4 | 2.02
No bold cells. No stars. Do not write a caret.
""",
    ),
    (
        "table_routing.png",
        "16:9",
        TABLE_STYLE
        + """

Five columns. The top-left header cell is empty white.
Other headers: HotpotQA | NQ | Correct | a dollar sign, a times sign, and 10 with superscript minus 4
One-shot | 60/150 | 40/150 | 100/300 | 1.72
Rewrite-first | 52/150 | 48/150 | 100/300 | 2.02
Adaptive-RAG | 55/150 | 45/150 | 100/300 | 1.96
Cheap label | 60/150 | 40/150 | 100/300 | 1.72
Oracle, two | 65/150 | 57/150 | 122/300 | 1.74
Oracle + max | 67/150 | 59/150 | 126/300 | 1.81
Make only these cells heavier: both 1.72 dollar cells, 67/150, 59/150, and 126/300.
No stars. Do not write a caret. Do not add a seventh row.
""",
    ),
    (
        "table_retrieve.png",
        "16:9",
        TABLE_STYLE
        + """

Seven columns. Headers: Seed | Epoch | Correct | a dollar sign times 10 to the minus 4 | Repairs | Breaks | p
42 | 17 | 103/300 | 4.60 | 7 | 4 | 0.549
43 | 2 | 100/300 | 1.72 | 0 | 0 | 1.000
44 | 19 | 103/300 | 5.62 | 9 | 6 | 0.607
No bold cells. No stars. Do not write a caret. Do not add a one-shot row.
""",
    ),
    (
        "table_sweep.png",
        "16:9",
        TABLE_STYLE
        + """

Seven columns. The first header cell is only the Greek letter λ. Do not write the English word lambda, and do not write lambd next to the letter.
Other headers: Epoch | Correct | a dollar sign times 10 to the minus 4 | Repairs | Breaks | p
0 | 18 | 104/300 | 4.82 | 10 | 6 | 0.454
5 | 7 | 100/300 | 2.60 | 0 | 0 | 1.000
20 | 17 | 103/300 | 4.60 | 7 | 4 | 0.549
80 | 4 | 100/300 | 1.72 | 0 | 0 | 1.000
The last row label is one-shot. Its epoch cell is empty. Cells: 100/300 | 1.72 | and the repairs, breaks, and p cells are empty.
No bold cells. No stars. Do not write a caret. The first header is the single character λ.
""",
    ),
]


def one(api_key: str, model: str, name: str, ratio: str, prompt: str) -> str:
    path = OUT / name
    last = "unknown error"
    for attempt in range(3):
        try:
            image = generate(api_key, model, prompt, aspect_ratio=ratio)
            path.write_bytes(image)
            crop_margin(path)
            return f"wrote {path} with {model}"
        except SystemExit as exc:
            last = str(exc)
            time.sleep(4 * (attempt + 1))
    raise SystemExit(f"{name} failed: {last}")


def main() -> None:
    env = load_env(ENV_PATH)
    api_key = os.environ.get("GEMINI_API_KEY") or env.get("GEMINI_API_KEY", "")
    model = os.environ.get("NANO_BANANA_MODEL") or env.get("NANO_BANANA_MODEL", DEFAULT_MODEL)
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in .env.")
    jobs = [FIGURE, *TABLES]
    only = set(sys.argv[1:])
    if only:
        jobs = [job for job in jobs if job[0] in only or Path(job[0]).stem in only]
        if not jobs:
            raise SystemExit(f"No matching asset in {only}")
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(one, api_key, model, *job) for job in jobs]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
