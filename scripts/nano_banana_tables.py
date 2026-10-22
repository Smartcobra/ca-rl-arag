#!/usr/bin/env python3
"""Draw the four paper tables with Nano Banana 2.

Writes paper_v4/figures/table_related.png, table_frozen.png,
table_terms.png, and table_learned.png. Every cell is copied from
paper.tex. The script does not invent rows.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nano_banana_pipeline import DEFAULT_MODEL, ENV_PATH, crop_margin, generate, load_env

OUT = ROOT / "paper_v4" / "figures"

STYLE = """
Draw one publication table, nothing else. No title, no caption, no page number, no watermark.
White background. Soft powder-blue header row, white body, hairline gray rules, dark gray text.
Sans-serif. Flat. No photographs, no 3D, no icons, no extra rows, no extra columns.
Copy every string below exactly, including slashes, minus signs, and decimals.
Heavier weight means a bold font. Never surround a cell with asterisks or stars.
""".strip()

TABLES = [
    (
        "table_related.png",
        "16:9",
        """
Four columns. Header: Method | What it decides | What it prices | What we add
Row 1: Adaptive-RAG (Jeong et al. 2024) | No retrieval, one retrieval, or multi-step retrieval, from the question alone | Step count and relative time | A second look after the passages arrive, with dollars and a per-action charge
Row 2: Self-RAG (Asai et al. 2024) | Whether to retrieve, and whether a segment is supported, as tokens in the generator | Hand-set critique weights | A frozen generator. Critique is one typed action with a measured cost
Row 3: CRAG (Yan et al. 2024) | Keep, discard, or mix the retrieved set. Discard triggers a web search | An evaluator threshold | Rewrite, rerank, and verify on a closed corpus
Row 4: Search-R1 (Jin et al. 2025) | When the language model itself emits a search during reasoning | Answer correctness | A small controller, so a change in exact match is a change in control
Row 5: GRASP (Gandhi et al. 2026) | Semantic search, keyword search, or a paragraph read | Answer quality, grounded reading, using both searches, and a turn count | Rewrite, rerank, and verify, priced in dollars
""",
    ),
    (
        "table_frozen.png",
        "16:9",
        """
Five columns. The top-left header cell is empty white. Do not write the word blank.
Other headers: One-shot | Rule | Max tools | Six-step recipe
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
        """
Exactly 4 body rows and 1 header row. Six columns. Each row label appears once.
Top-left header cell is empty.
Header cells 2 to 6: Qans with ans as a subscript, Qground with ground as a subscript, Qcal with cal as a subscript, dollar sign times 10 to the minus 4, Reward.
Body row 1 label One-shot. Cells: 0.366, 0.646, -0.137, 1.72, 0.581. The last two cells are bold.
Body row 2 label Rule. Cells: 0.360, 0.652, -0.146, 5.04, 0.534. None bold.
Body row 3 label Max tools. Cells: 0.385, 0.658, -0.127, 7.47, 0.503. The first three cells are bold.
Body row 4 label Six-step recipe. Cells: 0.366, 0.646, -0.137, 4.62, 0.498. None bold.
The One-shot calibration cell contains only -0.137. Do not also write -0.127 in that cell.
Do not repeat the word One-shot.
""",
    ),
    (
        "table_learned.png",
        "16:9",
        """
Seven columns. Headers: Seed | Epoch | HotpotQA | Natural Questions | Correct | Steps | a dollar sign, a times sign, and 10 with superscript minus 4
42 | 17 | 52/150 | 48/150 | 100/300 | 4 | 2.02
43 | 14 | 52/150 | 48/150 | 100/300 | 4 | 2.02
44 | 11 | 52/150 | 48/150 | 100/300 | 4 | 2.02
No bold cells. No stars. Do not write a caret.
""",
    ),
]


def main() -> None:
    env = load_env(ENV_PATH)
    api_key = os.environ.get("GEMINI_API_KEY") or env.get("GEMINI_API_KEY", "")
    model = os.environ.get("NANO_BANANA_MODEL") or env.get("NANO_BANANA_MODEL", DEFAULT_MODEL)
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in .env.")
    only = set(sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, ratio, body in TABLES:
        if only and name not in only and Path(name).stem not in only:
            continue
        path = OUT / name
        image = generate(api_key, model, STYLE + "\n" + body.strip(), aspect_ratio=ratio)
        path.write_bytes(image)
        crop_margin(path)
        print(f"wrote {path} with {model}")


if __name__ == "__main__":
    main()
