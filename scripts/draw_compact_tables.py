#!/usr/bin/env python3
"""Compact paper tables. Same cells as paper.tex, tight row height."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_v4" / "figures"

INK = "#243040"
HEADER = "#D6E8F6"
LINE = "#C5D0DA"
ZEBRA = "#F4F8FB"
BOLD = "#142033"


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def draw(path: Path, headers: list[str], rows: list[list[str]], wraps: list[int],
         bold: set[tuple[int, int]], col_w: list[float]) -> None:
    wrapped_headers = [wrap(h, w) if h else "" for h, w in zip(headers, wraps)]
    wrapped_rows = [[wrap(cell, w) for cell, w in zip(row, wraps)] for row in rows]

    def lines(text: str) -> int:
        return max(1, text.count("\n") + 1) if text else 1

    header_lines = max(lines(h) for h in wrapped_headers)
    row_lines = [max(lines(c) for c in row) for row in wrapped_rows]
    line_h = 0.20
    pad = 0.07
    header_h = header_lines * line_h + pad
    body_h = [n * line_h + pad for n in row_lines]
    total_h = header_h + sum(body_h)
    total_w = sum(col_w)

    fig_w = total_w
    fig_h = total_h + 0.02
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    y = total_h
    x0 = 0
    y -= header_h
    for header, width in zip(wrapped_headers, col_w):
        ax.add_patch(Rectangle((x0, y), width, header_h, facecolor=HEADER,
                               edgecolor=LINE, linewidth=0.6))
        if header:
            ax.text(x0 + width / 2, y + header_h / 2, header, ha="center", va="center",
                    fontsize=8.2, color=BOLD, fontweight="bold", linespacing=1.15)
        x0 += width

    for r, row in enumerate(wrapped_rows):
        h = body_h[r]
        y -= h
        x0 = 0
        face = ZEBRA if r % 2 else "white"
        for c, (cell, width) in enumerate(zip(row, col_w)):
            ax.add_patch(Rectangle((x0, y), width, h, facecolor=face,
                                   edgecolor=LINE, linewidth=0.6))
            weight = "bold" if (r, c) in bold else "regular"
            color = BOLD if (r, c) in bold else INK
            ax.text(x0 + width / 2, y + h / 2, cell, ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight, linespacing=1.15)
            x0 += width

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(path)


def main() -> None:
    draw(
        OUT / "table_related.png",
        ["Method", "What it decides", "What it prices", "What we add"],
        [
            ["Adaptive-RAG\n(Jeong et al. 2024)",
             "No retrieval, one retrieval, or multi-step retrieval, from the question alone",
             "Step count and relative time",
             "A second look after the passages arrive, with dollars and a per-action charge"],
            ["Self-RAG\n(Asai et al. 2024)",
             "Whether to retrieve, and whether a segment is supported, as tokens in the generator",
             "Hand-set critique weights",
             "A frozen generator. Critique is one typed action with a measured cost"],
            ["CRAG\n(Yan et al. 2024)",
             "Keep, discard, or mix the retrieved set. Discard triggers a web search",
             "An evaluator threshold",
             "Rewrite, rerank, and verify on a closed corpus"],
            ["Search-R1\n(Jin et al. 2025)",
             "When the language model itself emits a search during reasoning",
             "Answer correctness",
             "A small controller, so a change in exact match is a change in control"],
            ["GRASP\n(Gandhi et al. 2026)",
             "Semantic search, keyword search, or a paragraph read",
             "Answer quality, grounded reading, using both searches, and a turn count",
             "Rewrite, rerank, and verify, priced in dollars"],
        ],
        [18, 36, 24, 32],
        set(),
        [1.7, 3.15, 2.15, 2.9],
    )
    draw(
        OUT / "table_frozen.png",
        ["", "One-shot", "Rule", "Max tools", "Six-step recipe"],
        [
            ["HotpotQA", "60/150", "56/150", "62/150", "60/150"],
            ["Natural Questions", "40/150", "41/150", "43/150", "40/150"],
            ["Correct", "100/300", "97/300", "105/300", "100/300"],
            ["$ × 10⁻⁴", "1.72", "5.04", "7.47", "4.62"],
            ["Reward", "0.581", "0.534", "0.503", "0.498"],
        ],
        [16, 12, 10, 12, 14],
        {(0, 3), (1, 3), (2, 3), (3, 1), (4, 1)},
        [1.7, 1.15, 1.05, 1.2, 1.45],
    )
    draw(
        OUT / "table_terms.png",
        ["", r"$Q_{\mathrm{ans}}$", r"$Q_{\mathrm{ground}}$", r"$Q_{\mathrm{cal}}$",
         "$ × 10⁻⁴", "Reward"],
        [
            ["One-shot", "0.366", "0.646", "−0.137", "1.72", "0.581"],
            ["Rule", "0.360", "0.652", "−0.146", "5.04", "0.534"],
            ["Max tools", "0.385", "0.658", "−0.127", "7.47", "0.503"],
            ["Six-step recipe", "0.366", "0.646", "−0.137", "4.62", "0.498"],
        ],
        [16, 10, 12, 10, 12, 10],
        {(0, 4), (0, 5), (2, 1), (2, 2), (2, 3)},
        [1.55, 0.95, 1.15, 0.95, 1.15, 0.95],
    )
    draw(
        OUT / "table_learned.png",
        ["Seed", "Epoch", "HotpotQA", "Natural Questions", "Correct", "Steps", "$ × 10⁻⁴"],
        [
            ["42", "17", "52/150", "48/150", "100/300", "4", "2.02"],
            ["43", "14", "52/150", "48/150", "100/300", "4", "2.02"],
            ["44", "11", "52/150", "48/150", "100/300", "4", "2.02"],
        ],
        [8, 8, 12, 14, 10, 8, 12],
        set(),
        [0.7, 0.75, 1.15, 1.55, 1.05, 0.7, 1.05],
    )
    draw(
        OUT / "table_adaptive.png",
        ["", "HotpotQA", "Natural Questions", "Correct", "One-shot", "Rewrite-first", "$ × 10⁻⁴"],
        [
            ["Adaptive-RAG", "55/150", "45/150", "100/300", "63/300", "237/300", "1.96"],
        ],
        [14, 12, 16, 10, 12, 14, 12],
        set(),
        [1.45, 1.15, 1.55, 1.05, 1.15, 1.4, 1.05],
    )


if __name__ == "__main__":
    main()
