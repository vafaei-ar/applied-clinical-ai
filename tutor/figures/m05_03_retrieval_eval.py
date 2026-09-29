"""Figures for lesson m05-03-retrieval-eval."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from _style import GREEN, INK, MUTED, ORANGE, RED, figure, save

ROWS = [
    ("Hemorrhagic stroke: BP targets", 0),
    ("Minor stroke: DAPT for 21 days", 2),
    ("DAPT trials summary (CHANCE, POINT)", 1),
    ("TIA workup: imaging", 0),
    ("NIHSS ≤ 3: aspirin + clopidogrel", 2),
]
GRADE = {0: ("not relevant", MUTED), 1: ("partly relevant (1)", ORANGE),
         2: ("highly relevant (2)", GREEN)}


def fig_ranked_list():
    """Top-5 retrieved chunks with graded relevance, plus one relevant chunk that was missed."""
    fig, ax = figure(11, 7.4)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.text(0.1, 7.2, 'Query: "How long is DAPT after a minor ischemic stroke?"', fontsize=16,
            weight="bold")

    top = 6.25
    for i, (title, g) in enumerate(ROWS):
        y = top - i * 1.0
        label, color = GRADE[g]
        ax.add_patch(FancyBboxPatch((0.1, y - 0.38), 10.8, 0.76,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=color, alpha=0.13 if g else 0.07, lw=0))
        ax.text(0.45, y, f"{i + 1}", fontsize=20, weight="bold", va="center", color=INK)
        ax.text(1.1, y, title, fontsize=16, va="center", color=INK)
        ax.text(10.75, y, label, fontsize=14, va="center", ha="right", color=color,
                weight="bold")

    y = top - 5 * 1.0 - 0.35
    ax.plot([0.1, 10.9], [y + 0.45, y + 0.45], color=MUTED, lw=1.2, ls="--")
    ax.text(0.45, y - 0.1, "…", fontsize=20, va="center")
    ax.text(1.1, y - 0.1, "rank 14: Secondary prevention: DAPT then single agent", fontsize=15,
            va="center", color=INK)
    ax.text(10.75, y - 0.1, "partly relevant (1)", fontsize=14, va="center", ha="right",
            color=ORANGE, weight="bold")
    ax.text(0.1, 0.05, "Gold set for this query: 4 relevant chunks (grades 2, 2, 1, 1).",
            fontsize=14, color=RED)
    ax.set_title("What the retriever returned (k = 5)", loc="left", pad=6)
    return save(fig, "m05-03-ranked-list")
