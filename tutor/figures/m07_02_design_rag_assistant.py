"""Figures for lesson m07-02-design-rag-assistant."""

from __future__ import annotations

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, figure, save


def _box(ax, x, y, w, h, color, title, body, dashed=False):
    style = "round,pad=0.02,rounding_size=0.12"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=color, alpha=0.13, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor="none", edgecolor=color,
                                lw=2, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=14.5, weight="bold",
            color=color, linespacing=1.05)
    ax.text(x + w / 2, y + 0.2, body, ha="center", va="bottom", fontsize=12, color=INK,
            linespacing=1.15)


def _arrow(ax, a, b, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=18, color=color, lw=2.1,
                                 ls=ls))


def fig_architecture():
    """Offline ingestion lane, online query lane, and the evaluation/monitoring lane."""
    fig, ax = figure(12.5, 6.8)
    ax.set_xlim(0, 12.9)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    ax.text(0.1, 6.6, "OFFLINE  ingestion", fontsize=14, color=MUTED, weight="bold")
    w1, h = 2.75, 1.45
    xs1 = [0.1, 3.4, 6.7, 10.0]
    y1 = 4.95
    _box(ax, xs1[0], y1, w1, h, BLUE, "Guidelines", "PDFs, intranet,\norder sets")
    _box(ax, xs1[1], y1, w1, h, BLUE, "Parse + chunk", "by section,\nkeep tables")
    _box(ax, xs1[2], y1, w1, h, BLUE, "Metadata", "version, effective\ndate, owner")
    _box(ax, xs1[3], y1, w1, h, BLUE, "Index", "BM25 + vectors,\nactive only")
    for i in range(3):
        _arrow(ax, (xs1[i] + w1 + 0.03, y1 + h / 2), (xs1[i + 1] - 0.03, y1 + h / 2))

    ax.text(0.1, 4.3, "ONLINE  per question", fontsize=14, color=MUTED, weight="bold")
    w2 = 2.15
    xs2 = [0.1, 2.75, 5.4, 8.05, 10.7]
    y2 = 2.6
    _box(ax, xs2[0], y2, w2, h, ORANGE, "Guard", "scope +\nPHI check")
    _box(ax, xs2[1], y2, w2, h, GREEN, "Retrieve", "hybrid top-k\n+ rerank")
    _box(ax, xs2[2], y2, w2, h, GREEN, "Generate", "answer only\nfrom sources")
    _box(ax, xs2[3], y2, w2, h, ORANGE, "Verify", "citations support\nclaims? else abstain")
    _box(ax, xs2[4], y2, w2, h, GREEN, "Answer", "with section\n+ version cited")
    for i in range(4):
        _arrow(ax, (xs2[i] + w2 + 0.03, y2 + h / 2), (xs2[i + 1] - 0.03, y2 + h / 2))
    _arrow(ax, (xs1[3] + 0.4, y1 - 0.03), (xs2[1] + w2 / 2 + 0.3, y2 + h + 0.03), color=BLUE)

    _box(ax, 0.1, 0.2, 12.75, 1.6, PURPLE, "Evaluate + monitor",
         "golden questions · retrieval recall · faithfulness · abstention · clinician review\n"
         "logs are PHI · regression suite on every corpus, prompt, or model change",
         dashed=True)
    ax.set_title("Guideline RAG assistant: three lanes", loc="left", pad=6)
    return save(fig, "m07-02-architecture")
