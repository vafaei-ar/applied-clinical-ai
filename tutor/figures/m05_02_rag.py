"""Figures for lesson m05-02-rag."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, RED, figure, save


def _box(ax, x, y, w, h, text, color, fontsize=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=color, alpha=0.16, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec=color, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=INK, weight="bold", linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2, "mutation_scale": 18})


def _fail(ax, x, y, n):
    ax.plot(x, y, "o", ms=24, color=RED, zorder=6)
    ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=13, weight="bold",
            zorder=7)


def fig_rag_pipeline():
    """Offline indexing and online answering, with numbered failure points."""
    fig, ax = figure(12, 7.4)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    # Offline row
    ax.text(0.1, 6.95, "OFFLINE  (when guidelines change)", fontsize=14, color=MUTED,
            weight="bold")
    w, h, y = 2.6, 1.25, 5.35
    xs = [0.1, 3.2, 6.3, 9.3]
    labels = ["Guidelines +\ndischarge\nsummaries", "Section-aware\nchunking\n+ metadata",
              "Embed chunks\n+ BM25 index", "Vector + lexical\nstore"]
    colors = [MUTED, ORANGE, PURPLE, BLUE]
    for x, lab, c in zip(xs, labels, colors):
        _box(ax, x, y, w, h, lab, c)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, a + w + 0.02, y + h / 2, b - 0.05, y + h / 2)

    # Online row
    ax.text(0.1, 4.25, "ONLINE  (every question)", fontsize=14, color=MUTED, weight="bold")
    y2 = 2.55
    labels2 = ["Question\n+ filters\n(date, source)", "Hybrid search\ntop 50",
               "Rerank\n(cross-encoder)\ntop 5", "LLM: answer\nonly from\ncited chunks"]
    colors2 = [MUTED, BLUE, PURPLE, GREEN]
    for x, lab, c in zip(xs, labels2, colors2):
        _box(ax, x, y2, w, h, lab, c)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, a + w + 0.02, y2 + h / 2, b - 0.05, y2 + h / 2)
    # store feeds search
    _arrow(ax, 10.6, y - 0.05, 4.9, y2 + h + 0.05)

    # Output
    _box(ax, 6.3, 0.35, 5.6, 1.15, "Answer + [chunk IDs]  →  check citations", GREEN,
         fontsize=15)
    _arrow(ax, 10.6, y2 - 0.05, 10.6, 1.55)

    # Failure points
    _fail(ax, 3.45, y + h + 0.05, 1)
    _fail(ax, 0.35, y2 + h + 0.05, 2)
    _fail(ax, 3.45, y2 + h + 0.05, 3)
    _fail(ax, 9.55, y2 + h + 0.05, 4)
    ax.text(0.1, 1.45, "1 chunk splits a recommendation\n2 wrong or missing filter\n"
            "3 right chunk not retrieved\n4 answer ignores or goes beyond context",
            fontsize=13, color=RED, va="top", linespacing=1.4)
    ax.set_title("A RAG pipeline and where it breaks", loc="left", pad=6)
    return save(fig, "m05-02-rag-pipeline")
