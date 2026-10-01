"""Figures for lesson m01-02-sql-core."""

from __future__ import annotations

from matplotlib.patches import Rectangle

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, figure, save


def _grid(ax, x, y, header, rows, color, col_w, row_h=0.52, highlight=None):
    """Draw a small table with its top-left corner at (x, y)."""
    total_w = sum(col_w)
    ax.add_patch(Rectangle((x, y - row_h), total_w, row_h, fc=color, alpha=0.85, lw=0))
    cx = x
    for h, w in zip(header, col_w):
        ax.text(cx + 0.12, y - row_h / 2, h, fontsize=13, color="white", weight="bold", va="center",
                family="monospace")
        cx += w
    for i, row in enumerate(rows):
        ry = y - row_h * (i + 2)
        ax.add_patch(Rectangle((x, ry), total_w, row_h, fc=color, alpha=0.07 if i % 2 else 0.14, lw=0))
        cx = x
        for j, (val, w) in enumerate(zip(row, col_w)):
            c = RED if highlight is not None and j == highlight else INK
            ax.text(cx + 0.12, ry + row_h / 2, val, fontsize=13, color=c, va="center",
                    family="monospace", weight="bold" if c == RED else "normal")
            cx += w


def fig_fanout():
    """One encounter, three diagnoses, two procedures, one claim: the join returns six rows."""
    fig, ax = figure(10.2, 7.2)
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    _grid(ax, 0.2, 7.2, ["claim", "paid"], [["E1", "$500"]], ORANGE, [1.2, 1.3])
    _grid(ax, 0.2, 5.6, ["dx"], [["I63.9"], ["I10"], ["I48.91"]], GREEN, [2.5])
    _grid(ax, 0.2, 2.9, ["proc"], [["70450"], ["70551"]], BLUE, [2.5])
    ax.text(0.2, 0.75, "3 × 2 × 1\n= 6 rows", fontsize=16, weight="bold", color=INK, va="center")

    ax.annotate("", xy=(4.3, 4.2), xytext=(3.0, 4.2),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2.2})
    ax.text(3.65, 4.55, "JOIN", ha="center", fontsize=13, color=MUTED, weight="bold")

    rows = [[d, p, "$500"] for d in ["I63.9", "I10", "I48.91"] for p in ["70450", "70551"]]
    _grid(ax, 4.5, 6.6, ["dx", "proc", "paid"], rows, MUTED, [2.0, 1.8, 1.4], highlight=2)
    ax.text(4.5, 2.25, "SUM(paid) = $3,000", fontsize=18, color=RED, weight="bold")
    ax.text(4.5, 1.6, "true spend = $500", fontsize=16, color=GREEN, weight="bold")
    ax.text(4.5, 0.75, "Fix: aggregate each child table to one row per\nencounter first, then join.",
            fontsize=13.5, color=INK, va="center")
    ax.set_title("Joining two child tables multiplies rows", loc="left", pad=6)
    return save(fig, "m01-02-fanout")
