"""Figures for lesson m01-07-scale-and-tools."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch, Rectangle

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, RED, figure, save

COLS = ["id", "patient", "start", "type", "facility"]
COL_COLORS = [BLUE, PURPLE, GREEN, ORANGE, RED]


def fig_row_vs_columnar():
    """The same encounters table laid out row-wise and column-wise; one query reads one column."""
    fig, ax = figure(11, 5.6)
    ax.set_xlim(0, 11)
    ax.set_ylim(0.2, 5.75)
    ax.axis("off")

    cell_w, cell_h, n_rows = 0.42, 0.52, 4

    def strip(y, order, label, highlight_type):
        ax.text(0.15, y + cell_h + 0.22, label, fontsize=16, weight="bold", va="bottom")
        x = 0.15
        for k, (r, c) in enumerate(order):
            col = COL_COLORS[c]
            hit = c == 3
            ax.add_patch(Rectangle((x, y), cell_w, cell_h, fc=col,
                                   alpha=0.95 if (hit and highlight_type) else 0.28,
                                   ec="white", lw=1.5))
            x += cell_w
            if highlight_type is False and c == 4 and r < n_rows - 1:
                x += 0.08
            if highlight_type is True and r == n_rows - 1 and c < 4:
                x += 0.08
        return x

    row_order = [(r, c) for r in range(n_rows) for c in range(5)]
    col_order = [(r, c) for c in range(5) for r in range(n_rows)]

    strip(4.35, row_order, "Row store (PostgreSQL heap): whole records together", False)
    # In the row store every block holding the type column must be read
    ax.add_patch(Rectangle((0.1, 4.28), 8.75, 0.66, fill=False, ec=INK, lw=2, ls="--"))
    ax.text(9.0, 4.61, "read\neverything", fontsize=13.5, va="center", color=INK)

    strip(1.75, col_order, "Column store (DuckDB, Parquet): each column together", True)
    x_type = 0.15 + 3 * (n_rows * cell_w + 0.08)
    ax.add_patch(Rectangle((x_type - 0.05, 1.68), n_rows * cell_w + 0.1, 0.66, fill=False,
                           ec=INK, lw=2, ls="--"))
    ax.text(9.0, 2.01, "read one\ncolumn", fontsize=13.5, va="center", color=INK)

    for i, (name, col) in enumerate(zip(COLS, COL_COLORS)):
        xx = 0.3 + i * 1.75
        ax.add_patch(Rectangle((xx, 0.35), 0.35, 0.35, fc=col, alpha=0.8, lw=0))
        ax.text(xx + 0.45, 0.52, name, fontsize=14, va="center")
    ax.set_title("SELECT encounter_type, COUNT(*) … GROUP BY 1", loc="left", pad=4,
                 fontsize=18, family="monospace")
    return save(fig, "m01-07-row-vs-columnar")


def _node(ax, x, y, text, color, w=1.9, h=0.9, fs=14):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=color, alpha=0.18))
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec=color, lw=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK)


def _arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw,
                            "shrinkA": 2, "shrinkB": 2})


def fig_shuffle_vs_broadcast():
    """Shuffle join moves both sides across the network; broadcast join copies the small side."""
    fig, ax = figure(11, 7.2)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Left: shuffle (sort-merge) join
    ax.text(0.2, 6.75, "Shuffle join", fontsize=18, weight="bold", va="center")
    ax.text(0.2, 6.3, "both sides re-partitioned by key", fontsize=13, color=MUTED, va="center")
    xs_top = [0.9, 2.55, 4.2]
    xs_bot = [0.9, 2.55, 4.2]
    for x in xs_top:
        _node(ax, x, 5.2, "input\npartition", BLUE, w=1.45, h=0.95, fs=12.5)
    for x in xs_bot:
        _node(ax, x, 1.7, "same key,\nsame task", GREEN, w=1.45, h=0.95, fs=12.5)
    for x0 in xs_top:
        for x1 in xs_bot:
            _arrow(ax, x0, 4.7, x1, 2.2, color=RED, lw=1.5)
    ax.text(2.55, 3.45, "network\nshuffle", ha="center", va="center", fontsize=14,
            color=RED, weight="bold", bbox={"fc": "white", "ec": "none", "pad": 2})
    ax.text(2.55, 0.65, "needed when both sides are big", ha="center", fontsize=13, color=INK)

    ax.plot([5.45, 5.45], [0.4, 7.0], color="#e4e7eb", lw=2)

    # Right: broadcast hash join
    ax.text(5.8, 6.75, "Broadcast join", fontsize=18, weight="bold", va="center")
    ax.text(5.8, 6.3, "small table copied to every executor", fontsize=13, color=MUTED,
            va="center")
    _node(ax, 8.35, 5.2, "facilities\n(small)", ORANGE, w=2.0, h=0.95, fs=13)
    xs = [6.6, 8.35, 10.1]
    for x in xs:
        _node(ax, x, 1.7, "diagnoses\npartition\n+ copy", BLUE, w=1.45, h=1.2, fs=12)
        _arrow(ax, 8.35, 4.7, x, 2.35, color=ORANGE, lw=2)
    ax.text(8.35, 0.65, "big side never moves", ha="center", fontsize=13, color=INK)
    return save(fig, "m01-07-shuffle-vs-broadcast")
