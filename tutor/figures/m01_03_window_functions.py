"""Figures for lesson m01-03-window-functions."""

from __future__ import annotations

from matplotlib.patches import Rectangle

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, figure, save


def fig_ranking_timeline():
    """One patient's encounters in time, with ROW_NUMBER, RANK, and DENSE_RANK on the strokes."""
    fig, ax = figure(11, 6.6)
    ax.set_xlim(-0.4, 10.6)
    ax.set_ylim(-3.6, 2.4)
    ax.axis("off")

    ax.annotate("", xy=(10.4, 0), xytext=(-0.3, 0),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2})
    ax.text(10.4, 0.2, "time", ha="right", va="bottom", color=MUTED, fontsize=13)
    ax.text(-0.3, 1.95, "OVER (PARTITION BY patient_id ORDER BY index_date)",
            fontsize=13.5, color=MUTED, family="monospace")

    for x in [0.6, 2.1, 5.6, 7.3]:
        ax.plot(x, 0, "o", ms=11, color=MUTED, alpha=0.6, zorder=4)
    ax.text(0.6, -0.45, "outpatient", fontsize=11.5, color=MUTED, ha="center", va="top")

    strokes = [(3.6, "2022-03-04\nE101 · ED"), (4.25, None), (8.8, "2023-07-19\nE230 · inpt")]
    for x, _ in strokes:
        ax.plot(x, 0, "o", ms=16, color=ORANGE, zorder=5)
    ax.text(3.9, 0.45, "2022-03-04\nE101 ED + E102 inpt", ha="center", va="bottom", fontsize=12.5,
            color=ORANGE, weight="bold")
    ax.text(8.8, 0.45, "2023-07-19\nE230 inpt", ha="center", va="bottom", fontsize=12.5,
            color=ORANGE, weight="bold")

    labels = [("ROW_NUMBER", ["1", "2", "3"], GREEN), ("RANK", ["1", "1", "3"], BLUE),
              ("DENSE_RANK", ["1", "1", "2"], BLUE)]
    for i, (name, vals, color) in enumerate(labels):
        y = -1.35 - i * 0.72
        ax.add_patch(Rectangle((-0.3, y - 0.3), 10.6, 0.6, fc=color, alpha=0.07 if i % 2 else 0.13, lw=0))
        ax.text(-0.15, y, name, fontsize=14, weight="bold", color=color, va="center",
                family="monospace")
        for (x, _), v in zip(strokes, vals):
            ax.text(x, y, v, fontsize=17, weight="bold", ha="center", va="center", color=INK)
    ax.text(10.5, -3.45, "ROW_NUMBER breaks the tie arbitrarily unless you add encounter_id",
            fontsize=12, color=MUTED, style="italic", ha="right")
    ax.set_title("Same-day tie: three ranking functions", loc="left", pad=4)
    return save(fig, "m01-03-ranking-timeline")


def fig_coverage_gap():
    """Two coverage periods with a gap found by LAG, and a lookback window that falls into it."""
    fig, ax = figure(11, 5.6)
    ax.set_xlim(-0.2, 10.6)
    ax.set_ylim(-2.4, 2.6)
    ax.axis("off")

    ax.add_patch(Rectangle((0.2, 0.9), 4.3, 0.75, fc=BLUE, alpha=0.8, lw=0))
    ax.text(2.35, 1.27, "coverage row 1", ha="center", va="center", color="white", fontsize=14,
            weight="bold")
    ax.add_patch(Rectangle((6.0, 0.9), 4.4, 0.75, fc=BLUE, alpha=0.8, lw=0))
    ax.text(8.2, 1.27, "coverage row 2", ha="center", va="center", color="white", fontsize=14,
            weight="bold")

    ax.annotate("", xy=(6.0, 2.05), xytext=(4.5, 2.05),
                arrowprops={"arrowstyle": "<->", "color": RED, "lw": 2})
    ax.text(5.25, 2.2, "gap", ha="center", va="bottom", color=RED, fontsize=15, weight="bold")
    ax.text(4.5, 0.75, "LAG(coverage_end)", ha="center", va="top", fontsize=12.5, color=RED,
            family="monospace")
    ax.text(6.35, 0.75, "coverage_start", ha="left", va="top", fontsize=12.5, color=RED,
            family="monospace")

    ax.add_patch(Rectangle((3.4, -1.35), 5.1, 0.7, fc=ORANGE, alpha=0.2, lw=0))
    ax.text(7.25, -1.0, "365-day lookback", ha="center", va="center", color=ORANGE, fontsize=13.5,
            weight="bold")
    ax.plot([8.5, 8.5], [-1.7, 0.2], color=INK, lw=3)
    ax.text(8.5, -2.05, "index date", ha="center", va="center", fontsize=14, weight="bold")
    ax.add_patch(Rectangle((4.5, -1.35), 1.5, 0.7, fc="none", ec=RED, lw=2, hatch="//"))
    ax.text(2.9, -1.0, "unobserved\ntime inside\nthe window →", ha="right", va="center",
            fontsize=12.5, color=RED)
    ax.set_title("LAG finds the gap; the gap breaks the lookback", loc="left", pad=4)
    return save(fig, "m01-03-coverage-gap")
