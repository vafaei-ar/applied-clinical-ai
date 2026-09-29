"""Figures for lesson m01-05-temporal-leakage."""

from __future__ import annotations

from matplotlib.patches import Rectangle

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, figure, save


def fig_leakage_timeline():
    """Lookback, index, and outcome windows on one patient timeline, with two leaks marked."""
    fig, ax = figure(11, 5.8)
    ax.set_xlim(-0.3, 10.6)
    ax.set_ylim(-1.75, 2.3)
    ax.axis("off")

    # Schematic, not to scale: lookback 0..6, index at 6.6, outcome 6.8..9.3
    ax.annotate("", xy=(10.4, 0), xytext=(-0.2, 0),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2})
    ax.text(10.4, 0.18, "time", ha="right", va="bottom", color=MUTED, fontsize=13)

    ax.add_patch(Rectangle((0, 0.3), 6.2, 0.9, color=BLUE, alpha=0.16, lw=0))
    ax.text(3.1, 0.75, "LOOKBACK — features allowed\nindex − 365d  …  index − 1d",
            ha="center", va="center", fontsize=14, color=BLUE, weight="bold")

    ax.add_patch(Rectangle((6.9, 0.3), 2.6, 0.9, color=GREEN, alpha=0.2, lw=0))
    ax.text(8.2, 0.75, "OUTCOME\nday 1 … 30", ha="center", va="center", fontsize=14,
            color=GREEN, weight="bold")

    ax.plot([6.55, 6.55], [-0.4, 1.7], color=INK, lw=3)
    ax.text(6.55, 1.8, "index date = time zero", ha="center", va="bottom", fontsize=15,
            weight="bold")

    ax.plot(5.6, 0, "o", ms=15, color=ORANGE, zorder=5)
    ax.text(5.6, -0.45, "Diagnosis dated before index,\nbut recorded 3 weeks after it\n"
            "→ unknown at time zero", ha="center", va="top", fontsize=12.5, color=ORANGE)

    ax.plot(8.0, 0, "o", ms=15, color=RED, zorder=5)
    ax.text(8.6, -0.45, "Readmission diagnosis\nused as a predictor\n→ the outcome leaks in",
            ha="center", va="top", fontsize=12.5, color=RED)

    ax.text(-0.2, -1.7, "schematic, not to scale", fontsize=11, color=MUTED, style="italic")
    ax.set_title("Every feature must be knowable before time zero", loc="left", pad=4)
    return save(fig, "m01-05-leakage-timeline")
