"""Figures for lesson m03-01-package-and-test."""

from __future__ import annotations

from matplotlib.patches import Polygon

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, figure, save


def fig_test_pyramid():
    """A test pyramid for ML: unit, data, model behaviour, integration."""
    fig, ax = figure(11, 7)
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.9, 6.3)
    ax.axis("off")

    apex_x, base_l, base_r, top = 4.2, 0.2, 8.2, 5.6
    levels = [
        (0.0, 1.5, BLUE, "Unit", "pure functions\nms each, hundreds"),
        (1.5, 2.9, GREEN, "Data", "schema, ranges,\nleakage rules"),
        (2.9, 4.2, PURPLE, "Model behaviour", "invariance, direction,\ngolden outputs"),
        (4.2, 5.6, ORANGE, "Integration", "end to end,\nseconds, few"),
    ]

    def half_width(y):
        return (base_r - base_l) / 2 * (1 - y / top)

    for y0, y1, color, name, note in levels:
        w0, w1 = half_width(y0), half_width(y1)
        poly = Polygon(
            [(apex_x - w0, y0), (apex_x + w0, y0), (apex_x + w1, y1), (apex_x - w1, y1)],
            closed=True, facecolor=color, alpha=0.22, edgecolor="white", lw=3,
        )
        ax.add_patch(poly)
        ym = (y0 + y1) / 2
        ax.text(apex_x, ym, name, ha="center", va="center", fontsize=16 if y0 < 4 else 14,
                weight="bold", color=color)
        ax.text(apex_x + w0 + 0.35 if y0 > 0 else 8.5, ym, note, ha="left", va="center",
                fontsize=13.5, color=INK)

    ax.annotate("", xy=(10.7, 5.4), xytext=(10.7, 0.2),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 2})
    ax.text(10.55, 2.8, "slower, fewer", rotation=90, ha="right", va="center", fontsize=13,
            color=MUTED)

    ax.text(apex_x, -0.5, "fed by tiny synthetic fixtures with a fixed seed", ha="center",
            va="center", fontsize=14, color=MUTED, style="italic")
    ax.set_title("A test pyramid for ML code", loc="left", pad=4)
    return save(fig, "m03-01-test-pyramid")
