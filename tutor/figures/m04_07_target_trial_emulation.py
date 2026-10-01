"""Figures for lesson m04-07-target-trial-emulation."""

from __future__ import annotations

from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, figure, save


def _axis(ax, y):
    ax.annotate("", xy=(10.3, y), xytext=(0.2, y),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.8})


def _mark(ax, x, y, color, label, above=True):
    ax.plot(x, y, "o", ms=14, color=color, zorder=5)
    ax.text(x, y + (0.28 if above else -0.3), label, ha="center",
            va="bottom" if above else "top", fontsize=13, color=color, weight="bold")


def fig_time_zero():
    """Correct emulation aligns eligibility, assignment, and follow-up; two common failures don't."""
    fig, ax = figure(11, 8.2)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(-0.3, 8.2)
    ax.axis("off")

    # Row 1: correct emulation
    y = 6.5
    ax.text(0.2, y + 1.25, "Correct: all three line up at time zero", fontsize=16,
            weight="bold", color=GREEN)
    _axis(ax, y)
    ax.plot([3.0, 3.0], [y - 0.45, y + 0.95], color=GREEN, lw=3)
    ax.add_patch(Rectangle((3.0, y - 0.18), 6.8, 0.36, color=to_rgba(GREEN, 0.25), lw=0))
    ax.text(3.1, y + 0.62, "eligible + strategy assigned + follow-up starts", fontsize=13,
            color=INK, va="center")
    ax.text(6.4, y - 0.4, "follow-up", fontsize=13, color=GREEN, ha="center", va="top")
    _mark(ax, 1.2, y, MUTED, "index\nstroke", above=False)

    # Row 2: immortal time
    y = 3.75
    ax.text(0.2, y + 1.25, "Immortal time: follow-up starts before assignment", fontsize=16,
            weight="bold", color=RED)
    _axis(ax, y)
    ax.add_patch(Rectangle((1.2, y - 0.18), 2.6, 0.36, color=to_rgba(RED, 0.3), lw=0))
    ax.add_patch(Rectangle((3.8, y - 0.18), 6.0, 0.36, color=to_rgba(MUTED, 0.2), lw=0))
    ax.plot([1.2, 1.2], [y - 0.45, y + 0.6], color=INK, lw=3)
    ax.text(1.2, y + 0.68, "follow-up starts", fontsize=13, ha="center", va="bottom")
    _mark(ax, 3.8, y, ORANGE, "first statin fill\n(day 60) → 'user'")
    ax.text(2.5, y - 0.4, "must survive\nto be a 'user'", fontsize=13, color=RED,
            ha="center", va="top")

    # Row 3: prevalent user
    y = 1.0
    ax.text(0.2, y + 1.25, "Prevalent user: treatment started long before", fontsize=16,
            weight="bold", color=ORANGE)
    _axis(ax, y)
    _mark(ax, 0.9, y, BLUE, "statin started\nyears earlier", above=False)
    ax.add_patch(Rectangle((0.9, y - 0.08), 5.0, 0.16, color=to_rgba(BLUE, 0.35), lw=0))
    ax.plot([5.9, 5.9], [y - 0.45, y + 0.6], color=INK, lw=3)
    ax.text(5.9, y + 0.68, "follow-up starts at stroke", fontsize=13, ha="center", va="bottom")
    ax.add_patch(Rectangle((5.9, y - 0.18), 3.9, 0.36, color=to_rgba(MUTED, 0.2), lw=0))
    ax.text(3.4, y - 0.35, "early harms and dropouts\nnever enter the study", fontsize=13,
            color=ORANGE, ha="center", va="top")

    ax.text(10.3, -0.25, "schematic, not to scale", fontsize=11, color=MUTED,
            style="italic", ha="right")
    return save(fig, "m04-07-time-zero")


if __name__ == "__main__":
    print(fig_time_zero())
