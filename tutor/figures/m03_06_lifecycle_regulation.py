"""Figures for lesson m03-06-lifecycle-regulation."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, figure, save


def _box(ax, x, y, w, h, text, color, fs=14, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=color, alpha=0.16, edgecolor=color, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            weight=weight)


def _arrow(ax, x0, y0, x1, y1, label=None, lx=None, ly=None, color=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2})
    if label:
        ax.text(lx, ly, label, fontsize=13, color=color, weight="bold", ha="center",
                va="center")


def fig_device_decision():
    """Simplified flow: is it a device, is it non-device CDS, which pathway."""
    fig, ax = figure(11, 8)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    _box(ax, 0.2, 6.3, 5.0, 1.1, "Medical purpose?\n(diagnose, treat, prevent disease)", BLUE)
    _box(ax, 6.6, 6.3, 4.2, 1.1, "Likely not a device\n(e.g. administrative)", MUTED)
    _arrow(ax, 5.2, 6.85, 6.6, 6.85, "no", 5.9, 7.15, GREEN)

    _box(ax, 0.2, 4.2, 5.0, 1.2, "Meets all four\nnon-device CDS criteria?", BLUE)
    _box(ax, 6.6, 4.2, 4.2, 1.2, "Non-device CDS\n(not regulated as a device)", MUTED)
    _arrow(ax, 2.7, 6.3, 2.7, 5.4, "yes", 3.1, 5.85, RED)
    _arrow(ax, 5.2, 4.8, 6.6, 4.8, "yes", 5.9, 5.1, GREEN)

    _box(ax, 0.2, 2.2, 5.0, 1.1, "Device software function\n(SaMD): pick a pathway", ORANGE,
         weight="bold")
    _arrow(ax, 2.7, 4.2, 2.7, 3.3, "no", 3.05, 3.75, RED)

    _box(ax, 0.2, 0.3, 3.3, 1.3, "510(k)\npredicate exists", ORANGE, fs=13.5)
    _box(ax, 3.8, 0.3, 3.4, 1.3, "De Novo\nnovel, low–moderate\nrisk", ORANGE, fs=13)
    _box(ax, 7.5, 0.3, 3.3, 1.3, "PMA\nhigh risk\n(Class III)", ORANGE, fs=13.5)
    _arrow(ax, 1.6, 2.2, 1.85, 1.6)
    _arrow(ax, 3.4, 2.2, 5.5, 1.6)
    _arrow(ax, 5.2, 2.5, 9.1, 1.6)

    ax.text(10.8, 2.75, "a PCCP can be\nincluded in any\nof the three", ha="right",
            va="center", fontsize=13, color=MUTED, style="italic")
    ax.set_title("Is it a device, and which pathway?", loc="left", pad=4)
    ax.text(0.2, -0.05, "simplified; real determinations depend on intended use and claims",
            fontsize=11.5, color=MUTED, style="italic", va="top")
    return save(fig, "m03-06-device-decision")
