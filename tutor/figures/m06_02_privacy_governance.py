"""Figures for lesson m06-02-privacy-governance."""

from __future__ import annotations

from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch

from _style import GREEN, INK, MUTED, ORANGE, RED, figure, save

COLUMNS = [
    ("Identified PHI", RED),
    ("Limited data set", ORANGE),
    ("De-identified\n(Safe Harbor)", GREEN),
]
ROWS = [
    ("Names, MRN,\nSSN, address", ["kept", "removed", "removed"]),
    ("Dates", ["full dates", "full dates\n(birth, admit, death)", "year only;\nages 90+ → '90+'"]),
    ("ZIP code", ["full address", "5-digit ZIP,\ncity, state", "3 digits if >20,000\npeople, else 000"]),
    ("To use it for\nresearch", ["authorization or\nIRB waiver", "data use\nagreement", "not PHI, but\nre-ID risk remains"]),
]


def fig_data_tiers():
    """What each HIPAA data tier keeps, and the paperwork it needs (simplified)."""
    fig, ax = figure(11, 7.6)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    x0, col_w, row_h = 2.3, 2.85, 1.45
    top = 6.4
    for j, (name, color) in enumerate(COLUMNS):
        x = x0 + j * col_w
        ax.add_patch(FancyBboxPatch((x + 0.06, top), col_w - 0.12, 0.95,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=to_rgba(color, 0.16), edgecolor=color, lw=2.2))
        ax.text(x + col_w / 2, top + 0.475, name, ha="center", va="center", fontsize=15,
                weight="bold", color=color)

    for i, (label, cells) in enumerate(ROWS):
        y = top - 0.15 - (i + 1) * row_h
        ax.plot([0.1, x0 + 3 * col_w], [y, y], color=MUTED, lw=0.8, alpha=0.6)
        ax.text(0.1, y + row_h / 2, label, ha="left", va="center", fontsize=14,
                weight="bold", color=INK)
        for j, cell in enumerate(cells):
            ax.text(x0 + j * col_w + col_w / 2, y + row_h / 2, cell, ha="center",
                    va="center", fontsize=13.5, color=INK, linespacing=1.3)

    ax.text(0.1, 0.1, "simplified summary · not legal advice", fontsize=12, color=MUTED,
            style="italic")
    ax.set_title("Three tiers of health data under HIPAA", loc="left", pad=6)
    return save(fig, "m06-02-data-tiers")


if __name__ == "__main__":
    print(fig_data_tiers())
