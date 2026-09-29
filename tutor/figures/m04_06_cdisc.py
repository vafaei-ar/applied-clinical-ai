"""Figures for lesson m04-06-cdisc."""

from __future__ import annotations

from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, figure, save


def _box(ax, x, y, w, h, color, title, body):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=to_rgba(color, 0.12), edgecolor=color, lw=2.5))
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=19,
            weight="bold", color=color)
    ax.text(x + w / 2, y + h - 0.95, body, ha="center", va="top", fontsize=15,
            color=INK, linespacing=1.4)


def _arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=26,
                                 color=INK, lw=2.2))


def fig_cdisc_flow():
    """CDASH collection -> SDTM tabulation -> ADaM analysis -> TLFs, with define.xml below."""
    fig, ax = figure(12, 6.2)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    top, h, w = 2.35, 3.45, 2.6
    xs = [0.05, 3.15, 6.25, 9.35]
    _box(ax, xs[0], top, w, h, ORANGE, "CDASH",
         "CRF fields\nin the EDC\n\nhow data is\ncollected")
    _box(ax, xs[1], top, w, h, BLUE, "SDTM",
         "DM  AE  LB\nVS  EX  DS\nCM ...\n\nwhat was collected")
    _box(ax, xs[2], top, w, h, GREEN, "ADaM",
         "ADSL\nADTTE  ADLB\nADAE ...\n\nready to analyse")
    _box(ax, xs[3], top, w, h, PURPLE, "TLFs",
         "tables\nlistings\nfigures\n\nthe CSR")
    for i in range(3):
        _arrow(ax, xs[i] + w + 0.04, xs[i + 1] - 0.04, top + h / 2)

    ax.add_patch(FancyBboxPatch((xs[1], 1.2), xs[2] + w - xs[1], 0.85,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="white", edgecolor=MUTED, lw=2, ls="--"))
    ax.text((xs[1] + xs[2] + w) / 2, 1.625, "define.xml: metadata for every dataset,\n"
            "variable, codelist, and derivation", ha="center", va="center", fontsize=14,
            color=INK)
    ax.add_patch(FancyArrowPatch((xs[3] + w, 0.45), (xs[0], 0.45), arrowstyle="-|>",
                                 mutation_scale=24, color=MUTED, lw=2))
    ax.text(6.0, 0.62, "traceability: every result traces back to the collected data",
            ha="center", va="bottom", fontsize=14, color=MUTED, style="italic")
    ax.set_title("From case report form to submission", loc="left", pad=6)
    return save(fig, "m04-06-cdisc-flow")


if __name__ == "__main__":
    print(fig_cdisc_flow())
