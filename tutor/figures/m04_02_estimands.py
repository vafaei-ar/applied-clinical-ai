"""Figures for lesson m04-02-estimands."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, RED, figure, save


def fig_estimand_attributes():
    """The five ICH E9(R1) attributes, filled in for the synthetic stroke trial."""
    fig, ax = figure(8.2, 11)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    cards = [
        ("Population", BLUE,
         "Adults within 7 days of ischemic stroke\nor high-risk TIA, no indication\nfor anticoagulation"),
        ("Treatment", GREEN,
         "Zetagrel + aspirin vs aspirin alone,\nstarted within 7 days"),
        ("Variable (endpoint)", ORANGE,
         "Time from randomization to first\nrecurrent stroke, over all follow-up"),
        ("Intercurrent events", RED,
         "Bleeding, drug stopped: treatment policy\nNew AF, anticoagulant: hypothetical\nNon-stroke death: while alive"),
        ("Population-level summary", PURPLE,
         "Hazard ratio, plus 12-month\nrisk difference from KM"),
    ]
    top, h, gap = 10.3, 1.78, 0.25
    for i, (name, color, body) in enumerate(cards):
        y = top - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((0.15, y), 9.7, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    fc=color, alpha=0.11, ec=color, lw=2))
        ax.text(0.45, y + h - 0.28, name, fontsize=18, weight="bold", color=color, va="top")
        ax.text(0.45, y + h - 0.78, body, fontsize=15.5, color=INK, va="top", linespacing=1.3)

    ax.text(0.15, 10.55, "The question comes before the analysis", fontsize=20, weight="bold",
            va="bottom")
    ax.text(0.15, 0.05, "synthetic trial; zetagrel is a fictional drug", fontsize=11.5,
            color=MUTED, style="italic")
    return save(fig, "m04-02-estimand-attributes")
