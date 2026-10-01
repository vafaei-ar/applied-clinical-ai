"""Figures for lesson m02-01-problem-framing."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, figure, save


def fig_prediction_time():
    """Two candidate prediction times for the same readmission label, and what each can see."""
    fig, ax = figure(11, 5.0)
    ax.set_xlim(-0.3, 10.8)
    ax.set_ylim(-1.75, 2.2)
    ax.axis("off")

    ax.annotate("", xy=(10.6, 0), xytext=(-0.2, 0),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2})
    ax.text(10.6, 0.15, "time", ha="right", va="bottom", color=MUTED, fontsize=13)

    # lookback 0..4.2, index stay 4.4..6.4, outcome 6.6..9.8
    ax.add_patch(Rectangle((0, 0.3), 4.2, 0.9, color=BLUE, alpha=0.16, lw=0))
    ax.text(2.1, 0.75, "prior-year history", ha="center", va="center", fontsize=15,
            color=BLUE, weight="bold")
    ax.add_patch(Rectangle((4.4, 0.3), 2.0, 0.9, color=PURPLE, alpha=0.16, lw=0))
    ax.text(5.4, 0.75, "index stay", ha="center", va="center", fontsize=15,
            color=PURPLE, weight="bold")
    ax.add_patch(Rectangle((6.6, 0.3), 3.2, 0.9, color=GREEN, alpha=0.2, lw=0))
    ax.text(8.2, 0.75, "30-day outcome", ha="center", va="center", fontsize=15,
            color=GREEN, weight="bold")

    for x, label, color in [(4.3, "A: admission", ORANGE), (6.5, "D: discharge", INK)]:
        ax.plot([x, x], [-0.35, 1.55], color=color, lw=3)
        ax.text(x, 1.65, label, ha="center", va="bottom", fontsize=15, weight="bold",
                color=color)

    ax.text(0.0, -0.6, "Predict at A: uses history only.\nEarliest, but sees the least.",
            ha="left", va="top", fontsize=13.5, color=ORANGE)
    ax.text(5.2, -0.6, "Predict at D: adds length of stay,\nin-hospital labs, disposition.\n"
            "Matches the discharge-planning action.",
            ha="left", va="top", fontsize=13.5, color=INK)

    ax.set_title("Same label, two prediction times, different legal features", loc="left",
                 pad=4, fontsize=18)
    return save(fig, "m02-01-prediction-time")


def fig_ppv_prevalence():
    """PPV as a function of prevalence at fixed sensitivity and specificity."""
    fig, ax = figure(10, 6.5)
    prev = np.linspace(0.005, 0.30, 300)
    sens = 0.80
    for spec, color in [(0.95, BLUE), (0.80, ORANGE)]:
        ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
        ax.plot(prev * 100, ppv * 100, color=color, lw=3)
        ax.text(30.5, ppv[-1] * 100, f"specificity {spec:.2f}", color=color, fontsize=14,
                va="center", ha="left", weight="bold")

    for p in (0.02, 0.12):
        for spec, color in [(0.95, BLUE), (0.80, ORANGE)]:
            v = sens * p / (sens * p + (1 - spec) * (1 - p)) * 100
            ax.plot(p * 100, v, "o", ms=10, color=color, zorder=5,
                    markeredgecolor="white", markeredgewidth=2)
            ax.text(p * 100 + 0.7, v - 1.2, f"{v:.0f}%", fontsize=14, color=INK, va="top")

    ax.axvline(2, color=MUTED, lw=1, ls=":")
    ax.axvline(12, color=MUTED, lw=1, ls=":")
    ax.text(2.3, 96, "rare\noutcome", fontsize=12.5, color=MUTED, va="top")
    ax.text(12.3, 96, "typical stroke\nreadmission", fontsize=12.5, color=MUTED, va="top")

    ax.set_xlim(0, 38)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Prevalence (%)")
    ax.set_ylabel("PPV (%)")
    ax.set_title("Sensitivity fixed at 0.80: PPV is set by prevalence", loc="left", fontsize=18)
    return save(fig, "m02-01-ppv-prevalence")
