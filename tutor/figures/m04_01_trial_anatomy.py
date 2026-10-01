"""Figures for lesson m04-01-trial-anatomy."""

from __future__ import annotations

import math

import numpy as np

from _style import BLUE, GRID, INK, MUTED, ORANGE, figure, save

Z_ALPHA = 1.959964  # two-sided alpha = 0.05
Z_POWER = {0.80: 0.841621, 0.90: 1.281552}


def events_needed(hr: float, power: float) -> float:
    """Schoenfeld approximation for 1:1 allocation: D = 4 (z_a + z_b)^2 / (log HR)^2."""
    return 4 * (Z_ALPHA + Z_POWER[power]) ** 2 / math.log(hr) ** 2


def fig_events_needed():
    """Events needed vs the true hazard ratio, for 80% and 90% power."""
    fig, ax = figure(10, 6.5)
    hrs = np.linspace(0.60, 0.90, 200)
    for power, color in [(0.90, BLUE), (0.80, ORANGE)]:
        d = [events_needed(h, power) for h in hrs]
        ax.plot(hrs, d, color=color, lw=3.5, label=f"{int(power * 100)}% power")

    for hr in (0.75, 0.85):
        d = events_needed(hr, 0.90)
        ax.plot(hr, d, "o", ms=11, color=BLUE, zorder=5)
        ax.annotate(f"HR {hr}: ~{round(d):,} events", xy=(hr, d),
                    xytext=(hr - 0.115, d + 250), fontsize=15, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1.5})

    ax.set_xlim(0.60, 0.90)
    ax.set_ylim(0, 3000)
    ax.set_xlabel("True hazard ratio (treatment vs control)")
    ax.set_ylabel("Primary-endpoint events needed")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    ax.text(0.60, -560, "Two-sided alpha 0.05, 1:1 allocation (Schoenfeld approximation)",
            fontsize=12, color=MUTED, style="italic")
    ax.set_title("Smaller effects need far more events", loc="left", pad=8)
    return save(fig, "m04-01-events-needed")
