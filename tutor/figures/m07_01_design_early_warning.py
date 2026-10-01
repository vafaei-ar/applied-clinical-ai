"""Figures for lesson m07-01-design-early-warning."""

from __future__ import annotations

from math import erf, sqrt

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, RED, figure, save

# Alert-budget scenario (shared with the lesson text): 30-bed ward, 12-hour shifts,
# 2% of patient-shifts are followed by sepsis onset, binormal scores with d' = 1.4.
BEDS = 30
PREVALENCE = 0.02
D_PRIME = 1.4
BUDGET = 3


def _phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def _box(ax, x, y, w, h, color, title, body, dashed=False):
    style = "round,pad=0.02,rounding_size=0.12"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=color, alpha=0.13, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor="none", edgecolor=color,
                                lw=2, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=15.5, weight="bold",
            color=color)
    ax.text(x + w / 2, y + 0.25, body, ha="center", va="bottom", fontsize=12.5, color=INK,
            linespacing=1.15)


def _arrow(ax, a, b, color=INK, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=20, color=color, lw=2.2,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}"))


def fig_architecture():
    """Data path, action path, and the monitoring/learning loop for a sepsis early warning."""
    fig, ax = figure(12, 6.4)
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    w, h = 2.7, 1.65
    xs = [0.1, 3.2, 6.3, 9.4]
    top, bot = 4.3, 1.1
    _box(ax, xs[0], top, w, h, BLUE, "EHR", "vitals, labs,\norders, notes")
    _box(ax, xs[1], top, w, h, BLUE, "Feed", "HL7v2 / FHIR\nnear real time")
    _box(ax, xs[2], top, w, h, BLUE, "Feature state", "as-of values,\ntrends, missingness")
    _box(ax, xs[3], top, w, h, GREEN, "Risk model", "scores every\nhour per patient")
    _box(ax, xs[3], bot, w, h, ORANGE, "Alert logic", "threshold, snooze,\nskip if treated")
    _box(ax, xs[2], bot, w, h, ORANGE, "Nurse / RRT", "worklist +\npage, with reasons")
    _box(ax, xs[1], bot, w, h, ORANGE, "Action", "assess, cultures,\nlactate, antibiotics")
    _box(ax, xs[0], bot, w, h, PURPLE, "Monitor + learn", "alert rate, PPV,\ndrift, feed outages",
         dashed=True)

    mid_top, mid_bot = top + h / 2, bot + h / 2
    for i in range(3):
        _arrow(ax, (xs[i] + w + 0.03, mid_top), (xs[i + 1] - 0.03, mid_top))
    _arrow(ax, (xs[3] + w / 2, top - 0.03), (xs[3] + w / 2, bot + h + 0.03))
    for i in (3, 2, 1):
        _arrow(ax, (xs[i] - 0.03, mid_bot), (xs[i - 1] + w + 0.03, mid_bot))
    _arrow(ax, (xs[0] + w / 2 + 0.4, bot + h + 0.05), (xs[3] + 0.3, top - 0.05), color=PURPLE,
           ls="--")
    ax.text(5.2, 3.45, "recalibrate / retrain", color=PURPLE, fontsize=13, ha="center",
            rotation=9, style="italic")

    ax.text(0.1, 0.35, "Blue: data path    Green: model    Orange: workflow    Purple: feedback",
            fontsize=12.5, color=MUTED)
    ax.set_title("Sepsis early warning: the whole system", loc="left", pad=6)
    return save(fig, "m07-01-architecture")


def alert_budget_numbers(alerts_per_shift: float):
    """Sensitivity and PPV at a given alert volume under the lesson's scenario."""
    fpr_grid = np.linspace(1e-4, 0.6, 4000)
    z = np.array([_inv_phi(1 - f) for f in fpr_grid])
    tpr = np.array([_phi(D_PRIME - zi) for zi in z])
    alerts = BEDS * (PREVALENCE * tpr + (1 - PREVALENCE) * fpr_grid)
    i = int(np.argmin(np.abs(alerts - alerts_per_shift)))
    ppv = BEDS * PREVALENCE * tpr[i] / alerts[i]
    return tpr[i], ppv


def _inv_phi(p):
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fig_alert_budget():
    """Sensitivity vs alerts per shift, with the budget line and PPV at a few points."""
    fpr = np.linspace(1e-4, 0.5, 600)
    tpr = np.array([_phi(D_PRIME - _inv_phi(1 - f)) for f in fpr])
    alerts = BEDS * (PREVALENCE * tpr + (1 - PREVALENCE) * fpr)

    fig, ax = figure(10, 6.4)
    ax.plot(alerts, tpr * 100, color=BLUE, lw=3.5)
    ax.axvline(BUDGET, color=ORANGE, lw=2.5, ls="--")
    ax.text(BUDGET + 0.15, 8, f"alert budget:\n{BUDGET} per shift", color=ORANGE, fontsize=14,
            va="bottom")
    for a in (1, BUDGET, 8):
        sens, ppv = alert_budget_numbers(a)
        ax.plot(a, sens * 100, "o", ms=11, color=RED if a == BUDGET else INK, zorder=5)
        ax.text(a + 0.2, sens * 100 - 3, f"sens {sens:.0%}\nPPV {ppv:.0%}", fontsize=13,
                va="top", color=RED if a == BUDGET else INK)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Alerts per 30-bed ward per 12-h shift")
    ax.set_ylabel("Sepsis cases flagged (%)")
    ax.text(12, 1, "simulated: 2% of patient-shifts, AUROC ≈ 0.84", ha="right", fontsize=11,
            color=MUTED, style="italic")
    ax.set_title("Pick the threshold from the alert budget", loc="left", pad=8)
    return save(fig, "m07-01-alert-budget")


if __name__ == "__main__":
    for a in (1, 2, 3, 5, 8):
        print(a, alert_budget_numbers(a))
    print("AUROC", _phi(D_PRIME / sqrt(2)))
