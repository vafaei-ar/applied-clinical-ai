"""Figures for lesson m02-05-calibration."""

from __future__ import annotations

import numpy as np

from _style import BLUE, INK, MUTED, ORANGE, RED, figure, save


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _binned(p, y, n_bins=10):
    """Equal-mass bins: mean predicted vs observed rate."""
    order = np.argsort(p)
    groups = np.array_split(order, n_bins)
    return (np.array([p[g].mean() for g in groups]),
            np.array([y[g].mean() for g in groups]))


def _simulate(n=20000, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.normal(-2.3, 1.0, n)          # true logit, prevalence about 13%
    p_true = _sigmoid(z)
    y = rng.binomial(1, p_true)
    return z, p_true, y


def fig_reliability_diagram():
    """Calibrated vs overconfident model on the same simulated readmission cohort."""
    z, p_true, y = _simulate()
    center = -2.3
    p_over = _sigmoid(center + 1.8 * (z - center))   # too extreme: calibration slope ~0.56

    fig, ax = figure(8.6, 8.4)
    ax.plot([0, 0.7], [0, 0.7], color=MUTED, lw=2, ls="--")
    ax.text(0.53, 0.575, "perfect", color=MUTED, fontsize=14, rotation=40)

    for p, color, label in [(p_true, BLUE, "calibrated"),
                            (p_over, ORANGE, "overconfident")]:
        mp, obs = _binned(p, y)
        ax.plot(mp, obs, "-o", color=color, lw=3, ms=9, label=label)

    ax.annotate("high risks\ntoo high", xy=(0.655, 0.37), xytext=(0.47, 0.47),
                fontsize=14, color=ORANGE,
                arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.8})
    ax.annotate("low risks\ntoo low", xy=(0.018, 0.05), xytext=(0.02, 0.24),
                fontsize=14, color=ORANGE,
                arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.8})

    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted risk (decile mean)")
    ax.set_ylabel("Observed readmission rate")
    ax.legend(loc="upper left")
    ax.set_title("Same ranking, different honesty", loc="left")
    ax.text(0.69, 0.01, "simulated, n = 20,000\n10 equal-size bins", ha="right",
            va="bottom", fontsize=12, color=MUTED, style="italic")
    return save(fig, "m02-05-reliability-diagram")


def fig_prevalence_shift():
    """A model calibrated at site A applied at site B where readmission is rarer."""
    rng = np.random.default_rng(11)
    n = 20000
    x = rng.normal(0, 1.0, n)
    # Same risk-factor effect at both sites; site B has a lower baseline risk.
    logit_a = -1.9 + 1.0 * x
    logit_b = -2.7 + 1.0 * x
    y_a = rng.binomial(1, _sigmoid(logit_a))
    y_b = rng.binomial(1, _sigmoid(logit_b))
    pred = _sigmoid(logit_a)     # model learned at site A, used everywhere

    fig, ax = figure(8.6, 8.4)
    ax.plot([0, 0.6], [0, 0.6], color=MUTED, lw=2, ls="--")
    for yy, color, label in [(y_a, BLUE, f"site A (prevalence {y_a.mean():.0%})"),
                             (y_b, RED, f"site B (prevalence {y_b.mean():.0%})")]:
        mp, obs = _binned(pred, yy)
        ax.plot(mp, obs, "-o", color=color, lw=3, ms=9, label=label)

    ax.text(0.28, 0.06, "same ranking at site B,\nbut every risk is\noverestimated",
            fontsize=15, color=RED, ha="left")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted risk (model from site A)")
    ax.set_ylabel("Observed readmission rate")
    ax.legend(loc="upper left")
    ax.set_title("Prevalence shift breaks calibration", loc="left", color=INK)
    return save(fig, "m02-05-prevalence-shift")
