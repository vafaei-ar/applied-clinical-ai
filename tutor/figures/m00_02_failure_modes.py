"""Figures for lesson m00-02-failure-modes."""

from __future__ import annotations

import numpy as np

from _style import BLUE, INK, MUTED, ORANGE, figure, save


def fig_cost_proxy():
    """Simulated version of the cost-as-proxy pattern: same score, different underlying need."""
    rng = np.random.default_rng(2019)
    n = 40_000
    group_b = rng.random(n) < 0.25
    need = rng.poisson(2.2, n) + rng.poisson(0.6, n)  # chronic conditions, same distribution
    access = np.where(group_b, 0.7, 1.0)  # less spending at the same need
    cost = need * access * rng.lognormal(0, 0.45, n) + rng.lognormal(-1.2, 0.5, n)
    pct = cost.argsort().argsort() / n * 100  # algorithm "risk" percentile = cost rank

    bins = np.arange(0, 101, 5)
    mids = (bins[:-1] + bins[1:]) / 2
    fig, ax = figure(10, 6.4)
    for mask, color, label in [(~group_b, BLUE, "Group A"), (group_b, ORANGE, "Group B")]:
        means = [need[mask & (pct >= lo) & (pct < hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])]
        ax.plot(mids, means, "o-", color=color, lw=3, ms=7, label=label)

    ax.axvline(90, color=MUTED, lw=2, ls="--")
    ax.text(89, 0.6, "program\ncutoff", ha="right", va="bottom", color=MUTED, fontsize=13)
    ax.set_xlabel("Risk score percentile (model predicts cost)")
    ax.set_ylabel("Actual illness\n(chronic conditions)")
    ax.set_xlim(0, 101)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    ax.text(100, 0.15, "simulated illustration", ha="right", fontsize=11, color=MUTED,
            style="italic")
    ax.set_title("Same score, sicker patients in group B", loc="left", pad=8)
    return save(fig, "m00-02-cost-proxy")


def _normal(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def fig_shift_types():
    """Three panels: covariate shift, label shift, concept drift."""
    fig, axes = figure(12.5, 5.0)
    fig.clf()
    axes = fig.subplots(1, 3)
    x = np.linspace(20, 100, 300)

    ax = axes[0]
    ax.fill_between(x, _normal(x, 58, 11), color=BLUE, alpha=0.3, lw=0)
    ax.fill_between(x, _normal(x, 72, 10), color=ORANGE, alpha=0.3, lw=0)
    ax.plot(x, _normal(x, 58, 11), color=BLUE, lw=2.5)
    ax.plot(x, _normal(x, 72, 10), color=ORANGE, lw=2.5)
    ax.set_title("Covariate shift\nP(x) changes", fontsize=17)
    ax.set_xlabel("age")
    ax.set_yticks([])

    ax = axes[1]
    ax.bar([0, 1], [5, 15], color=[BLUE, ORANGE], width=0.6, alpha=0.85)
    ax.set_xticks([0, 1], ["train", "deploy"])
    ax.set_ylabel("% positive")
    ax.set_title("Label shift\nP(y) changes", fontsize=17)
    ax.set_ylim(0, 18)
    ax.set_yticks([0, 5, 10, 15])

    ax = axes[2]
    s = np.linspace(-4, 4, 200)
    ax.plot(s, 1 / (1 + np.exp(-1.4 * s)), color=BLUE, lw=3)
    ax.plot(s, 1 / (1 + np.exp(-1.4 * (s - 1.6))), color=ORANGE, lw=3)
    ax.set_xticks([])
    ax.set_xlabel("same features x")
    ax.set_ylabel("P(outcome | x)")
    ax.set_title("Concept drift\nP(y | x) changes", fontsize=17)

    for a in axes:
        a.tick_params(labelsize=13)
        a.xaxis.label.set_size(14)
        a.yaxis.label.set_size(14)
    fig.suptitle("blue = training data      orange = deployment data", fontsize=15,
                 color=INK, y=0.99)
    return save(fig, "m00-02-shift-types")

