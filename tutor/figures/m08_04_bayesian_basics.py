"""Figures for lesson m08-04-bayesian-basics (all data simulated with numpy)."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, figure, save

_lgamma = np.frompyfunc(math.lgamma, 1, 1)


def _lg(x):
    return np.asarray(_lgamma(x), dtype=float)


def _beta_pdf(x, a, b):
    return np.exp(_lg(a + b) - _lg(a) - _lg(b) + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x))


def fig_beta_update():
    """One prior, two datasets: the prior dominates when data are sparse."""
    x = np.linspace(0.002, 0.6, 600)
    prior = (4, 36)                                   # mean 0.10, worth 40 patients
    cases = [(3, 12, "12 patients, 3 readmitted"), (60, 240, "240 patients, 60 readmitted")]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for ax, (k, n, title) in zip(axes, cases):
        flat = (1 + k, 1 + n - k)
        informed = (prior[0] + k, prior[1] + n - k)
        ax.plot(x, _beta_pdf(x, *prior), color=MUTED, lw=2.5, ls="--")
        ax.plot(x, _beta_pdf(x, *flat), color=BLUE, lw=3.5)
        ax.plot(x, _beta_pdf(x, *informed), color=ORANGE, lw=3.5)
        ax.fill_between(x, _beta_pdf(x, *informed), color=ORANGE, alpha=0.15, lw=0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=17)
        mf, mi = flat[0] / sum(flat), informed[0] / sum(informed)
        top = max(_beta_pdf(x, *flat).max(), _beta_pdf(x, *informed).max(), _beta_pdf(x, *prior).max())
        ax.set_ylim(0, top * 1.22)
        ax.text(0.98, 0.93, f"posterior, flat prior: mean {mf:.2f}", transform=ax.transAxes, ha="right",
                va="top", color=BLUE, fontsize=15, weight="bold")
        ax.text(0.98, 0.80, f"posterior, Beta(4, 36) prior: mean {mi:.2f}", transform=ax.transAxes,
                ha="right", va="top", color=ORANGE, fontsize=15, weight="bold")
    axes[1].text(0.078, axes[1].get_ylim()[1] * 0.70, "prior\n(mean 0.10)", color=MUTED, fontsize=14,
                 weight="bold", ha="center", va="bottom")
    axes[1].set_xlabel("30-day readmission rate at one hospital")
    axes[1].set_xlim(0, 0.6)
    return save(fig, "m08-04-beta-update")


def _bb_loglik(a, b, k, n):
    return np.sum(_lg(n + 1) - _lg(k + 1) - _lg(n - k + 1) + _lg(k + a) + _lg(n - k + b)
                  - _lg(n + a + b) + _lg(a + b) - _lg(a) - _lg(b))


def _fit_beta_prior(k, n):
    """Empirical Bayes: maximize the beta-binomial marginal likelihood over a grid."""
    best, best_ab = -np.inf, None
    for mean in np.linspace(0.03, 0.30, 55):
        for kappa in np.exp(np.linspace(np.log(4), np.log(3000), 45)):
            ll = _bb_loglik(mean * kappa, (1 - mean) * kappa, k, n)
            if ll > best:
                best, best_ab = ll, (mean * kappa, (1 - mean) * kappa)
    return best_ab


def fig_partial_pooling():
    """Twelve hospitals of very different size: raw rates versus partially pooled estimates."""
    sizes = np.array([600, 400, 250, 150, 90, 60, 45, 30, 20, 15, 12, 8])
    rng = np.random.default_rng(3)
    true = 1 / (1 + np.exp(-(np.log(0.10 / 0.90) + 0.35 * rng.normal(size=len(sizes)))))
    events = rng.binomial(sizes, true)
    a, b = _fit_beta_prior(events, sizes)
    raw = events / sizes
    pooled = (a + events) / (a + b + sizes)

    fig, ax = figure(10, 8.2)
    ys = np.arange(len(sizes))[::-1]
    for y, r, p in zip(ys, raw, pooled):
        ax.annotate("", xy=(p, y), xytext=(r, y),
                    arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.8,
                                "shrinkA": 7, "shrinkB": 8})
    ax.plot(raw, ys, "o", mfc="white", mec=BLUE, mew=2.8, ms=13, label="Raw rate", zorder=4)
    ax.plot(pooled, ys, "o", color=ORANGE, ms=12, label="Partially pooled", zorder=5)
    ax.plot(true, ys, "D", color=INK, ms=7, label="True rate (known here)", zorder=6)
    ax.axvline(a / (a + b), color=MUTED, ls="--", lw=1.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"n = {n}" for n in sizes], fontsize=15)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.02, 0.32)
    ax.set_ylim(-0.7, len(sizes) - 0.3)
    ax.set_xlabel("30-day readmission rate (simulated)")
    ax.legend(loc="upper right", fontsize=13, bbox_to_anchor=(1.0, 1.02))
    ax.set_title("Small hospitals get pulled toward the average", loc="left")
    return save(fig, "m08-04-partial-pooling")
