"""Figures for lesson m05-04-generation-eval."""

from __future__ import annotations

import numpy as np

from _style import BLUE, GRID, INK, MUTED, ORANGE, figure, save


def _risk_coverage(conf, correct):
    order = np.argsort(-conf)
    c = correct[order]
    n = np.arange(1, len(c) + 1)
    coverage = n / len(c)
    risk = np.cumsum(1 - c) / n
    return coverage, risk


def fig_risk_coverage():
    """Risk (error rate among answered) vs coverage for an informative and an uninformative score."""
    rng = np.random.default_rng(7)
    n = 2000
    # Latent difficulty drives both correctness and (noisily) confidence.
    z = rng.normal(size=n)
    p_correct = 1 / (1 + np.exp(-(1.9 + 1.6 * z)))
    correct = (rng.random(n) < p_correct).astype(float)
    good_conf = z + rng.normal(scale=0.6, size=n)
    random_conf = rng.random(n)

    fig, ax = figure(10, 6.6)
    cov, risk = _risk_coverage(good_conf, correct)
    cov_r, risk_r = _risk_coverage(random_conf, correct)
    keep = cov >= 0.05
    ax.plot(cov_r[keep] * 100, risk_r[keep] * 100, color=MUTED, lw=3,
            label="random confidence")
    ax.plot(cov[keep] * 100, risk[keep] * 100, color=BLUE, lw=3.5,
            label="informative confidence")

    full_risk = risk[-1] * 100
    i70 = int(0.70 * n) - 1
    ax.plot(70, risk[i70] * 100, "o", ms=14, color=ORANGE, zorder=5)
    ax.annotate(f"answer 70%, abstain 30%\nerror {risk[i70] * 100:.0f}% among answered",
                xy=(70, risk[i70] * 100), xytext=(14, 15.5), fontsize=14,
                color=ORANGE, weight="bold",
                arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1.5})
    ax.text(99, full_risk + 0.6, f"answer everything:\n{full_risk:.0f}% error", ha="right",
            va="bottom", fontsize=13, color=INK)

    ax.set_xlabel("coverage: % of questions answered")
    ax.set_ylabel("risk: % wrong among answered")
    ax.set_xlim(0, 102)
    ax.set_ylim(0, full_risk + 8)
    ax.grid(True, axis="y", color=GRID)
    ax.legend(loc="lower right")
    ax.set_title("Abstaining buys accuracy: risk vs coverage", loc="left", pad=10)
    ax.text(101, -0.17 * (full_risk + 8), "simulated data", fontsize=11, color=MUTED,
            style="italic", ha="right")
    return save(fig, "m05-04-risk-coverage")
