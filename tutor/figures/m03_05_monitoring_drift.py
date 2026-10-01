"""Figures for lesson m03-05-monitoring-drift."""

from __future__ import annotations

import numpy as np

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, figure, save


def psi(reference, current, n_bins=10, eps=1e-4, edges=None):
    """Population stability index with quantile bins from the reference sample."""
    if edges is None:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, eps, None), np.clip(cur, eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _ages(rng, n, shift=0.0):
    # Synthetic stroke-cohort ages: skewed toward older adults, clipped to 18-100
    return np.clip(rng.normal(70 + shift, 12, n), 18, 100)


def fig_shifted_distributions():
    """Reference vs current distribution of predicted risk, with PSI."""
    rng = np.random.default_rng(3)
    ref = rng.beta(2.0, 11.0, 20000)
    cur = rng.beta(2.0, 16.0, 3000)  # predictions shifted lower after an upstream change
    value = psi(ref, cur)

    fig, ax = figure(10, 6.2)
    bins = np.linspace(0, 0.6, 40)
    ax.hist(ref, bins=bins, density=True, color=BLUE, alpha=0.35, label="training reference")
    ax.hist(cur, bins=bins, density=True, histtype="step", lw=3, color=ORANGE,
            label="this month")
    ax.set_xlabel("predicted 30-day readmission risk")
    ax.set_ylabel("density")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    ax.text(0.33, ax.get_ylim()[1] * 0.55, f"PSI = {value:.2f}", fontsize=20, weight="bold",
            color=RED)
    ax.set_title("Prediction drift: the whole distribution moved", loc="left", pad=8)
    return save(fig, "m03-05-shifted-distributions")


def fig_psi_over_time():
    """Weekly PSI for one feature: noise, then an upstream coding change."""
    rng = np.random.default_rng(11)
    ref = rng.binomial(1, 0.62, 20000).astype(float)  # prior_hypertension prevalence
    weeks = np.arange(1, 27)
    values = []
    for w in weeks:
        n = int(rng.integers(160, 240))
        p = 0.62 if w < 15 else 0.30  # site switches coding in week 15
        values.append(psi(ref, rng.binomial(1, p, n).astype(float),
                          edges=np.array([-np.inf, 0.5, np.inf])))
    values = np.array(values)

    fig, ax = figure(10.5, 6.2)
    ax.axhspan(0, 0.1, color=GREEN, alpha=0.08)
    ax.axhspan(0.1, 0.25, color=ORANGE, alpha=0.08)
    ax.axhspan(0.25, 0.7, color=RED, alpha=0.06)
    ax.plot(weeks, values, "o-", color=INK, lw=2.5, ms=7)
    ax.axvline(14.5, color=MUTED, ls="--", lw=2)
    ax.text(14.2, 0.66, "coding change\nat one site", ha="right", va="top", fontsize=14,
            color=MUTED)
    ax.text(26.4, 0.05, "< 0.1 stable", ha="right", va="center", fontsize=13, color=GREEN)
    ax.text(26.4, 0.175, "0.1–0.25 watch", ha="right", va="center", fontsize=13, color=ORANGE)
    ax.text(1.0, 0.31, "> 0.25 investigate", ha="left", va="center", fontsize=13, color=RED)
    ax.set_ylim(0, 0.7)
    ax.set_xlim(0.5, 26.8)
    ax.set_xlabel("week since go-live")
    ax.set_ylabel("PSI, prior_hypertension\n(one site)")
    ax.set_title("Weekly PSI: noise, then a real shift", loc="left", pad=8)
    return save(fig, "m03-05-psi-over-time")
