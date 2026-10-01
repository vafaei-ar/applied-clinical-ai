"""Figures for lesson m09-01-mechanisms."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, save

SCENARIOS = [
    "MCAR",
    "MAR on age (Z)",
    "MAR on outcome (Y)",
    "MNAR on X itself",
    "MNAR on unmeasured U",
]


def _ols(y, cols):
    a = np.column_stack([np.ones(len(y))] + cols)
    return np.linalg.lstsq(a, y, rcond=None)[0]


def _miss_prob(linear_part, target):
    """Logistic missingness probability whose average equals `target`."""
    lo, hi = -12.0, 12.0
    for _ in range(50):
        mid = (lo + hi) / 2
        avg = (1 / (1 + np.exp(-(mid + linear_part)))).mean()
        if avg > target:
            hi = mid
        else:
            lo = mid
    return 1 / (1 + np.exp(-(lo + linear_part)))


def simulate_cca_bias(n=4000, reps=300, target_missing=0.40, strength=1.0, seed=20260930):
    """Complete-case bias for the mean of X and the slope of Y on X (adjusted for Z).

    Data: Z (age, standardised) and U (unmeasured frailty) are independent N(0,1);
    X (an LDL-like lab, variance 1) depends on both; Y is a continuous outcome that
    depends on X, Z and U. Only X is ever missing (about 40% of patients).
    Bias is measured against the estimate from the same data with no values missing.
    Returns {scenario: (bias_mean_x, bias_slope, observed_missing_share)}.
    """
    rng = np.random.default_rng(seed)
    acc = {name: [] for name in SCENARIOS}
    for _ in range(reps):
        z = rng.normal(size=n)
        u = rng.normal(size=n)
        x = 0.5 * z + 0.5 * u + np.sqrt(0.5) * rng.normal(size=n)
        y = 1.0 * x + 0.5 * z + 0.8 * u + rng.normal(size=n)
        y_std = (y - y.mean()) / y.std()
        full_slope = _ols(y, [x, z])[1]
        probs = {
            "MCAR": np.full(n, target_missing),
            "MAR on age (Z)": _miss_prob(strength * z, target_missing),
            "MAR on outcome (Y)": _miss_prob(strength * y_std, target_missing),
            "MNAR on X itself": _miss_prob(strength * x, target_missing),
            "MNAR on unmeasured U": _miss_prob(strength * u, target_missing),
        }
        for name, p in probs.items():
            missing = rng.random(n) < p
            seen = ~missing
            acc[name].append(
                (
                    x[seen].mean() - x.mean(),
                    _ols(y[seen], [x[seen], z[seen]])[1] - full_slope,
                    missing.mean(),
                )
            )
    return {k: tuple(np.mean(v, axis=0)) for k, v in acc.items()}


def fig_cca_bias():
    """Bias of complete-case estimates under five missingness mechanisms."""
    res = simulate_cca_bias()
    names = SCENARIOS
    mean_bias = np.array([res[k][0] for k in names])
    slope_bias = np.array([res[k][1] for k in names])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True)
    ypos = np.arange(len(names))[::-1]
    for ax, vals, title, xlim in (
        (ax1, mean_bias, "Mean of X (LDL)\nbias, in SD units", (-0.62, 0.12)),
        (ax2, slope_bias, "Slope of Y on X\nbias (true slope = 1.0)", (-0.15, 0.06)),
    ):
        colors = [GREEN if abs(v) < 0.01 else RED for v in vals]
        ax.barh(ypos, vals, color=colors, height=0.62)
        ax.axvline(0, color=INK, lw=1.5)
        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=16, loc="left")
        ax.tick_params(axis="x", labelsize=13)
        for y_i, v in zip(ypos, vals, strict=False):
            if abs(v) < 0.005:
                ax.text(0.006 * (xlim[1] - xlim[0]) / 0.18, y_i, "0.00", va="center",
                        ha="left", fontsize=14, color=GREEN, weight="bold")
            else:
                ax.text(v - 0.012 * (xlim[1] - xlim[0]) / 0.18, y_i, f"{v:+.2f}", va="center",
                        ha="right", fontsize=14, color=RED, weight="bold")
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(names, fontsize=14)
    fig.text(
        0.01, 0.005,
        "Simulation: 4,000 patients x 300 runs, 40% of X missing, complete-case analysis. "
        "Bias is measured against the same data with nothing missing.",
        fontsize=10.5, color=MUTED, style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "m09-01-cca-bias")


def fig_repo_missingness():
    """Two different kinds of missing in the repo, with the measured numbers."""
    # Measured on the seed-20260908 database (5,000 patients): labs table, and
    # stroke_features (333 cohort patients).
    lab_items = [
        ("creatinine", 2.7),
        ("hemoglobin", 3.1),
        ("HbA1c", 2.9),
        ("LDL", 3.0),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    labels = [n for n, _ in lab_items] + ["Cohort patients with\nno usable LDL in prior year"]
    vals = [v for _, v in lab_items] + [82.6]
    ypos = [5.0, 4.2, 3.4, 2.6, 0.7]
    colors = [BLUE] * 4 + [ORANGE]
    ax.barh(ypos, vals, color=colors, height=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_xlabel("percent missing", fontsize=15)
    for y_i, v in zip(ypos, vals, strict=False):
        ax.text(v + 1.2, y_i, f"{v:.1f}%", va="center", fontsize=15, weight="bold")
    ax.text(30, 3.8, "lab result value is blank\n(a flat 3% draw per result)",
            fontsize=14, color=BLUE, va="center")
    ax.text(30, 1.3, "no LDL result dated in the 365 days\nbefore the index date (275 of 333)",
            fontsize=14, color=ORANGE, va="center")
    ax.set_title("Two kinds of missing, and both are pure chance here", loc="left", fontsize=18)
    fig.text(0.01, 0.005, "Measured on the repo's synthetic database (seed 20260908, 5,000 patients).",
             fontsize=10.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "m09-01-repo-missingness")
