"""Figures for lesson m02-06-subgroups-shift."""

from __future__ import annotations

import numpy as np

from _style import BLUE, INK, MUTED, ORANGE, figure, save


def _auroc(y, s):
    """Mann-Whitney AUROC (ties are rare with continuous scores)."""
    ranks = np.empty(len(s))
    ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _boot_ci(y, s, rng, n_boot=1000):
    stats = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if 0 < yy.sum() < n:
            stats.append(_auroc(yy, s[idx]))
    return np.percentile(stats, [2.5, 97.5])


def fig_subgroup_forest():
    """Subgroup AUROC with bootstrap 95% CIs; small groups get wide intervals."""
    rng = np.random.default_rng(2024)
    n = 4000
    sex = rng.choice(["Female", "Male"], n, p=[0.51, 0.49])
    race = rng.choice(["White", "Black", "Hispanic", "Asian", "Other"], n,
                      p=[0.58, 0.15, 0.16, 0.07, 0.04])
    plan = rng.choice(["Commercial", "Medicare", "Medicaid"], n, p=[0.32, 0.50, 0.18])
    x = rng.normal(0, 1, n)                     # true risk factor
    base = -2.0 + 0.35 * (plan == "Medicaid")
    y = rng.binomial(1, 1 / (1 + np.exp(-(base + 1.1 * x))))
    # The model sees the risk factor less clearly for Medicaid patients
    # (for example, fragmented care means missing history).
    noise_sd = np.where(plan == "Medicaid", 1.4, 0.5)
    score = x + rng.normal(0, 1, n) * noise_sd

    masks = [("Overall", np.ones(n, bool))]
    masks += [(g, sex == g) for g in ["Female", "Male"]]
    masks += [(g, race == g) for g in ["White", "Black", "Hispanic", "Asian", "Other"]]
    masks += [(g, plan == g) for g in ["Commercial", "Medicare", "Medicaid"]]
    rows = []
    for label, m in masks:
        lo, hi = _boot_ci(y[m], score[m], rng)
        rows.append((label, int(m.sum()), int(y[m].sum()), _auroc(y[m], score[m]), lo, hi))

    fig, ax = figure(10, 9)
    overall = rows[0][3]
    ax.axvline(overall, color=MUTED, ls="--", lw=1.8)
    labels = []
    for i, (label, n, events, est, lo, hi) in enumerate(rows):
        yy = len(rows) - 1 - i
        color = INK if i == 0 else (ORANGE if label == "Medicaid" else BLUE)
        ax.plot([lo, hi], [yy, yy], color=color, lw=3.5, solid_capstyle="round")
        ax.plot(est, yy, "s" if i == 0 else "o", color=color, ms=11)
        ax.text(1.02, yy, f"{events}", ha="left", va="center", fontsize=14,
                color=MUTED, transform=ax.get_yaxis_transform())
        labels.append(label)
    ax.text(1.02, len(rows) - 0.35, "events", ha="left", va="bottom", fontsize=13,
            color=MUTED, transform=ax.get_yaxis_transform())

    ax.set_yticks(range(len(rows) - 1, -1, -1))
    ax.set_yticklabels(labels, fontsize=15)
    ax.get_yticklabels()[0].set_weight("bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.5, 0.95)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("AUROC with bootstrap 95% CI  (simulated)")
    ax.set_title("Subgroup AUROC: read the widths", loc="left")
    return save(fig, "m02-06-subgroup-forest")
