"""Figures for lesson m02-04-metrics-imbalance (all data simulated, fixed seed)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import BLUE, GREEN, INK, MUTED, ORANGE, figure, save


def _curves(scores, y):
    """Empirical ROC and PR points, sorted by descending score."""
    order = np.argsort(-scores)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / y.sum()
    fpr = fp / (1 - y).sum()
    precision = tp / (tp + fp)
    ap = np.sum(precision * y) / y.sum()  # average precision (step-wise)
    return fpr, tpr, tpr, precision, ap


def _sample(rng, n, prev, shift=1.4):
    y = (rng.random(n) < prev).astype(int)
    s = rng.normal(0, 1, n) + shift * y
    return s, y


def fig_roc_vs_pr():
    """Same score distributions at 20% and 2% prevalence: ROC unchanged, PR collapses."""
    rng = np.random.default_rng(42)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.2))
    results = {}
    for prev, color in [(0.20, BLUE), (0.02, ORANGE)]:
        s, y = _sample(rng, 300_000, prev)
        fpr, tpr, rec, prec, ap = _curves(s, y)
        auroc = np.trapezoid(tpr, fpr)
        results[prev] = (auroc, ap)
        a1.plot(fpr, tpr, color=color, lw=3 if prev == 0.20 else 2,
                ls="-" if prev == 0.20 else "--")
        a2.plot(rec, prec, color=color, lw=3)
        a2.axhline(prev, color=color, lw=1.5, ls=":")

    a1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    a1.text(0.36, 0.30, f"AUROC {results[0.20][0]:.2f}\nat both prevalences", fontsize=15,
            color=INK)
    a1.set_xlabel("False positive rate")
    a1.set_ylabel("Sensitivity (recall)")
    a1.set_title("ROC: identical", loc="left")

    a2.text(0.30, 0.90, f"20%: AUPRC {results[0.20][1]:.2f}", color=BLUE, fontsize=15,
            weight="bold")
    a2.text(0.33, 0.30, f"2%: AUPRC {results[0.02][1]:.2f}", color=ORANGE, fontsize=15,
            weight="bold")
    a2.text(0.55, 0.215, "baseline 0.20", color=BLUE, fontsize=12.5, ha="left", va="bottom")
    a2.text(0.55, 0.035, "baseline 0.02", color=ORANGE, fontsize=12.5, ha="left", va="bottom")
    a2.set_xlabel("Recall (sensitivity)")
    a2.set_ylabel("Precision (PPV)")
    a2.set_title("PR: collapses", loc="left")
    for a in (a1, a2):
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
        a.set_aspect("equal")
    return save(fig, "m02-04-roc-vs-pr")


def fig_decision_curve():
    """Net benefit of a calibrated simulated model vs treat-all and treat-none."""
    rng = np.random.default_rng(11)
    n = 200_000
    z = rng.normal(0, 1, n)
    risk = 1 / (1 + np.exp(-(-2.35 + 1.0 * z)))
    y = (rng.random(n) < risk).astype(int)
    prev = y.mean()

    pts = np.linspace(0.01, 0.45, 120)
    nb_model, nb_all = [], []
    for pt in pts:
        flag = risk >= pt
        tp = np.sum(flag & (y == 1)) / n
        fp = np.sum(flag & (y == 0)) / n
        w = pt / (1 - pt)
        nb_model.append(tp - fp * w)
        nb_all.append(prev - (1 - prev) * w)

    fig, ax = figure(10, 6.4)
    ax.plot(pts * 100, nb_all, color=ORANGE, lw=2.5)
    ax.axhline(0, color=MUTED, lw=2)
    ax.plot(pts * 100, nb_model, color=BLUE, lw=3.5)

    ax.text(5.8, 0.075, "treat all", color=ORANGE, fontsize=15, weight="bold")
    ax.text(30, -0.014, "treat none", color=MUTED, fontsize=15, weight="bold")
    ax.text(21, 0.03, "model", color=BLUE, fontsize=15, weight="bold")
    ax.axvline(prev * 100, color=MUTED, lw=1, ls=":")
    ax.text(prev * 100 + 0.5, 0.118, f"prevalence ≈ {prev * 100:.0f}%", fontsize=12.5,
            color=MUTED)

    ax.set_xlim(0, 45)
    ax.set_ylim(-0.03, 0.125)
    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve: is using the model worth it?", loc="left")
    return save(fig, "m02-04-decision-curve")
