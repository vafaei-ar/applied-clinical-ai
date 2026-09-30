"""Figures for lesson m08-03-sample-size-power (all data simulated with numpy)."""

from __future__ import annotations

import math
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, figure, save


def _auroc(y, s):
    ranks = np.empty(len(s))
    ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _calibration_slope(y, lp, iters=25):
    """Logistic regression of y on [1, lp] by Newton-Raphson; returns the slope on lp."""
    x = np.c_[np.ones(len(lp)), lp]
    b = np.array([0.0, 1.0])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(x @ b)))
        w = p * (1 - p)
        step = np.linalg.solve((x * w[:, None]).T @ x, x.T @ (y - p))
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b[1]


def fig_validation_precision():
    """How the 95% range of AUROC and calibration slope shrinks with validation events.

    The model is perfectly calibrated by construction (true slope 1), prevalence about 10%,
    true AUROC about 0.72, so every bit of spread is pure sampling error.
    """
    rng = np.random.default_rng(10)
    a, b = -2.45, 0.85
    sizes = np.array([250, 500, 1000, 2000, 3000, 5000, 10000, 20000])
    n_rep = 800
    lo_a, hi_a, lo_s, hi_s = [], [], [], []
    for n in sizes:
        auc, slope = np.empty(n_rep), np.empty(n_rep)
        for r in range(n_rep):
            lp = a + b * rng.normal(size=n)
            y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
            auc[r] = _auroc(y, lp)
            slope[r] = _calibration_slope(y, lp)
        qa = np.percentile(auc, [2.5, 97.5])
        qs = np.percentile(slope, [2.5, 97.5])
        lo_a.append(qa[0]); hi_a.append(qa[1]); lo_s.append(qs[0]); hi_s.append(qs[1])

    events = sizes * 0.10
    big = np.random.default_rng(3)
    lp = a + b * big.normal(size=1_000_000)
    y = big.binomial(1, 1 / (1 + np.exp(-lp)))
    true_auc = _auroc(y, lp)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 9.2), sharex=True)
    for axis, lo, hi, truth, color, label in [
        (ax, lo_a, hi_a, true_auc, BLUE, "AUROC"),
        (bx, lo_s, hi_s, 1.0, ORANGE, "Calibration slope"),
    ]:
        axis.fill_between(events, lo, hi, color=color, alpha=0.22, lw=0)
        axis.plot(events, lo, color=color, lw=2.5)
        axis.plot(events, hi, color=color, lw=2.5)
        axis.axhline(truth, color=INK, ls="--", lw=1.8)
        for ev in (100, 200):
            axis.axvline(ev, color=MUTED, lw=1.5, ls=":")
        axis.set_ylabel(label)
    ax.text(105, ax.get_ylim()[0] + 0.004, "100", color=MUTED, fontsize=14)
    ax.text(215, ax.get_ylim()[0] + 0.004, "200 events", color=MUTED, fontsize=14)
    ax.text(events[-1], true_auc + 0.004, f"true {true_auc:.2f}", ha="right", fontsize=14)
    bx.text(events[-1], 1.03, "true 1.0 (perfectly calibrated)", ha="right", fontsize=14)
    i100 = int(np.argmin(np.abs(events - 100)))
    ax.annotate(f"{lo_a[i100]:.2f} to {hi_a[i100]:.2f}\nat 100 events", (100, hi_a[i100]),
                xytext=(260, hi_a[i100] + 0.012), fontsize=14, color=BLUE, weight="bold",
                arrowprops={"arrowstyle": "-", "color": BLUE})
    bx.annotate(f"{lo_s[i100]:.2f} to {hi_s[i100]:.2f}\nat 100 events", (100, hi_s[i100]),
                xytext=(260, hi_s[i100] + 0.12), fontsize=14, color=ORANGE, weight="bold",
                arrowprops={"arrowstyle": "-", "color": ORANGE})
    bx.set_xscale("log")
    bx.set_xlim(25, 2200)
    bx.set_xticks([25, 50, 100, 200, 500, 1000, 2000])
    bx.set_xticklabels(["25", "50", "100", "200", "500", "1000", "2000"])
    bx.set_xlabel("Events in the validation set (prevalence 10%)")
    ax.set_title("What 95% of validation sets would show", loc="left")
    return save(fig, "m08-03-validation-precision")


def fig_power_misspecified():
    """Power of a trial sized for HR 0.75 (80% power) when the true HR is different."""
    nd = NormalDist()
    z_a = nd.inv_cdf(0.975)
    events = 4 * (z_a + nd.inv_cdf(0.80)) ** 2 / math.log(0.75) ** 2      # about 379
    hrs = np.linspace(0.97, 0.55, 200)
    power = np.array([nd.cdf(abs(math.log(h)) * math.sqrt(events / 4) - z_a) for h in hrs]) * 100

    fig, ax = figure(10, 6.4)
    ax.plot(hrs, power, color=BLUE, lw=3.5)
    ax.axhline(80, color=MUTED, ls=":", lw=1.5)
    marks = [(0.75, GREEN, "planned HR 0.75"), (0.85, ORANGE, "true HR 0.85"),
             (0.90, RED, "true HR 0.90")]
    offsets = {0.75: (0.74, 62, "left"), 0.85: (0.862, 52, "right"), 0.90: (0.893, 34, "right")}
    for hr, color, label in marks:
        pw = nd.cdf(abs(math.log(hr)) * math.sqrt(events / 4) - z_a) * 100
        ax.plot(hr, pw, "o", color=color, ms=13, zorder=5)
        tx, ty, ha = offsets[hr]
        ax.annotate(f"{label}\npower {pw:.0f}%", (hr, pw), xytext=(tx, ty), fontsize=14,
                    color=color, weight="bold", ha=ha,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 1.2})
    ax.set_xlim(0.98, 0.54)           # stronger effects to the right
    ax.set_ylim(0, 105)
    ax.set_xlabel("True hazard ratio (lower = bigger benefit)")
    ax.set_ylabel("Power (%)")
    ax.set_title(f"A trial with {events:.0f} events, sized for HR 0.75", loc="left")
    return save(fig, "m08-03-power-misspecified")
