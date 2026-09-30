"""Figures for lesson m09-03-deployment-sensitivity."""

from __future__ import annotations

from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from _style import BLUE, GREEN, INK, MUTED, ORANGE, save


def _sigmoid(t):
    return 1 / (1 + np.exp(-t))


# ---------------------------------------------------------------------------
# Tipping-point (delta-adjustment) sensitivity analysis on an invented trial
# ---------------------------------------------------------------------------

def _rubin(estimates, variances):
    m = len(estimates)
    b = np.var(estimates, ddof=1)
    return np.mean(estimates), np.sqrt(np.mean(variances) + (1 + 1 / m) * b)


def _ols(y, design):
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    cov = resid @ resid / (len(y) - design.shape[1]) * np.linalg.inv(design.T @ design)
    return beta, cov


@lru_cache(maxsize=1)
def simulate_tipping(n_per_arm=300, m=100, seed=20260930):
    """Invented trial: intensive vs usual lipid lowering, outcome LDL at month 12 (mg/dL).

    About 30% of patients have no month-12 LDL (dropout at random given baseline LDL).
    The main analysis imputes the missing outcomes under MAR (within arm, given baseline),
    m = 100 proper draws. The sensitivity analysis then adds delta mg/dL to the imputed
    outcomes of the intensive arm only (dropouts there did worse than MAR predicts) and
    pools with Rubin's rules. True effect in the simulation: -25 mg/dL.
    Returns (deltas, effect, lower, upper, tipping_delta, missing_share).
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_arm
    arm = np.repeat([0, 1], n_per_arm)
    base = rng.normal(130, 30, n)
    y = 15 + 0.75 * base - 25 * arm + rng.normal(0, 28, n)
    lin = 0.03 * (base - 130)
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(mid + lin).mean() > 0.30:
            hi = mid
        else:
            lo = mid
    miss = rng.random(n) < _sigmoid(lo + lin)

    imputations = []
    for _ in range(m):
        out = y.copy()
        for g in (0, 1):
            obs = (arm == g) & ~miss
            gap = (arm == g) & miss
            a_obs = np.column_stack([np.ones(obs.sum()), base[obs]])
            b_hat = np.linalg.lstsq(a_obs, y[obs], rcond=None)[0]
            resid = y[obs] - a_obs @ b_hat
            dof = obs.sum() - 2
            s2 = resid @ resid / dof
            sigma2 = s2 * dof / rng.chisquare(dof)
            b_draw = rng.multivariate_normal(b_hat, sigma2 * np.linalg.inv(a_obs.T @ a_obs))
            a_gap = np.column_stack([np.ones(gap.sum()), base[gap]])
            out[gap] = a_gap @ b_draw + rng.normal(size=gap.sum()) * np.sqrt(sigma2)
        imputations.append(out)

    design = np.column_stack([np.ones(n), arm, base])
    deltas = np.arange(0, 81, 2.0)
    effect, lower, upper = [], [], []
    for delta in deltas:
        est, var = [], []
        for out in imputations:
            shifted = out.copy()
            shifted[(arm == 1) & miss] += delta
            beta, cov = _ols(shifted, design)
            est.append(beta[1])
            var.append(cov[1, 1])
        q, se = _rubin(est, var)
        effect.append(q)
        lower.append(q - 1.96 * se)
        upper.append(q + 1.96 * se)
    effect, lower, upper = map(np.array, (effect, lower, upper))
    crossing = np.where(upper >= 0)[0]
    if len(crossing):
        i = crossing[0]
        tip = deltas[i - 1] + (0 - upper[i - 1]) * (deltas[i] - deltas[i - 1]) / (upper[i] - upper[i - 1])
    else:
        tip = np.nan
    return deltas, effect, lower, upper, tip, miss.mean()


def fig_tipping_point():
    """Treatment effect as dropouts in the active arm drift away from the MAR assumption."""
    deltas, effect, lower, upper, tip, _ = simulate_tipping()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.fill_between(deltas, lower, upper, color=BLUE, alpha=0.18, lw=0)
    ax.plot(deltas, effect, color=BLUE, lw=3.5)
    ax.axhline(0, color=INK, lw=2)
    ax.text(1, 1.0, "no effect", fontsize=13, color=INK, va="bottom")

    # reference: dropouts lose the entire benefit
    ax.axvline(25, color=MUTED, lw=2, ls="--")
    ax.text(25.8, -31.5, "dropouts lose the\nwhole benefit", fontsize=13, color=MUTED, va="bottom")
    ax.plot([25], [np.interp(25, deltas, effect)], "o", color=BLUE, ms=11)

    ax.plot([tip], [0], "o", color=ORANGE, ms=15, zorder=5)
    ax.annotate(f"tipping point\ndelta = {tip:.0f} mg/dL", xy=(tip, 0), xytext=(tip - 4, -13.5),
                fontsize=15, color=ORANGE, weight="bold", ha="right",
                arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 2})
    ax.plot([0], [effect[0]], "o", color=GREEN, ms=11)
    ax.text(1.5, effect[0] - 3.4, f"MAR analysis\n{effect[0]:.0f} mg/dL", fontsize=13, color=GREEN,
            weight="bold", va="top")
    ax.set_xlim(0, 82)
    ax.set_ylim(-33, 6)
    ax.set_xlabel("delta: extra LDL (mg/dL) in intensive-arm dropouts, beyond the MAR prediction",
                  fontsize=14)
    ax.set_ylabel("effect on month-12 LDL (mg/dL)", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.set_title("How wrong must the dropouts be to change the conclusion?", loc="left", fontsize=17)
    fig.text(0.01, 0.005,
             "Invented trial, 300 per arm, about 30% missing month-12 LDL, 100 imputations, band = 95% CI.",
             fontsize=10.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return save(fig, "m09-03-tipping-point")


# ---------------------------------------------------------------------------
# Deployment shift: missing-indicator model when the ordering process changes
# ---------------------------------------------------------------------------

def _irls(y, a, iters=40, ridge=1e-6):
    b = np.zeros(a.shape[1])
    for _ in range(iters):
        p = _sigmoid(a @ b)
        w = p * (1 - p)
        b = b + np.linalg.solve(a.T @ (a * w[:, None]) + ridge * np.eye(a.shape[1]), a.T @ (y - p))
    return b


def _auroc(y, score):
    ranks = score.argsort().argsort() + 1.0
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


SCENARIOS = ["Same process", "Routine ordering", "Feed outage"]


@lru_cache(maxsize=1)
def simulate_shift(n=20000, reps=40, seed=20260930):
    """Logistic risk model with and without a missing indicator, under process shift.

    Z is an observed covariate, U an unmeasured severity, X an LDL-like lab. Y (binary,
    about 25% prevalence) depends on X, Z and U. In training the lab is recorded when
    the clinician is worried (probability rises with U), so a missing X means a
    healthier patient. At deployment the process is either unchanged, becomes routine
    (lab recorded for 90% of patients regardless of severity), or breaks (only 20% arrive,
    unrelated to severity). Returns {(scenario, model): (AUROC, calibration-in-the-large)}.
    """
    rng = np.random.default_rng(seed)

    def draw(size, rule):
        z = rng.normal(size=size)
        u = rng.normal(size=size)
        x = 0.5 * z + 0.5 * u + np.sqrt(0.5) * rng.normal(size=size)
        y = (rng.random(size) < _sigmoid(-1.5 + 0.6 * x + 0.4 * z + 0.8 * u)).astype(float)
        return z, x, y, rule(u, size)

    rules = {
        "Same process": lambda u, s: rng.random(s) < _sigmoid(-0.4 + 1.5 * u),
        "Routine ordering": lambda u, s: rng.random(s) < 0.90,
        "Feed outage": lambda u, s: rng.random(s) < 0.20,
    }

    def design(z, x, seen, mean_x, indicator):
        cols = [np.ones(len(z)), z, np.where(seen, x, mean_x)]
        if indicator:
            cols.append((~seen).astype(float))
        return np.column_stack(cols)

    acc = {}
    for _ in range(reps):
        z, x, y, seen = draw(n, rules["Same process"])
        mean_x = x[seen].mean()
        fits = {ind: _irls(y, design(z, x, seen, mean_x, ind)) for ind in (True, False)}
        for name in SCENARIOS:
            zt, xt, yt, st = draw(n, rules[name])
            for ind in (True, False):
                p = _sigmoid(design(zt, xt, st, mean_x, ind) @ fits[ind])
                acc.setdefault((name, ind), []).append((_auroc(yt, p), p.mean() - yt.mean()))
    return {k: tuple(np.mean(v, axis=0)) for k, v in acc.items()}


def fig_process_shift():
    """AUROC and calibration-in-the-large for indicator vs no-indicator models under shift."""
    res = simulate_shift()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 6))
    xs = np.arange(len(SCENARIOS))
    styles = ((True, ORANGE, "with indicator", -0.17), (False, BLUE, "without indicator", 0.17))

    # Panel 1: AUROC as dots (a dot plot needs no zero baseline)
    for x_i, name in zip(xs, SCENARIOS, strict=False):
        a, b = res[(name, True)][0], res[(name, False)][0]
        ax1.plot([x_i - 0.17, x_i + 0.17], [a, b], color=MUTED, lw=2, zorder=1)
    for ind, color, label, off in styles:
        vals = np.array([res[(s_, ind)][0] for s_ in SCENARIOS])
        ax1.scatter(xs + off, vals, s=230, color=color, zorder=3, label=label)
        for x_i, v, name in zip(xs + off, vals, SCENARIOS, strict=False):
            other = res[(name, not ind)][0]
            higher = v >= other
            ax1.text(x_i, v + (0.0065 if higher else -0.0065), f"{v:.3f}", ha="center",
                     va="bottom" if higher else "top", fontsize=12.5, weight="bold", color=INK)
    ax1.set_ylim(0.64, 0.78)
    ax1.set_xlim(-0.6, 2.6)
    ax1.set_title("Discrimination\n(AUROC, higher is better)", fontsize=15, loc="left")
    ax1.legend(loc="lower left", fontsize=12.5)

    # Panel 2: calibration-in-the-large as bars from zero
    w = 0.36
    for ind, color, _, _ in styles:
        off = -w / 2 if ind else w / 2
        vals = np.array([res[(s_, ind)][1] for s_ in SCENARIOS]) * 100
        ax2.bar(xs + off, vals, width=w, color=color)
        for x_i, v in zip(xs + off, vals, strict=False):
            ax2.text(x_i, v + (0.25 if v >= 0 else -0.25), f"{v:+.1f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=12, weight="bold", color=INK)
    ax2.axhline(0, color=INK, lw=1.5)
    ax2.set_ylim(-7, 7)
    ax2.set_title("Calibration error\n(points of predicted risk)", fontsize=15, loc="left")

    for ax in (ax1, ax2):
        ax.set_xticks(xs)
        ax.set_xticklabels(["Same\nprocess", "Routine\nordering", "Feed\noutage"], fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
    fig.text(0.01, 0.005,
             "Simulation: lab recorded when clinicians are worried (training). 40 runs, 20,000 patients, prevalence about 25%.",
             fontsize=10.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return save(fig, "m09-03-process-shift")
