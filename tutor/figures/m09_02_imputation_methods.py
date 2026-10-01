"""Figures for lesson m09-02-imputation-methods."""

from __future__ import annotations

import warnings
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, save

METHODS = [
    "Complete-case",
    "Mean",
    "Regression",
    "Regression + Y",
    "Stochastic x1",
    "MI, no Y",
    "MI + Y",
]


def _miss_prob(linear_part, target):
    lo, hi = -12.0, 12.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + linear_part)))).mean() > target:
            hi = mid
        else:
            lo = mid
    return 1 / (1 + np.exp(-(lo + linear_part)))


def _ols_slope(y, x, z):
    a = np.column_stack([np.ones(len(y)), x, z])
    beta = np.linalg.lstsq(a, y, rcond=None)[0]
    res = y - a @ beta
    s2 = res @ res / (len(y) - a.shape[1])
    cov = s2 * np.linalg.inv(a.T @ a)
    return beta[1], cov[1, 1]


def _impute(x, obs, preds, rng, proper, noise):
    """Normal linear imputation of x from `preds`; one draw.

    proper=True also draws the regression parameters (a Bayesian posterior draw),
    which is what makes the imputation "proper" for multiple imputation.
    """
    a = np.column_stack([np.ones(len(x))] + preds)
    ao, xo = a[obs], x[obs]
    b_hat = np.linalg.lstsq(ao, xo, rcond=None)[0]
    res = xo - ao @ b_hat
    dof = obs.sum() - a.shape[1]
    s2 = res @ res / dof
    if proper:
        sigma2 = s2 * dof / rng.chisquare(dof)
        b = rng.multivariate_normal(b_hat, sigma2 * np.linalg.inv(ao.T @ ao))
    else:
        sigma2, b = s2, b_hat
    out = x.copy()
    mu = a[~obs] @ b
    out[~obs] = mu + (rng.normal(size=(~obs).sum()) * np.sqrt(sigma2) if noise else 0.0)
    return out


def rubin(estimates, variances):
    """Rubin's rules for a scalar: pooled estimate and total variance."""
    m = len(estimates)
    q_bar = np.mean(estimates)
    u_bar = np.mean(variances)
    b = np.var(estimates, ddof=1)
    return q_bar, u_bar + (1 + 1 / m) * b


def _masked_stats(xc, x, miss):
    """RMSE on the masked entries, and SD of imputed over true values in the missing rows."""
    return np.sqrt(np.mean((xc[miss] - x[miss]) ** 2)), xc[miss].std() / x[miss].std()


def _row(y, z, xc, true_prev, rmse, sdr):
    """Result row for one completed dataset analysed as if it were real (single imputation)."""
    b, v = _ols_slope(y, xc, z)
    return [rmse, sdr, b - 1.0, float(abs(b - 1.0) <= 1.96 * np.sqrt(v)),
            100 * ((xc > 1).mean() - true_prev)]


@lru_cache(maxsize=1)
def simulate_methods(n=1500, reps=400, m=20, seed=20260930):
    """Compare handling methods when 40% of X is missing at random given (Z, W, Y).

    X depends on Z and W (both observed), Y = 1.0*X + 0.5*Z + noise, so the true
    slope of X is exactly 1.0. Missingness of X depends on the outcome Y and on Z
    (MAR). Returns per-method means over `reps` runs of: masked RMSE, SD ratio of
    imputed to true values in the missing rows, slope bias, 95% CI coverage of the
    true slope, and bias in the estimated share with X > 1 (percentage points).
    """
    rng = np.random.default_rng(seed)
    rec = {k: [] for k in METHODS}
    for _ in range(reps):
        z = rng.normal(size=n)
        w = rng.normal(size=n)
        x = 0.4 * z + 0.3 * w + np.sqrt(1 - 0.16 - 0.09) * rng.normal(size=n)
        y = 1.0 * x + 0.5 * z + rng.normal(size=n)
        p = _miss_prob(0.9 * (y - y.mean()) / y.std() + 0.5 * z, 0.40)
        miss = rng.random(n) < p
        obs = ~miss
        true_prev = (x > 1).mean()

        b, v = _ols_slope(y[obs], x[obs], z[obs])
        rec["Complete-case"].append(
            [np.nan, np.nan, b - 1.0, float(abs(b - 1.0) <= 1.96 * np.sqrt(v)),
             100 * ((x[obs] > 1).mean() - true_prev)]
        )
        xm = x.copy()
        xm[miss] = x[obs].mean()
        xr = _impute(x, obs, [z, w], rng, proper=False, noise=False)
        xry = _impute(x, obs, [z, w, y], rng, proper=False, noise=False)
        xs = _impute(x, obs, [z, w, y], rng, proper=True, noise=True)
        for name, xc in (("Mean", xm), ("Regression", xr), ("Regression + Y", xry),
                         ("Stochastic x1", xs)):
            rec[name].append(_row(y, z, xc, true_prev, *_masked_stats(xc, x, miss)))
        for name, preds in (("MI, no Y", [z, w]), ("MI + Y", [z, w, y])):
            qs, us, rm, sd, pv = [], [], [], [], []
            for _ in range(m):
                xc = _impute(x, obs, preds, rng, proper=True, noise=True)
                b, v = _ols_slope(y, xc, z)
                qs.append(b)
                us.append(v)
                r, s = _masked_stats(xc, x, miss)
                rm.append(r)
                sd.append(s)
                pv.append((xc > 1).mean())
            q_bar, t = rubin(qs, us)
            rec[name].append(
                [np.mean(rm), np.mean(sd), q_bar - 1.0, float(abs(q_bar - 1.0) <= 1.96 * np.sqrt(t)),
                 100 * (np.mean(pv) - true_prev)]
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns for complete-case
        return {k: np.nanmean(np.array(v), axis=0) for k, v in rec.items()}


def fig_rmse_vs_inference():
    """Masked RMSE, distribution fidelity, and slope bias for seven handling methods."""
    res = simulate_methods()
    rmse = np.array([res[k][0] for k in METHODS])
    sdr = np.array([res[k][1] for k in METHODS])
    bias = np.array([res[k][2] for k in METHODS])

    fig, axes = plt.subplots(1, 3, figsize=(12, 6.4), sharey=True)
    ypos = np.arange(len(METHODS))[::-1]
    colors = []
    for k in METHODS:
        colors.append(ORANGE if k == "Regression + Y" else BLUE if k == "MI + Y" else MUTED)

    panels = [
        (axes[0], rmse, "Error on masked\nvalues (RMSE)", (0, 1.45), None, "{:.2f}"),
        (axes[1], sdr, "Spread of imputed\nvalues (true = 1)", (0, 1.45), 1.0, "{:.2f}"),
        (axes[2], bias, "Slope bias\n(true slope = 1)", (-0.42, 0.42), 0.0, "{:+.2f}"),
    ]
    for ax, vals, title, xlim, ref, fmt in panels:
        for y_i, v, c in zip(ypos, vals, colors, strict=False):
            if np.isnan(v):
                ax.text(xlim[0] + 0.03 * (xlim[1] - xlim[0]), y_i, "n/a", va="center",
                        ha="left", fontsize=13, color=MUTED)
                continue
            ax.barh(y_i, v, color=c, height=0.62)
            span = xlim[1] - xlim[0]
            label = "0.00" if abs(v) < 0.005 else fmt.format(v)
            if v >= -0.005:
                ax.text(max(v, 0) + 0.02 * span, y_i, label, va="center", ha="left",
                        fontsize=13, color=INK, weight="bold")
            else:
                ax.text(v - 0.02 * span, y_i, label, va="center", ha="right",
                        fontsize=13, color=INK, weight="bold")
        if ref is not None:
            ax.axvline(ref, color=INK, lw=1.5, ls="--" if ref == 1.0 else "-")
        ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=15, loc="left")
        ax.tick_params(axis="x", labelsize=12)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(METHODS, fontsize=14)
    fig.text(
        0.01, 0.005,
        "Simulation: 1,500 patients x 400 runs, 40% of X missing at random given Z, W and the outcome Y. "
        "Orange = best RMSE, blue = multiple imputation with Y.",
        fontsize=10, color=MUTED, style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "m09-02-rmse-vs-inference")


def fig_coverage():
    """95% confidence-interval coverage of the true slope."""
    res = simulate_methods()
    names = ["Complete-case", "Mean", "Stochastic x1", "MI, no Y", "MI + Y"]
    vals = np.array([100 * res[k][3] for k in names])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ypos = np.arange(len(names))[::-1]
    colors = [GREEN if 92 <= v <= 98 else RED for v in vals]
    ax.barh(ypos, vals, color=colors, height=0.62)
    top = len(names) - 0.55
    ax.vlines(95, -0.6, top, color=INK, lw=2, ls="--")
    ax.text(95, top + 0.08, "nominal 95%", ha="center", va="bottom", fontsize=13, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=14)
    ax.set_xlim(0, 112)
    ax.set_xlabel("share of runs whose 95% CI contains the true slope (%)", fontsize=14)
    for y_i, v in zip(ypos, vals, strict=False):
        if v > 30:
            ax.text(v - 1.5, y_i, f"{v:.0f}%", va="center", ha="right", fontsize=15,
                    weight="bold", color="white")
        else:
            ax.text(v + 1.5, y_i, f"{v:.0f}%", va="center", ha="left", fontsize=15, weight="bold")
    ax.set_title("Only proper multiple imputation gets the interval right", loc="left", fontsize=17)
    ax.set_ylim(-0.6, len(names) + 0.25)
    return save(fig, "m09-02-coverage")
