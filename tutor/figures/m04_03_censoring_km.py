"""Figures for lesson m04-03-censoring-km (numpy-only Kaplan-Meier, simulated synthetic trial)."""

from __future__ import annotations

import math

import numpy as np

from _style import BLUE, GRID, INK, MUTED, ORANGE, figure, save


def km(time: np.ndarray, event: np.ndarray):
    """Kaplan-Meier estimate with log(-log) Greenwood 95% CI. Returns step arrays."""
    ts = np.unique(time[event == 1])
    s, var_sum = 1.0, 0.0
    out_t, out_s, lo, hi = [0.0], [1.0], [1.0], [1.0]
    for t in ts:
        n = np.sum(time >= t)
        d = np.sum((time == t) & (event == 1))
        s *= 1 - d / n
        if n > d:
            var_sum += d / (n * (n - d))
        out_t.append(t)
        out_s.append(s)
        if 0 < s < 1:
            se = math.sqrt(var_sum) / abs(math.log(s))
            lo.append(s ** math.exp(1.96 * se))
            hi.append(s ** math.exp(-1.96 * se))
        else:
            lo.append(s)
            hi.append(s)
    return np.array(out_t), np.array(out_s), np.array(lo), np.array(hi)


def logrank_p(t1, e1, t2, e2) -> float:
    """Two-group log-rank test p-value (1 df chi-square)."""
    t = np.concatenate([t1, t2])
    e = np.concatenate([e1, e2])
    g = np.concatenate([np.zeros_like(t1), np.ones_like(t2)])
    o_minus_e, var = 0.0, 0.0
    for ti in np.unique(t[e == 1]):
        at_risk = t >= ti
        n = at_risk.sum()
        n1 = (at_risk & (g == 0)).sum()
        d = ((t == ti) & (e == 1)).sum()
        d1 = ((t == ti) & (e == 1) & (g == 0)).sum()
        o_minus_e += d1 - d * n1 / n
        if n > 1:
            var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    chi2 = o_minus_e**2 / var
    return math.erfc(math.sqrt(chi2 / 2))


def simulate_arm(rng, n, monthly_hazard, max_fu=30.0):
    """Exponential event times, staggered entry (admin censoring 12-30 mo), ~5%/yr dropout."""
    t_event = rng.exponential(1 / monthly_hazard, n)
    t_admin = rng.uniform(12, max_fu, n)
    t_drop = rng.exponential(12 / 0.05, n)
    time = np.minimum.reduce([t_event, t_admin, t_drop])
    event = (t_event <= np.minimum(t_admin, t_drop)).astype(int)
    return np.round(time, 2), event


def fig_km_two_arms():
    """KM curves for the two arms with 95% CIs and a numbers-at-risk table."""
    rng = np.random.default_rng(2026)
    base = -math.log(0.90) / 12  # ~10% 12-month recurrent stroke risk on aspirin alone
    arms = [("Aspirin alone", ORANGE, simulate_arm(rng, 1500, base)),
            ("Zetagrel + aspirin", BLUE, simulate_arm(rng, 1500, base * 0.75))]

    fig, first = figure(10, 8)
    first.remove()
    grid = fig.add_gridspec(2, 1, height_ratios=[5, 1.1])
    ax = fig.add_subplot(grid[0])
    table = fig.add_subplot(grid[1])
    ticks = [0, 6, 12, 18, 24, 30]
    for label, color, (t, e) in arms:
        xs, s, lo, hi = km(t, e)
        ax.step(xs, s, where="post", color=color, lw=3, label=label)
        ax.fill_between(xs, lo, hi, step="post", color=color, alpha=0.18, lw=0)
        ax.plot(t[e == 0][::25], np.interp(t[e == 0][::25], xs, s), "|", color=color,
                ms=9, alpha=0.6)  # thinned censoring ticks

    p = logrank_p(*arms[0][2], *arms[1][2])
    ax.text(1, 0.815, f"log-rank p = {p:.3f}", fontsize=15, color=INK)
    ax.set_xlim(0, 30)
    ax.set_ylim(0.8, 1.005)
    ax.set_xticks(ticks)
    ax.set_xlabel("Months since randomization", labelpad=6)
    ax.set_ylabel("Free of recurrent stroke")
    ax.yaxis.grid(True, color=GRID)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    table.set_xlim(0, 30)
    table.set_ylim(-0.4, 2.4)
    table.axis("off")
    table.text(-0.8, 2.1, "Number at risk", ha="left", fontsize=13, weight="bold", color=INK)
    for row, (label, color, (t, _)) in enumerate(arms):
        for tick in ticks:
            table.text(tick, 1.1 - row, f"{int(np.sum(t >= tick)):,}", ha="center", fontsize=13.5,
                       color=color)
    ax.text(30, 0.803, "y-axis starts at 0.80", fontsize=11.5, color=MUTED, ha="right",
            style="italic")
    ax.set_title("Synthetic trial: time to recurrent stroke", loc="left", pad=8)
    return save(fig, "m04-03-km-two-arms")


def fig_rmst_crossing():
    """Crossing survival curves: log-rank is weak, but the RMST difference is still defined."""
    tau = 24.0
    t = np.linspace(0, tau, 400)
    # Early harm, later benefit: treatment hazard falls over time; curves cross near month 8.
    s_ctrl = np.exp(-(t / 70.0))
    s_trt = np.exp(-((t / 297.0) ** 0.6))
    fig, ax = figure(10, 6.5)
    ax.fill_between(t, s_trt, s_ctrl, color=MUTED, alpha=0.15, lw=0)
    ax.plot(t, s_ctrl, color=ORANGE, lw=3.5, label="Control")
    ax.plot(t, s_trt, color=BLUE, lw=3.5, label="Treatment")
    ax.axvline(tau, color=INK, lw=1.5, ls="--")
    rmst_c = np.trapezoid(s_ctrl, t) if hasattr(np, "trapezoid") else np.trapz(s_ctrl, t)
    rmst_t = np.trapezoid(s_trt, t) if hasattr(np, "trapezoid") else np.trapz(s_trt, t)
    ax.text(0.5, 0.695, f"RMST (0-24 mo)\ncontrol {rmst_c:.1f} mo, treatment {rmst_t:.1f} mo",
            fontsize=14.5, color=INK, va="bottom")
    ax.set_xlim(0, 25)
    ax.set_ylim(0.68, 1.0)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["0", "6", "12", "18", "tau=24"])
    ax.set_xlabel("Months")
    ax.set_ylabel("Event-free probability")
    ax.legend(loc="upper right", bbox_to_anchor=(0.97, 1.0))
    ax.set_title("Curves cross: one HR can't summarize this", loc="left", pad=8)
    return save(fig, "m04-03-rmst-crossing")
