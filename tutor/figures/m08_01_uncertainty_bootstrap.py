"""Figures for lesson m08-01-uncertainty-bootstrap (all data simulated with numpy)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import BLUE, INK, MUTED, ORANGE, PURPLE, save


def _auroc(y, s):
    """Mann-Whitney AUROC (scores are continuous, so ties are ignored)."""
    ranks = np.empty(len(s))
    ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _encounters(rng, n_pat):
    """Repeated encounters per patient. Sicker patients have more encounters, and the
    model's error is partly persistent within a patient (both make rows non-independent)."""
    u = rng.normal(0, 1.3, n_pat)                       # patient-level frailty
    m = 1 + rng.poisson(np.exp(0.7 + 0.6 * u))          # encounters per patient
    pid = np.repeat(np.arange(n_pat), m)
    x = rng.normal(0, 1, len(pid))
    lp = -3.3 + u[pid] + 0.5 * x
    y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    score = lp + rng.normal(0, 1.0, n_pat)[pid] + rng.normal(0, 0.8, len(pid))
    return pid, y, score


def _row_boot(y, s, n_boot, rng):
    n = len(y)
    out = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        out[b] = _auroc(y[i], s[i])
    return out


def _cluster_boot(pid, y, s, n_boot, rng):
    n_pat = pid.max() + 1
    ids = np.arange(n_pat)
    starts = np.searchsorted(pid, ids)
    ends = np.searchsorted(pid, ids, side="right")
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_pat, n_pat)
        idx = np.concatenate([np.arange(starts[p], ends[p]) for p in pick])
        out[b] = _auroc(y[idx], s[idx])
    return out


def _smooth_density(vals, grid):
    """Histogram on a fine grid, smoothed with a Gaussian kernel (no scipy needed)."""
    h, _ = np.histogram(vals, bins=np.append(grid, grid[-1] + (grid[1] - grid[0])), density=True)
    k = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    return np.convolve(h, k / k.sum(), mode="same")


def fig_cluster_bootstrap():
    """Row-level vs patient-level bootstrap: interval width, and coverage over many studies."""
    n_pat, n_studies, n_boot_cov = 800, 250, 300
    big_pid, big_y, big_s = _encounters(np.random.default_rng(99), 300_000)
    truth = _auroc(big_y, big_s)

    rng = np.random.default_rng(7)
    hit_row = hit_cl = 0
    widths, studies = [], []
    for _ in range(n_studies):
        pid, y, s = _encounters(rng, n_pat)
        lo, hi = np.percentile(_row_boot(y, s, n_boot_cov, rng), [2.5, 97.5])
        hit_row += lo <= truth <= hi
        w_r = hi - lo
        lo, hi = np.percentile(_cluster_boot(pid, y, s, n_boot_cov, rng), [2.5, 97.5])
        hit_cl += lo <= truth <= hi
        widths.append((w_r, hi - lo))
        studies.append((pid, y, s))
    cov_row, cov_cl = 100 * hit_row / n_studies, 100 * hit_cl / n_studies

    # Show the study whose width ratio is the median, not a hand-picked one.
    ratio = np.array([c / r for r, c in widths])
    pid, y, s = studies[int(np.argsort(ratio)[n_studies // 2])]
    rng = np.random.default_rng(11)
    rows = _row_boot(y, s, 3000, rng)
    clus = _cluster_boot(pid, y, s, 3000, rng)
    w_row = np.subtract(*np.percentile(rows, [97.5, 2.5]))
    w_cl = np.subtract(*np.percentile(clus, [97.5, 2.5]))

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1.5, 1]})
    grid = np.linspace(min(rows.min(), clus.min()) - 0.005, max(rows.max(), clus.max()) + 0.005, 160)
    for vals, color in [(clus, BLUE), (rows, ORANGE)]:
        d = _smooth_density(vals, grid)
        ax.fill_between(grid, d, color=color, alpha=0.25, lw=0)
        ax.plot(grid, d, color=color, lw=3)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ax.plot([lo, hi], [-0.03 * d.max() * (1 if color == BLUE else 2.4)] * 2, color=color,
                lw=6, solid_capstyle="butt", clip_on=False)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("AUROC in 3,000 bootstrap resamples (bars: 95% CI)")
    ax.text(0.03, 0.95, f"row-level\nCI width {w_row:.3f}", transform=ax.transAxes,
            color=ORANGE, fontsize=15, weight="bold", va="top")
    ax.text(0.97, 0.95, f"patient-level\nCI width {w_cl:.3f}", transform=ax.transAxes,
            color=BLUE, fontsize=15, weight="bold", va="top", ha="right")
    ax.set_title(f"A typical dataset: {len(y):,} encounters, {pid.max() + 1} patients", loc="left",
                 fontsize=17)

    labels = ["Row-level", "Patient-level"]
    bx.barh(labels, [cov_row, cov_cl], color=[ORANGE, BLUE], height=0.55)
    bx.axvline(95, color=INK, ls="--", lw=2)
    bx.text(94, -0.5, "nominal 95%", ha="right", va="center", fontsize=13, color=INK)
    for i, v in enumerate([cov_row, cov_cl]):
        bx.text(v - 2, i, f"{v:.0f}%", ha="right", va="center", color="white", fontsize=17,
                weight="bold")
    bx.set_xlim(0, 100)
    bx.set_ylim(1.45, -0.75)
    bx.tick_params(axis="y", labelsize=16)
    bx.set_xlabel(f"How often the CI contained the true AUROC ({n_studies} studies)")
    return save(fig, "m08-01-cluster-bootstrap")


def fig_paired_difference():
    """Two models on the same patients: overlapping CIs, yet the paired difference excludes 0."""
    rng = np.random.default_rng(0)
    n = 4000
    lp = -2.2 + 1.1 * rng.normal(0, 1, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    common = rng.normal(0, 1.0, n)                      # error shared by both models
    s_a = lp + common + rng.normal(0, 0.5, n)
    s_b = lp + 0.75 * common + rng.normal(0, 0.5, n)

    a, b = _auroc(y, s_a), _auroc(y, s_b)
    boot = np.random.default_rng(100)
    ba, bb = np.empty(1000), np.empty(1000)
    for k in range(1000):
        i = boot.integers(0, n, n)
        ba[k], bb[k] = _auroc(y[i], s_a[i]), _auroc(y[i], s_b[i])
    ci_a, ci_b = np.percentile(ba, [2.5, 97.5]), np.percentile(bb, [2.5, 97.5])
    ci_d = np.percentile(bb - ba, [2.5, 97.5])

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 7.4), gridspec_kw={"height_ratios": [2, 1]})
    for yy, est, ci, color, name in [(1, a, ci_a, BLUE, "Model A"), (0, b, ci_b, ORANGE, "Model B")]:
        ax.plot(ci, [yy, yy], color=color, lw=5, solid_capstyle="round")
        ax.plot(est, yy, "o", color=color, ms=13)
        ax.text(ci[1] + 0.004, yy, f"{est:.3f}\n({ci[0]:.3f}-{ci[1]:.3f})", va="center",
                fontsize=14, color=color)
    ax.axvspan(max(ci_a[0], ci_b[0]), min(ci_a[1], ci_b[1]), color=MUTED, alpha=0.18, lw=0)
    ax.text((max(ci_a[0], ci_b[0]) + min(ci_a[1], ci_b[1])) / 2, 0.5, "overlap", ha="center",
            va="center", fontsize=14, color=MUTED)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Model A", "Model B"], fontsize=16)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.665, 0.79)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("AUROC with 95% bootstrap CI")
    ax.set_title("Overlapping CIs do not mean 'no difference'", loc="left")

    bx.axvline(0, color=INK, lw=2, ls="--")
    bx.plot(ci_d, [0, 0], color=PURPLE, lw=5, solid_capstyle="round")
    bx.plot(b - a, 0, "o", color=PURPLE, ms=13)
    bx.text(ci_d[1] + 0.003, 0, f"{b - a:+.3f}\n({ci_d[0]:+.3f} to {ci_d[1]:+.3f})", va="center",
            fontsize=14, color=PURPLE)
    bx.set_yticks([0])
    bx.set_yticklabels(["B minus A\n(paired)"], fontsize=16)
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xlim(-0.012, 0.062)
    bx.set_ylim(-0.7, 0.7)
    bx.set_xlabel("Difference in AUROC, resampling the same patients")
    return save(fig, "m08-01-paired-difference")
