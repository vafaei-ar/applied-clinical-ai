"""Figures for lesson m08-02-testing-multiplicity (all data simulated with numpy)."""

from __future__ import annotations

import math

import numpy as np

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, figure, save

_erfc = np.frompyfunc(math.erfc, 1, 1)


def _two_sided_p(z):
    return np.asarray(_erfc(np.abs(z) / math.sqrt(2)), dtype=float)


def _rejections(p, is_real, method, alpha=0.05):
    """Number of (real, null) hypotheses rejected per simulated study, for one method."""
    n_sim, m = p.shape
    order = np.argsort(p, axis=1)
    ps = np.take_along_axis(p, order, 1)
    if method == "none":
        k = (ps <= alpha).sum(1)
    elif method == "bonferroni":
        k = (ps <= alpha / m).sum(1)
    elif method == "holm":  # step-down: stop at the first failure
        k = np.cumprod(ps <= alpha / (m - np.arange(m)), axis=1).sum(1)
    elif method == "bh":  # step-up: largest k with p_(k) <= k/m * q
        ok = ps <= alpha * np.arange(1, m + 1) / m
        k = np.where(ok.any(1), m - np.argmax(ok[:, ::-1], axis=1), 0)
    else:
        raise ValueError(method)
    real_sorted = np.take_along_axis(np.broadcast_to(is_real, (n_sim, m)), order, 1)
    rejected = np.arange(m)[None, :] < k[:, None]
    return (real_sorted & rejected).sum(1), (~real_sorted & rejected).sum(1)


def fig_subgroup_false_positives():
    """Chance of at least one false-positive subgroup, with and without correction."""
    rng = np.random.default_rng(2)
    grid = [1, 2, 3, 5, 8, 12, 16, 20, 30, 40]
    n_sim = 10_000
    unc, cor = [], []
    for k in grid:
        p = _two_sided_p(rng.normal(size=(n_sim, k)))
        none_real, none_false = _rejections(p, np.zeros(k, bool), "none")
        holm_real, holm_false = _rejections(p, np.zeros(k, bool), "holm")
        unc.append((none_false > 0).mean() * 100)
        cor.append((holm_false > 0).mean() * 100)

    ks = np.arange(1, 41)
    theory = 100 * (1 - 0.95 ** ks)

    fig, ax = figure(10, 6.6)
    ax.plot(ks, theory, color=RED, lw=2.5, alpha=0.6)
    ax.plot(grid, unc, "o", color=RED, ms=10, label="No correction")
    ax.plot(grid, cor, "o", color=BLUE, ms=10, label="Holm (Bonferroni, BH look the same here)")
    ax.axhline(5, color=BLUE, lw=2.5, alpha=0.6)
    ax.text(40, 9.5, "5%", color=BLUE, ha="right", fontsize=15, weight="bold")
    for k, label in [(12, "12 subgroups: 46%"), (20, "20: 64%"), (40, "40: 87%")]:
        y0 = 100 * (1 - 0.95 ** k)
        ax.annotate(label, (k, y0), xytext=(k - 1.2, y0 + 7), ha="right", fontsize=14,
                    color=RED, weight="bold")
    ax.set_xlim(0, 41)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Number of subgroups tested (all true effects are zero)")
    ax.set_ylabel("Chance of at least one\nfalse-positive finding (%)")
    ax.set_title("Every extra subgroup is another lottery ticket", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.36))
    return save(fig, "m08-02-subgroup-false-positives")


def fig_correction_tradeoff():
    """50 tests, 10 real effects: real vs false discoveries for each correction rule."""
    rng = np.random.default_rng(2)
    m, n_real, n_sim = 50, 10, 10_000
    is_real = np.r_[np.ones(n_real, bool), np.zeros(m - n_real, bool)]
    z = rng.normal(size=(n_sim, m))
    z[:, :n_real] += 3.0
    p = _two_sided_p(z)

    methods = [("none", "No correction"), ("bonferroni", "Bonferroni"), ("holm", "Holm"),
               ("bh", "Benjamini-Hochberg")]
    real_found, false_found = [], []
    for key, _ in methods:
        r, f = _rejections(p, is_real, key)
        real_found.append(r.mean())
        false_found.append(f.mean())

    fig, ax = figure(10, 6.8)
    y = np.arange(len(methods))[::-1] * 1.0
    h = 0.36
    ax.barh(y + h / 2, real_found, height=h, color=GREEN, label="Real effects found (of 10)")
    ax.barh(y - h / 2, false_found, height=h, color=RED, label="False findings (among 40 nulls)")
    for yy, r, f in zip(y, real_found, false_found):
        ax.text(r + 0.12, yy + h / 2, f"{r:.1f}", va="center", fontsize=15, color=GREEN,
                weight="bold")
        ax.text(f + 0.12, yy - h / 2, f"{f:.2f}" if f < 1 else f"{f:.1f}", va="center",
                fontsize=15, color=RED, weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in methods], fontsize=16)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("Average number of findings per study (10,000 simulated studies)")
    ax.set_title("50 tests, 10 real effects", loc="left")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.68), fontsize=13)
    return save(fig, "m08-02-correction-tradeoff")
