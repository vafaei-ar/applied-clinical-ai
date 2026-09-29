"""Figures for lesson m01-06-data-quality."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, RED, figure, save


def fig_count_regression():
    """Nightly cohort counts with a tolerance band; a silent upstream change trips the alarm."""
    rng = np.random.default_rng(6)
    days = np.arange(1, 41)
    base = 333 + np.cumsum(rng.normal(0.35, 0.6, size=days.size))
    counts = np.round(base + rng.normal(0, 2.0, size=days.size))
    drop_day = 31
    counts[drop_day - 1:] = np.round(counts[drop_day - 1:] * 0.71)

    # Expected = trailing 7-run median; band = +/- 5%
    expected = np.array([np.median(counts[max(0, i - 7):i]) if i else counts[0]
                         for i in range(days.size)])
    expected[drop_day - 1:] = expected[drop_day - 2]
    lo, hi = expected * 0.95, expected * 1.05

    fig, ax = figure(11, 6.2)
    ax.fill_between(days, lo, hi, color=GREEN, alpha=0.15, lw=0, label="expected ± 5%")
    ok = counts >= lo
    ax.plot(days, counts, color=INK, lw=2)
    ax.plot(days[ok], counts[ok], "o", color=BLUE, ms=7)
    ax.plot(days[~ok], counts[~ok], "o", color=RED, ms=10)
    ax.annotate("upstream relabels 'ED' as\n'Emergency': test fails",
                xy=(drop_day, counts[drop_day - 1]), xytext=(15.5, 262),
                fontsize=14, color=RED, weight="bold",
                arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.8})
    ax.set_xlabel("nightly pipeline run")
    ax.set_ylabel("cohort size")
    ax.set_ylim(220, 375)
    ax.set_xlim(0, 41)
    ax.legend(loc="upper left")
    ax.set_title("Expected-count regression test", loc="left")
    ax.text(40.5, 224, "simulated", ha="right", fontsize=11, color=MUTED, style="italic")
    return save(fig, "m01-06-count-regression")


def fig_checks_pipeline():
    """Where each kind of check lives along the pipeline, from source to features."""
    fig, ax = figure(11, 6.4)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    stages = [
        ("raw\nfeeds", BLUE, ["freshness", "schema /\ncontract"]),
        ("staged\ntables", PURPLE, ["unique", "not_null", "relationships", "accepted\nvalues"]),
        ("cohort", GREEN, ["1 row per\npatient", "age ≥ 18", "count in\nrange"]),
        ("features", ORANGE, ["no date ≥\nindex", "plausible\nranges"]),
    ]
    w, gap = 2.35, 0.37
    for i, (name, col, checks) in enumerate(stages):
        x = 0.15 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 4.6), w, 1.25, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=col, ec=col, alpha=0.9))
        ax.text(x + w / 2, 5.22, name, ha="center", va="center", fontsize=16, weight="bold",
                color="white")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, 5.22), xytext=(x + w + 0.03, 5.22),
                        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2})
        for j, chk in enumerate(checks):
            y = 3.75 - j * 1.0
            ax.add_patch(FancyBboxPatch((x + 0.12, y - 0.4), w - 0.24, 0.8,
                                        boxstyle="round,pad=0.02,rounding_size=0.1",
                                        fc="white", ec=col, lw=1.8))
            ax.text(x + w / 2, y, chk, ha="center", va="center", fontsize=13, color=INK)
    ax.set_title("Test at every layer, closest to the cause", loc="left", pad=4)
    return save(fig, "m01-06-checks-pipeline")
