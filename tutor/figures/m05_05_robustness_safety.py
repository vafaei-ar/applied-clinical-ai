"""Figures for lesson m05-05-robustness-safety."""

from __future__ import annotations

import numpy as np

from _style import BLUE, INK, MUTED, ORANGE, RED, figure, save


def fig_perturbation_flips():
    """Recommendation flip rate by perturbation type, against the resampling noise floor."""
    # Illustrative numbers for a hypothetical evaluation run.
    rows = [
        ("Same prompt, resampled", 3, MUTED),
        ("Typos and abbreviations", 5, BLUE),
        ("Paraphrased question", 8, BLUE),
        ("Chunks in a different order", 11, BLUE),
        ("Irrelevant history added", 14, ORANGE),
        ("Sex swapped", 4, BLUE),
        ("Race swapped", 9, RED),
    ]
    labels = [r[0] for r in rows][::-1]
    vals = np.array([r[1] for r in rows][::-1])
    colors = [r[2] for r in rows][::-1]

    fig, ax = figure(10.5, 6.8)
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.3, yi, f"{v}%", va="center", fontsize=16, weight="bold", color=INK)
    noise = rows[0][1]
    ax.axvline(noise, color=MUTED, ls="--", lw=2)
    ax.text(noise + 0.2, len(rows) - 0.35, "noise floor", color=MUTED, fontsize=13,
            va="bottom")
    ax.set_yticks(y, labels, fontsize=15)
    ax.set_xlim(0, 17)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("% of questions where the recommendation changed")
    ax.set_title("Same question, different answer", loc="left", pad=10)
    ax.text(17, -1.5, "illustrative numbers", fontsize=11, color=MUTED, style="italic",
            ha="right")
    return save(fig, "m05-05-perturbation-flips")
