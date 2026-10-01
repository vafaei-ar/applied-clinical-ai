"""Figures for lesson m05-01-llm-systems."""

from __future__ import annotations

import numpy as np

from _style import BLUE, INK, MUTED, ORANGE, figure, save


def fig_cost_per_task():
    """Cost per request vs cost per completed task for a cheap and a strong model (illustrative)."""
    # Illustrative numbers, not real prices.
    # Small model: $0.004/request, 3 requests per attempt (retries), 60% of tasks correct.
    # Strong model: $0.020/request, 1.1 requests per attempt, 95% of tasks correct.
    per_request = np.array([0.004, 0.020])
    per_task = np.array([0.004 * 3 / 0.60, 0.020 * 1.1 / 0.95])
    labels = ["Small model", "Strong model"]

    fig, ax = figure(10, 6.2)
    x = np.arange(2)
    w = 0.36
    b1 = ax.bar(x - w / 2, per_request * 100, w, color=MUTED, label="per request")
    b2 = ax.bar(x + w / 2, per_task * 100, w, color=[ORANGE, BLUE], label="per completed task")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.1f}¢", ha="center",
                    va="bottom", fontsize=16, weight="bold", color=INK)
    ax.set_xticks(x, labels, fontsize=17)
    ax.set_ylabel("cost (cents)")
    ax.set_ylim(0, 3.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(-0.45, 3.45, "grey = per request      colored = per completed task", fontsize=14,
            color=MUTED)
    ax.text(-0.45, 3.2, "small model: 3 calls per attempt, 60% correct\n"
            "strong model: 1.1 calls per attempt, 95% correct",
            fontsize=13, color=MUTED, va="top")
    ax.set_title("5x cheaper per call, about the same per task", loc="left", pad=10)
    ax.text(1.45, -0.42, "illustrative numbers", fontsize=11, color=MUTED, style="italic",
            ha="right", transform=ax.transData)
    return save(fig, "m05-01-cost-per-task")
