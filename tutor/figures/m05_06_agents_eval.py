"""Figures for lesson m05-06-agents-eval."""

from __future__ import annotations

import numpy as np

from _style import BLUE, GRID, INK, MUTED, ORANGE, figure, save


def fig_cost_accuracy():
    """Task success vs cost per task for several system designs, with the Pareto frontier."""
    # Illustrative configurations: (label, cost per task in cents, success %, p95 seconds)
    configs = [
        ("small model,\nno retrieval", 0.2, 58, 2),
        ("small model\n+ RAG", 0.5, 76, 4),
        ("frontier model\n+ RAG", 2.5, 88, 7),
        ("agent,\nfrontier model", 9.0, 91, 28),
        ("agent + self-\ncritique loop", 22.0, 90, 65),
        ("frontier model,\nno retrieval", 1.8, 70, 5),
    ]
    frontier = {0, 1, 2, 3}
    LABEL_POS = {
        0: (0.26, 58, "left", "center"),
        1: (0.43, 77, "right", "center"),
        2: (2.2, 89.5, "right", "bottom"),
        3: (9.0, 88.3, "center", "top"),
        4: (22.0, 92.3, "center", "bottom"),
        5: (2.3, 70, "left", "center"),
    }

    fig, ax = figure(10.5, 7)
    for i, (lab, c, s, p95) in enumerate(configs):
        on = i in frontier
        ax.scatter(c, s, s=90 + p95 * 12, color=BLUE if on else ORANGE, alpha=0.85,
                   zorder=4, edgecolor="white", lw=1.5)
        tx, ty, ha, va = LABEL_POS[i]
        ax.text(tx, ty, f"{lab}\np95 {p95}s", fontsize=12.5, ha=ha, va=va, color=INK)
    fc = np.array([configs[i][1] for i in sorted(frontier)])
    fs = np.array([configs[i][2] for i in sorted(frontier)])
    ax.plot(fc, fs, color=BLUE, lw=2, ls="--", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 1, 5, 25], ["0.2¢", "1¢", "5¢", "25¢"])
    ax.set_xlim(0.12, 60)
    ax.set_ylim(50, 100)
    ax.set_xlabel("cost per task (log scale)")
    ax.set_ylabel("task success (%)")
    ax.grid(True, axis="y", color=GRID)
    ax.text(0.14, 97, "blue = Pareto frontier     bubble size = p95 latency", fontsize=13,
            color=MUTED)
    ax.set_title("More machinery is not always better", loc="left", pad=10)
    ax.text(58, 44, "illustrative numbers", fontsize=11, color=MUTED, style="italic",
            ha="right")
    return save(fig, "m05-06-cost-accuracy")
