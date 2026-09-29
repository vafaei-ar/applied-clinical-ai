"""Figures for lesson m07-03-telling-your-story."""

from __future__ import annotations

from _style import BLUE, GREEN, MUTED, ORANGE, PURPLE, RED, figure, save


def fig_walkthrough():
    """A 2-minute project walkthrough as a time budget (Gantt style)."""
    parts = [
        ("Problem + why it matters", 15, PURPLE),
        ("Data + cohort", 20, BLUE),
        ("The hardest decision", 35, ORANGE),
        ("Result, with numbers", 20, GREEN),
        ("Limitation + next step", 20, RED),
    ]
    fig, ax = figure(10.5, 6.0)
    start = 0
    for i, (label, secs, color) in enumerate(parts):
        y = len(parts) - 1 - i
        ax.barh(y, secs, left=start, height=0.62, color=color, alpha=0.85)
        ax.text(start + secs / 2, y, f"{secs}s", ha="center", va="center", color="white",
                fontsize=15, weight="bold")
        start += secs
    ax.set_yticks(range(len(parts)), [p[0] for p in parts][::-1], fontsize=15)
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 30, 60, 90, 110])
    ax.axvline(110, color=MUTED, ls="--", lw=1.8)
    ax.text(111, 4.45, "stop,\ninvite\nquestions", fontsize=12, color=MUTED, va="top")
    ax.set_xlabel("seconds")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("A 2-minute walkthrough: spend time on decisions", loc="left", pad=8,
                 fontsize=19)
    return save(fig, "m07-03-walkthrough")
