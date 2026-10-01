"""Figures for lesson m03-03-docker-ci."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from _style import GREEN, INK, ORANGE, figure, save


def _stack(ax, x, layers, title):
    ax.text(x + 2.1, 6.35, title, ha="center", va="center", fontsize=16, weight="bold")
    for i, (label, rebuilt) in enumerate(layers):
        y = 5.2 - i * 1.15
        color = ORANGE if rebuilt else GREEN
        ax.add_patch(FancyBboxPatch((x, y), 4.2, 0.9, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=color, alpha=0.22, edgecolor=color, lw=2))
        ax.text(x + 0.2, y + 0.45, label, ha="left", va="center", fontsize=13.5, color=INK,
                family="monospace")
        ax.text(x + 4.0, y + 0.45, "rebuilt" if rebuilt else "cached", ha="right", va="center",
                fontsize=12.5, color=color, weight="bold")


def fig_layer_cache():
    """Cache hits after a source edit: bad vs good instruction order."""
    fig, ax = figure(11, 6.2)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0.3, 6.8)
    ax.axis("off")

    _stack(ax, 0.2, [
        ("FROM python-slim", False),
        ("COPY . .", True),
        ("RUN install deps", True),
        ("CMD uvicorn ...", True),
    ], "Source copied first")
    _stack(ax, 5.6, [
        ("FROM python-slim", False),
        ("COPY uv.lock ...", False),
        ("RUN install deps", False),
        ("COPY src/ ...", True),
    ], "Lockfile copied first")

    ax.text(2.3, 0.75, "every edit: minutes", ha="center", fontsize=14, color=ORANGE,
            weight="bold")
    ax.text(7.7, 0.75, "every edit: seconds", ha="center", fontsize=14, color=GREEN,
            weight="bold")
    fig.suptitle("After editing one line of app.py", x=0.03, ha="left", fontsize=20,
                 weight="bold")
    return save(fig, "m03-03-layer-cache")
