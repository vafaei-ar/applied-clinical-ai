"""Shared figure style: phone-legible, consistent colors, one place to save outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMAGES = Path(__file__).resolve().parents[1] / "course" / "images"

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
BLUE = "#2f6fdf"
ORANGE = "#e8833a"
GREEN = "#2e9e6a"
RED = "#d64545"
PURPLE = "#7c5cc4"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, RED]

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.frameon": False,
        "font.size": 15,
        "text.color": INK,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    }
)


def figure(width: float = 10, height: float = 6.5):
    """Create a figure sized for a phone screen (landscape, ~1500px wide at save dpi)."""
    return plt.subplots(figsize=(width, height))


def save(fig, name: str) -> Path:
    """Save to course/images/<name>.png and close the figure."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path
