"""Figures for lesson m02-07-experiment-tracking."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, figure, save


def _box(ax, x, y, w, h, color, title, body):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=color, alpha=0.14, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec=color, lw=2.2))
    ax.text(x + 0.25, y + h - 0.22, title, ha="left", va="top", fontsize=17,
            weight="bold", color=color)
    ax.text(x + 0.25, y + h - 0.72, body, ha="left", va="top", fontsize=14, color=INK,
            linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2.2})
    if label:
        ax.text((x0 + x1) / 2 + 0.15, (y0 + y1) / 2, label, fontsize=13, color=MUTED,
                va="center")


def fig_lineage():
    """From data snapshot to the deployed model, with the IDs that link each hop."""
    fig, ax = figure(10, 8.4)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.3)
    ax.axis("off")

    _box(ax, 0.2, 6.6, 9.6, 1.6, BLUE, "1  Data snapshot",
         "stroke_features, built from sql/02_features.sql\n"
         "logged as a dataset input with its digest (hash)")
    _box(ax, 0.2, 4.45, 9.6, 1.6, ORANGE, "2  MLflow run",
         "git commit  ·  params  ·  metrics (val only)\n"
         "artifacts: model, reliability plot, env spec")
    _box(ax, 0.2, 2.3, 5.9, 1.6, PURPLE, "3  Registered model",
         "stroke-readmit  version 7\n"
         "points back to the run that made it")
    _box(ax, 6.5, 2.3, 3.3, 1.6, GREEN, "@champion",
         "alias → v7\n@challenger → v8")
    _box(ax, 0.2, 0.15, 9.6, 1.6, INK, "4  Service (Module 03)",
         "loads models:/stroke-readmit@champion\nlogs model version with each prediction")

    _arrow(ax, 1.4, 6.6, 1.4, 6.1, "logged as input")
    _arrow(ax, 1.4, 4.45, 1.4, 3.95, "register")
    _arrow(ax, 6.1, 3.1, 6.5, 3.1)
    _arrow(ax, 8.15, 2.3, 8.15, 1.8, "resolve alias")
    ax.set_title("Every hop keeps an ID you can follow back", loc="left", pad=6)
    return save(fig, "m02-07-lineage")
