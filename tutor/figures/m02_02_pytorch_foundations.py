"""Figures for lesson m02-02-pytorch-foundations."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, RED, figure, save


def _box(ax, x, y, w, h, title, code, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=color, alpha=0.14, edgecolor=color, lw=2))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=15,
            weight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.28, code, ha="center", va="center", fontsize=13,
            family="monospace", color=INK)


def _arrow(ax, x0, y0, x1, y1, color=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.2})


def fig_training_step():
    """One optimization step as a cycle of five calls."""
    fig, ax = figure(11, 6.4)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    w, h = 3.0, 1.35
    # top row, left to right
    _box(ax, 0.3, 4.3, w, h, "1. clear grads", "opt.zero_grad()", MUTED)
    _box(ax, 4.0, 4.3, w, h, "2. forward", "logits = model(X)", BLUE)
    _box(ax, 7.7, 4.3, w, h, "3. loss", "loss_fn(logits, y)", ORANGE)
    # bottom row, right to left
    _box(ax, 7.7, 1.2, w, h, "4. backward", "loss.backward()", PURPLE)
    _box(ax, 4.0, 1.2, w, h, "5. update", "opt.step()", GREEN)
    _box(ax, 0.3, 1.2, w, h, "next batch", "for X, y in loader", INK)

    _arrow(ax, 3.35, 4.97, 3.95, 4.97)
    _arrow(ax, 7.05, 4.97, 7.65, 4.97)
    _arrow(ax, 9.2, 4.25, 9.2, 2.6)
    _arrow(ax, 7.65, 1.87, 7.05, 1.87)
    _arrow(ax, 3.95, 1.87, 3.35, 1.87)
    _arrow(ax, 1.8, 2.6, 1.8, 4.25)

    ax.text(9.35, 3.42, "fills\n.grad", fontsize=12.5, color=PURPLE, va="center")
    ax.text(5.5, 3.42, "no zero_grad  →  grads from\nevery batch add up", fontsize=13,
            color=RED, ha="center", va="center")
    ax.text(0.3, 0.45, "model.train() before the loop; model.eval() + torch.no_grad() to validate",
            fontsize=12.5, color=MUTED)
    ax.set_title("One training step, five calls", loc="left", pad=2)
    return save(fig, "m02-02-training-step")
