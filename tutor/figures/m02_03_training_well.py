"""Figures for lesson m02-03-training-well."""

from __future__ import annotations

import numpy as np

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, figure, save


def fig_lr_schedule():
    """Linear warmup followed by cosine decay, stepped per batch."""
    fig, ax = figure(10, 5.6)
    total, warm, peak = 3000, 300, 1e-3
    step = np.arange(total)
    lr = np.where(
        step < warm,
        peak * (0.1 + 0.9 * step / warm),
        peak * 0.5 * (1 + np.cos(np.pi * (step - warm) / (total - warm))),
    )
    ax.axvspan(0, warm, color=ORANGE, alpha=0.14, lw=0)
    ax.plot(step, lr * 1e3, color=BLUE, lw=3)
    ax.text(warm / 2, 1.08, "warmup", ha="center", fontsize=14, color=ORANGE, weight="bold")
    ax.text(1650, 0.72, "cosine decay", fontsize=14, color=BLUE, weight="bold")
    ax.set_xlim(0, total)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("Optimizer step (batch)")
    ax.set_ylabel("Learning rate (×10⁻³)")
    ax.set_title("Warm up, then decay", loc="left")
    return save(fig, "m02-03-lr-schedule")


def fig_learning_curve():
    """Train and validation loss per epoch, with the best epoch and the patience window."""
    rng = np.random.default_rng(7)
    epochs = np.arange(1, 41)
    train = 0.24 + 0.22 * np.exp(-epochs / 7) - 0.0018 * epochs
    val = 0.262 + 0.20 * np.exp(-epochs / 6) + 0.00016 * (epochs - 14).clip(0) ** 2
    val = val + rng.normal(0, 0.0035, len(epochs))
    patience = 8
    best = int(np.argmin(val[:30]))
    stop = best + patience

    fig, ax = figure(10, 6.2)
    ax.axvspan(epochs[best], epochs[stop], color=ORANGE, alpha=0.13, lw=0)
    ax.plot(epochs[: stop + 1], train[: stop + 1], color=BLUE, lw=3)
    ax.plot(epochs[: stop + 1], val[: stop + 1], color=GREEN, lw=3)
    ax.plot(epochs[stop:], train[stop:], color=BLUE, lw=2, ls=":", alpha=0.5)
    ax.plot(epochs[stop:], val[stop:], color=GREEN, lw=2, ls=":", alpha=0.5)

    ax.plot(epochs[best], val[best], "o", ms=13, color=GREEN, zorder=5,
            markeredgecolor="white", markeredgewidth=2)
    ax.annotate(f"best epoch {epochs[best]}\nsave + restore this", xy=(epochs[best], val[best]),
                xytext=(epochs[best] - 11, val[best] - 0.045), fontsize=14, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1.2})
    ax.axvline(epochs[stop], color=RED, lw=2)
    ax.text(epochs[stop] + 0.5, 0.43, f"stop at {epochs[stop]}\n(patience {patience})",
            fontsize=14, color=RED, va="top")
    ax.text(epochs[best] + patience / 2, 0.44, "no\nimprovement", va="top", ha="center", fontsize=12.5,
            color=ORANGE)

    ax.text(epochs[stop] - 0.5, train[stop] - 0.012, "train", color=BLUE, fontsize=15,
            weight="bold", ha="right", va="top")
    ax.text(epochs[stop] - 0.5, val[stop] + 0.012, "validation", color=GREEN, fontsize=15,
            weight="bold", ha="right", va="bottom")
    ax.set_xlim(0, 41)
    ax.set_ylim(0.16, 0.45)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (BCE)")
    ax.set_title("Early stopping on the validation curve", loc="left")
    return save(fig, "m02-03-learning-curve")
