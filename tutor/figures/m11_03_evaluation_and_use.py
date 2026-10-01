"""Figures for lesson m11-03-evaluation-and-use."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _style import BLUE, GREEN, MUTED, ORANGE, PURPLE, RED, save

N_GOLD = 200


def simulate_eval(seed: int = 11):
    """A small simulated span-extraction evaluation against 200 gold spans.

    Each gold span gets one outcome. Matched spans (exact or boundary-only) also get an
    assertion outcome (negated / uncertain / family status right or wrong). The system also emits
    spurious spans that overlap no gold span.
    """
    rng = np.random.default_rng(seed)
    outcomes = ["exact", "boundary", "wrong_type", "missed"]
    p = [0.72, 0.12, 0.03, 0.13]
    draw = rng.choice(outcomes, size=N_GOLD, p=p)
    counts = {o: int((draw == o).sum()) for o in outcomes}

    matched = counts["exact"] + counts["boundary"]
    assertion_wrong = int(rng.binomial(matched, 0.10))
    spurious = int(rng.poisson(24))
    return counts, assertion_wrong, spurious


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def scores():
    """Precision, recall, and F1 under three matching rules from the same predictions."""
    c, aw, sp = simulate_eval()
    n_pred = c["exact"] + c["boundary"] + c["wrong_type"] + sp

    # Strict: exact boundaries and the right type.
    strict = _prf(c["exact"], n_pred - c["exact"], N_GOLD - c["exact"])
    # Relaxed: any overlap with the right type.
    tp_relaxed = c["exact"] + c["boundary"]
    relaxed = _prf(tp_relaxed, n_pred - tp_relaxed, N_GOLD - tp_relaxed)
    # Usable as a feature: relaxed match and the assertion (negated / uncertain / family) is right.
    tp_usable = tp_relaxed - aw
    usable = _prf(tp_usable, n_pred - tp_usable, N_GOLD - tp_usable)
    return {"strict": strict, "relaxed": relaxed, "usable": usable}, (c, aw, sp, n_pred)


def fig_span_errors():
    """Where 200 gold spans went, and how the score depends on the matching rule."""
    sc, (c, aw, sp, n_pred) = scores()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.6), gridspec_kw={"height_ratios": [1.25, 1]})

    correct = c["exact"] - 0  # exact-boundary matches
    rows = [
        ("Exact span + type", correct, GREEN),
        ("Boundary differs", c["boundary"], ORANGE),
        ("Wrong type", c["wrong_type"], PURPLE),
        ("Missed", c["missed"], RED),
        ("Spurious", sp, MUTED),
    ]
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    colors = [r[2] for r in rows][::-1]
    ax1.barh(labels, values, color=colors, height=0.62)
    for i, v in enumerate(values):
        ax1.text(v + 2, i, str(v), va="center", fontsize=15, weight="bold")
    ax1.set_xlim(0, max(values) * 1.18)
    ax1.set_xlabel("count (out of 200 gold spans; spurious has no gold)")
    ax1.tick_params(axis="y", labelsize=14.5)
    ax1.set_title("Where the errors went", loc="left", pad=8)

    names = ["Strict\nexact + type", "Relaxed\noverlap + type", "Usable\nplus assertion right"]
    f1s = [sc["strict"][2], sc["relaxed"][2], sc["usable"][2]]
    bars = ax2.bar(names, f1s, color=[BLUE, GREEN, ORANGE], width=0.55)
    for b, v in zip(bars, f1s, strict=True):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=17,
                 weight="bold")
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("F1")
    ax2.tick_params(axis="x", labelsize=14)
    ax2.set_title("Same predictions, three F1 values", loc="left", pad=8)
    return save(fig, "m11-03-span-errors")
