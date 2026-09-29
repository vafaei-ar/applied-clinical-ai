"""Figures for lesson m03-02-serving-api."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, PURPLE, RED, figure, save


def _box(ax, x, y, w, h, text, color, fs=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=color, alpha=0.16, edgecolor=color, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def _arrow(ax, x0, y0, x1, y1, color=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2})


def fig_request_flow():
    """Path of one request through the prediction service."""
    fig, ax = figure(11, 5.8)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    _box(ax, 0.1, 3.4, 1.8, 1.2, "EHR\nclient", MUTED)
    _box(ax, 2.4, 3.4, 2.0, 1.2, "request ID\nmiddleware", PURPLE)
    _box(ax, 4.9, 3.4, 2.0, 1.2, "Pydantic\nvalidation", BLUE)
    _box(ax, 7.4, 3.4, 1.6, 1.2, "model\n(loaded\nonce)", GREEN, fs=13)
    _box(ax, 9.3, 3.4, 1.6, 1.2, "200\nresponse", GREEN)

    _arrow(ax, 1.9, 4.0, 2.4, 4.0)
    _arrow(ax, 4.4, 4.0, 4.9, 4.0)
    _arrow(ax, 6.9, 4.0, 7.4, 4.0)
    _arrow(ax, 9.0, 4.0, 9.3, 4.0)

    _box(ax, 4.9, 1.1, 2.0, 1.1, "422\nwith error list", RED)
    _arrow(ax, 5.9, 3.4, 5.9, 2.2, RED)
    ax.text(6.05, 2.8, "out of range,\nunknown field", fontsize=12.5, color=RED, va="center")

    _box(ax, 7.4, 1.1, 3.5, 1.1, "audit record: request ID,\ninput, output, model version",
         PURPLE, fs=13)
    _arrow(ax, 8.2, 3.4, 8.2, 2.2, PURPLE)

    ax.text(0.1, 5.1, "startup (lifespan): load model  →  /readyz turns 200", fontsize=14,
            color=GREEN, weight="bold")
    ax.text(0.1, 0.35, "every response carries X-Request-ID", fontsize=13, color=MUTED,
            style="italic")
    ax.set_title("POST /v1/predict, step by step", loc="left", pad=4)
    return save(fig, "m03-02-request-flow")
