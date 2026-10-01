"""Figures for lesson m06-03-platform-labs."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch, Rectangle

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, figure, save


def _box(ax, x, y, w, h, color, alpha=0.14, lw=2.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="none", alpha=alpha))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec=color, lw=lw))


def _arrow(ax, x0, y0, x1, y1, color=INK, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": style, "color": color, "lw": 2.2})


def fig_spark_cluster():
    """Driver plans the job; executors run one task per partition."""
    fig, ax = figure(10, 8.2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    _box(ax, 2.2, 6.3, 5.6, 1.8, BLUE)
    ax.text(5.0, 7.6, "Driver", ha="center", fontsize=19, weight="bold", color=BLUE)
    ax.text(5.0, 6.85, "your code · builds the plan (DAG)\nsplits jobs → stages → tasks",
            ha="center", va="center", fontsize=14)

    for i, x in enumerate([0.3, 3.5, 6.7]):
        _box(ax, x, 1.2, 3.0, 3.6, ORANGE)
        ax.text(x + 1.5, 4.3, f"Executor {i + 1}", ha="center", fontsize=16,
                weight="bold", color=ORANGE)
        for j in range(3):
            ax.add_patch(Rectangle((x + 0.3, 3.35 - j * 0.75), 2.4, 0.55, fc="white",
                                   ec=ORANGE, lw=1.6))
            ax.text(x + 1.5, 3.62 - j * 0.75, f"task · partition {i * 3 + j + 1}",
                    ha="center", va="center", fontsize=12.5)
        _arrow(ax, 5.0, 6.3, x + 1.5, 4.85)

    ax.text(5.0, 0.55, "shuffle (groupBy, join) = data moves between executors\n"
            "= a new stage", ha="center", va="center", fontsize=14, color=PURPLE)
    ax.set_title("Spark: one driver, many executors", loc="left", pad=4)
    return save(fig, "m06-03-spark-cluster")


def fig_k8s_request_path():
    """Ingress → Service → pods of a Deployment, with the HPA adjusting replicas."""
    fig, ax = figure(10, 8.2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    ax.text(0.3, 7.7, "EHR client", fontsize=16, weight="bold", va="center")
    _arrow(ax, 2.2, 7.7, 3.2, 7.7)
    _box(ax, 3.3, 7.1, 3.4, 1.2, PURPLE)
    ax.text(5.0, 7.7, "Ingress\nHTTPS, routing", ha="center", va="center", fontsize=14)

    _arrow(ax, 5.0, 7.1, 5.0, 6.35)
    _box(ax, 3.3, 5.1, 3.4, 1.2, BLUE)
    ax.text(5.0, 5.7, "Service\nstable name, load balance", ha="center", va="center",
            fontsize=14)

    _box(ax, 0.3, 0.9, 9.4, 3.5, GREEN, alpha=0.08)
    ax.text(5.0, 1.2, "Deployment: stroke-readmit-api (replicas: 3)", fontsize=14,
            weight="bold", color=GREEN, va="center", ha="center")
    for i, x in enumerate([0.9, 3.9, 6.9]):
        _box(ax, x, 1.7, 2.2, 2.3, GREEN, alpha=0.2)
        ax.text(x + 1.1, 3.25, f"Pod {i + 1}", ha="center", fontsize=15, weight="bold")
        ax.text(x + 1.1, 2.45, "container\nFastAPI :8000", ha="center", va="center",
                fontsize=12.5)
        _arrow(ax, 5.0, 5.1, x + 1.1, 4.05)

    _box(ax, 7.4, 5.1, 2.3, 1.2, ORANGE)
    ax.text(8.55, 5.7, "HPA\ntarget CPU 70%", ha="center", va="center", fontsize=14)
    _arrow(ax, 8.55, 5.1, 8.55, 4.45, color=ORANGE)
    ax.text(8.7, 4.72, "scale", fontsize=12.5, color=ORANGE, va="center")

    ax.text(0.3, 0.35, "Pods are disposable; the Deployment keeps the count right.",
            fontsize=13, color=MUTED, style="italic")
    ax.set_title("Kubernetes: the path of one request", loc="left", pad=4)
    return save(fig, "m06-03-k8s-request-path")
