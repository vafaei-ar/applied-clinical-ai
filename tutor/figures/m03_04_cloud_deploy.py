"""Figures for lesson m03-04-cloud-deploy."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, PURPLE, figure, save


def _box(ax, x, y, w, h, text, color, fs=13.5, alpha=0.16):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=color, alpha=alpha, edgecolor=color, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def _arrow(ax, x0, y0, x1, y1, color=INK, label=None, lx=None, ly=None, ha="center"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2})
    if label:
        ax.text(lx if lx is not None else (x0 + x1) / 2, ly if ly is not None else (y0 + y1) / 2,
                label, ha=ha, va="center", fontsize=12, color=color)


def fig_azure_architecture():
    """Registry, compute, identity, secrets, network, and logs for the readmission service."""
    fig, ax = figure(11, 7.6)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    # VNet boundary
    ax.add_patch(FancyBboxPatch((3.0, 0.4), 7.8, 5.6, boxstyle="round,pad=0.02,rounding_size=0.2",
                                facecolor="none", edgecolor=BLUE, lw=2, ls="--"))
    ax.text(10.6, 5.7, "hospital VNet (private)", ha="right", fontsize=13, color=BLUE, weight="bold")

    _box(ax, 0.1, 6.3, 2.5, 1.0, "GitHub Actions\n(OIDC, no secret)", MUTED)
    _box(ax, 4.2, 6.3, 2.6, 1.0, "Container Registry\n(ACR)", ORANGE)
    _arrow(ax, 2.6, 6.8, 4.2, 6.8, INK, "AcrPush", ly=7.1)

    _box(ax, 0.1, 2.6, 2.3, 1.2, "EHR\n(on-prem)", MUTED)
    _box(ax, 4.0, 2.4, 3.0, 1.6, "Container App\nreadmit-api\n+ managed identity", GREEN, fs=14)
    _arrow(ax, 2.4, 3.2, 4.0, 3.2, INK, "VPN /\nExpressRoute", lx=1.3, ly=2.1)
    _arrow(ax, 5.5, 6.3, 5.5, 4.0, ORANGE, "AcrPull", lx=5.6, ly=5.2, ha="left")

    _box(ax, 8.0, 4.1, 2.7, 1.3, "Key Vault\nrole: Secrets User\nprivate endpoint", PURPLE, fs=12.5)
    _box(ax, 8.0, 2.5, 2.7, 1.3, "Blob: models\nrole: Data Reader\nprivate endpoint", PURPLE, fs=12.5)
    _box(ax, 8.0, 0.9, 2.7, 1.2, "Log Analytics\n+ audit store", PURPLE, fs=13)
    _arrow(ax, 7.0, 3.7, 8.0, 4.7, PURPLE)
    _arrow(ax, 7.0, 3.2, 8.0, 3.15, PURPLE)
    _arrow(ax, 7.0, 2.7, 8.0, 1.5, PURPLE)

    ax.text(4.0, 1.3, "internal environment\n(no public endpoint)\npublic access disabled\non data services",
            fontsize=12.5, color=MUTED, style="italic", va="center")
    ax.set_title("The readmission service on Azure", loc="left", pad=4)
    return save(fig, "m03-04-azure-architecture")
