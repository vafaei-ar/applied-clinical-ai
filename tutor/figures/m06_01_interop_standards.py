"""Figures for lesson m06-01-interop-standards."""

from __future__ import annotations

from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, figure, save

ROWS = [
    ("patients", "person", ""),
    ("coverage", "observation_period", "+ payer_plan_period"),
    ("encounters", "visit_occurrence", ""),
    ("diagnoses", "condition_occurrence", "ICD-10-CM → SNOMED CT"),
    ("procedures", "procedure_occurrence", "CPT4 / HCPCS"),
    ("medications", "drug_exposure", "name → RxNorm"),
    ("labs", "measurement", "test_name → LOINC"),
    ("claims", "cost", ""),
]


def _pill(ax, x, y, w, text, color):
    ax.add_patch(FancyBboxPatch((x, y - 0.3), w, 0.6, boxstyle="round,pad=0.02,rounding_size=0.18",
                                facecolor=to_rgba(color, 0.12), edgecolor=color, lw=2))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=15, color=INK,
            family="monospace")


def fig_omop_mapping():
    """Module 01 tables on the left, the OMOP CDM tables they load into on the right."""
    fig, ax = figure(11, 8.2)
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.6, 9.0)
    ax.axis("off")

    ax.text(1.6, 8.55, "Module 01 schema", ha="center", fontsize=17, weight="bold", color=BLUE)
    ax.text(8.9, 8.55, "OMOP CDM", ha="center", fontsize=17, weight="bold", color=GREEN)

    for i, (src, dst, note) in enumerate(ROWS):
        y = 7.7 - i * 1.02
        _pill(ax, 0.1, y, 3.0, src, BLUE)
        _pill(ax, 6.6, y, 4.3, dst, GREEN)
        ax.add_patch(FancyArrowPatch((3.2, y), (6.5, y), arrowstyle="-|>", mutation_scale=20,
                                     color=MUTED, lw=1.8))
        if note:
            ax.text(4.85, y + 0.12, note, ha="center", va="bottom", fontsize=12.5, color=INK)

    ax.text(0.1, -0.45, "zip3 → location · clinical event tables key on person_id · codes → concept_id",
            fontsize=12.5, color=MUTED, style="italic")
    return save(fig, "m06-01-omop-mapping")


if __name__ == "__main__":
    print(fig_omop_mapping())
