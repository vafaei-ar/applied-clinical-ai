"""Figures for lesson m01-01-data-sources."""

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch

from _style import BLUE, GREEN, INK, MUTED, ORANGE, RED, figure, save

BOX_W, BOX_H = 3.3, 1.75


def _table(ax, x, y, name, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), BOX_W, BOX_H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec=color, alpha=0.13, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), BOX_W, BOX_H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec=color, lw=2.2))
    ax.text(x + 0.18, y + BOX_H - 0.3, name, fontsize=17, weight="bold", color=color, va="center")
    for i, line in enumerate(lines):
        ax.text(x + 0.18, y + BOX_H - 0.78 - i * 0.42, line, fontsize=13, color=INK, va="center",
                family="monospace")


def _link(ax, x1, y1, x2, y2, label=None, lx=None, ly=None):
    ax.plot([x1, x2], [y1, y2], color=MUTED, lw=2, zorder=0)
    ax.plot(x2, y2, "o", color=MUTED, ms=7, zorder=1)
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2,
                label, fontsize=12, color=MUTED, ha="center", va="center",
                bbox={"fc": "white", "ec": "none", "pad": 1.5})


def fig_schema():
    """ER-style sketch of the repo's eight tables: a patient hub and an encounter hub."""
    fig, ax = figure(11.5, 7.8)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    xl, xm, xr = 0.2, 4.6, 9.1
    yt, ymid, yb = 6.3, 3.35, 0.4

    _table(ax, xl, yt, "medications", ["patient_id  FK", "start / end_date"], BLUE)
    _table(ax, xm, yt, "labs", ["patient_id  FK", "lab_date, test_name"], BLUE)
    _table(ax, xl, ymid, "patients", ["patient_id  PK", "birth_date, sex"], BLUE)
    _table(ax, xl, yb, "coverage", ["patient_id  FK", "coverage_start/end"], BLUE)
    _table(ax, xm, ymid, "encounters", ["encounter_id PK", "patient_id  FK"], ORANGE)
    _table(ax, xr, yt, "diagnoses", ["encounter_id FK", "icd10_code"], GREEN)
    _table(ax, xr, ymid, "procedures", ["encounter_id FK", "cpt_hcpcs_code"], GREEN)
    _table(ax, xr, yb, "claims", ["encounter_id FK", "paid_amount"], GREEN)

    cx = xl + BOX_W / 2
    _link(ax, cx, ymid + BOX_H, cx, yt, "1 : many")
    _link(ax, cx, ymid, cx, yb + BOX_H, "1 : many")
    _link(ax, xl + BOX_W, ymid + BOX_H - 0.1, xm + 0.4, yt)
    _link(ax, xl + BOX_W, ymid + BOX_H / 2, xm, ymid + BOX_H / 2, "1 : many")
    ex = xm + BOX_W
    ey = ymid + BOX_H / 2
    _link(ax, ex, ey, xr, yt + BOX_H / 2, "1 : many", lx=8.5, ly=5.7)
    _link(ax, ex, ey, xr, ymid + BOX_H / 2)
    _link(ax, ex, ey, xr, yb + BOX_H / 2, "1 : 1 here", lx=8.5, ly=2.75)

    ax.text(xm + BOX_W / 2 - 0.3, 1.2, "Labs and medications\nlink to the patient only,\nnot to an encounter",
            ha="center", va="center", fontsize=13, color=BLUE, style="italic")
    ax.set_title("Two hubs: patient-level and encounter-level tables", loc="left", pad=6)
    return save(fig, "m01-01-schema")


def fig_icd10_tree():
    """ICD-10-CM prefix hierarchy around ischemic stroke, with the I69 sequela trap."""
    fig, ax = figure(11, 7)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    def node(x, y, code, label, color, w=2.5):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.42), w, 0.84,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=color, alpha=0.14, ec=color, lw=2))
        ax.text(x, y + 0.13, code, ha="center", va="center", fontsize=16, weight="bold", color=color)
        ax.text(x, y - 0.2, label, ha="center", va="center", fontsize=11.5, color=INK)

    def edge(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1 - 0.42, y2 + 0.42], color=MUTED, lw=1.8, zorder=0)

    node(5.5, 6.6, "I60–I69", "cerebrovascular diseases", MUTED, w=3.4)
    kids = [(1.5, "I60", "subarachnoid hem.", MUTED), (4.1, "I61", "intracerebral hem.", MUTED),
            (6.9, "I63", "cerebral infarction", GREEN), (9.5, "I69", "SEQUELAE of stroke", RED)]
    for x, code, label, color in kids:
        edge(5.5, 6.6, x, 4.9)
        node(x, 4.9, code, label, color)
    for x, code, label in [(5.6, "I63.4", "due to embolism"), (8.2, "I63.9", "unspecified")]:
        edge(6.9, 4.9, x, 3.2)
        node(x, 3.2, code, label, GREEN, w=2.3)
    edge(5.6, 3.2, 5.6, 1.5)
    node(5.6, 1.5, "I63.411", "embolism, right MCA", GREEN, w=2.9)

    ax.text(0.2, 3.35, "LIKE 'I63%'\ncatches the whole\ngreen subtree", fontsize=14, color=GREEN,
            va="center", weight="bold")
    ax.text(10.9, 3.35, "LIKE 'I6%' also\ncatches I69:\npast stroke,\nnot a new one",
            fontsize=13.5, color=RED, va="center", ha="right")
    ax.text(0.2, 0.35, "chapter letter · 3-char category · detail after the dot   (simplified)",
            fontsize=12, color=MUTED, style="italic")
    ax.set_title("ICD-10-CM is a prefix tree", loc="left", pad=4)
    return save(fig, "m01-01-icd10-tree")
