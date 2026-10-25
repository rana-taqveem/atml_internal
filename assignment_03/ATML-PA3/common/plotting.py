"""Shared matplotlib style for report figures (print, NeurIPS single/double column).

Colours follow the entity (method/condition), never its rank, and are assigned from a
fixed, colour-vision-checked categorical order.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8985"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Task 1 conditions -> fixed colour
T1_COLORS = {"baseline": NEUTRAL, "full_mid": "#9cc3ef", "full": SERIES[0], "l2sp": SERIES[1],
             "lora4": SERIES[2], "lora16": SERIES[3], "oplora16": SERIES[4]}
T1_LABELS = {"baseline": "Baseline", "full_mid": "Full FT (mid)", "full": "Full FT",
             "l2sp": "Full FT + L2-SP", "lora4": "LoRA-4", "lora16": "LoRA-16",
             "oplora16": "OPLoRA-16"}

SINGLE_COL = 3.25   # inches (NeurIPS column ~ 5.5in text width; two-up figures 2.7in)
TEXT_WIDTH = 5.5


def setup():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.4, "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.dpi": 300, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    })


def save(fig, path_no_ext):
    from .utils import rel

    p = rel(path_no_ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"))
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
