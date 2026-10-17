"""Supplementary Fig. 1 - Calibration of frozen models in the other health system before and after local recalibration
(deciles of predicted probability): a ICU-trained models in the ED; b ED-trained models in the ICU.

Input: data/figS1_calibration.csv (21_paper_analyses.py, external section)
Usage: python draw_figS1.py [--data-dir DIR] [--out-dir DIR]
"""
from pathlib import Path
import argparse, hashlib, json, platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
COL = {"MIMIC": "#2f6b85", "MC-MED": "#c0643c"}
MK = {"MIMIC": "o", "MC-MED": "s"}
MS = {"MIMIC": 3.4, "MC-MED": 3.1}
INK, MUTED, FRAME, REF = "#1a1a1a", "#666666", "#4d4d4d", "#8c8c8c"
DASH = (0, (3, 1.6))
FS = 6.4
WIDTH_MM, HEIGHT_MM = 183, 86
FONTS = ["Arial", "Liberation Sans", "DejaVu Sans"]
PANELS = [("a", "b", "MC-MED", "ED (MC-MED), ICU-trained model", 0.0661, 0.0103),
          ("b", "c", "MIMIC", "ICU (MIMIC), ED-trained model", 0.3399, 0.0096)]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONTS,
    "font.size": FS, "axes.labelsize": FS + 0.4, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "axes.linewidth": 0.7, "axes.edgecolor": FRAME, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.size": 2.5, "xtick.major.width": 0.6,
    "ytick.major.size": 2.5, "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "text.hinting": "no_hinting"})


def font_in_use():
    from matplotlib import font_manager as fm
    for f in FONTS:
        try:
            fm.findfont(fm.FontProperties(family=f), fallback_to_default=False); return f
        except ValueError:
            continue
    return "matplotlib default"


def read_data(d):
    name = "figS1_calibration.csv"
    C = pd.read_csv(d / name, encoding="utf-8-sig")
    assert len(C) == 40 and set(C.version) == {"Frozen", "Locally recalibrated"} and set(C.panel) == {"b", "c"}
    assert ((C.predicted > 0) & (C.predicted < 1) & (C.observed >= 0) & (C.observed <= 1)).all()
    for p in ("b", "c"):
        for v in ("Frozen", "Locally recalibrated"):
            g = C[(C.panel == p) & (C.version == v)]
            assert len(g) == 10 and sorted(g.decile) == list(range(1, 11)) and g.slope.nunique() == 1
    return C, [name]


def draw(data_dir, out_dir):
    font = font_in_use()
    if font not in ("Arial", "Liberation Sans"):
        print(f"[warning] Arial not found; using {font}. Text is wider than intended.")
    C, names = read_data(data_dir)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    T = lambda mm: 1 - mm / HEIGHT_MM
    H = lambda mm: mm / HEIGHT_MM
    side = 62.0
    LIM = 0.85
    for j, (letter, panel, test, title, ece_f, ece_r) in enumerate(PANELS):
        x0 = (0.075 + j * 0.51) * WIDTH_MM
        ax = fig.add_axes([x0 / WIDTH_MM, T(8 + side), side / WIDTH_MM, H(side)])
        fig.text((x0 - 12) / WIDTH_MM, T(1.5), letter, fontsize=9, fontweight="bold", va="top")
        ax.plot([0, LIM], [0, LIM], color=REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
        slopes = {}
        for v, fill, ls in (("Frozen", "white", DASH), ("Locally recalibrated", COL[test], "-")):
            g = C[(C.panel == panel) & (C.version == v)].sort_values("decile")
            slopes[v] = g.slope.iloc[0]
            ax.plot(g.predicted, g.observed, color=COL[test], lw=1.0, ls=ls, marker=MK[test], ms=MS[test],
                    mfc=fill, mec=COL[test], mew=0.8, zorder=3)
        ax.set(xlim=(0, LIM), ylim=(0, LIM), xticks=np.arange(0, 0.81, 0.2), yticks=np.arange(0, 0.81, 0.2))
        ax.set_aspect("equal")
        ax.set_xlabel("Predicted probability", labelpad=2); ax.set_ylabel("Observed proportion", labelpad=2)
        ax.set_title(title, fontsize=FS + 0.2, fontweight="bold", color=COL[test], loc="left", pad=3)
        ax.text(0.04, 0.96, "Calibration slope / ECE", transform=ax.transAxes, ha="left", va="top",
                fontsize=FS - 0.4, color=MUTED)
        ax.text(0.04, 0.89, f"Frozen: {slopes['Frozen']:.2f} / {ece_f:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=FS - 0.2)
        ax.text(0.04, 0.83, f"Recalibrated: {slopes['Locally recalibrated']:.2f} / {ece_r:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=FS - 0.2)
        for s in ax.spines.values():
            s.set_visible(True)
    hs = [Line2D([], [], color=INK, lw=1.0, ls=DASH, marker="o", ms=3.2, mfc="white", mec=INK, label="Frozen"),
          Line2D([], [], color=INK, lw=1.0, marker="o", ms=3.2, mfc=INK, mec=INK, label="Locally recalibrated"),
          Line2D([], [], color=REF, lw=0.6, ls=(0, (3, 2)), label="Ideal calibration"),
          Line2D([], [], color=COL["MIMIC"], lw=0, marker="o", ms=3.4, label="ICU (MIMIC)"),
          Line2D([], [], color=COL["MC-MED"], lw=0, marker="s", ms=3.1, label="ED (MC-MED)")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=FS - 0.2,
               handlelength=2.2, columnspacing=1.6, borderaxespad=0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "figS1_calibration"
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out_dir / "figS1_calibration_preview.png", dpi=300, facecolor="white")
    plt.close(fig)
    meta = {"canvas_mm": [WIDTH_MM, HEIGHT_MM], "font": font,
            "source_hashes": {n: hashlib.sha256((data_dir / n).read_bytes()).hexdigest() for n in names},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "matplotlib": matplotlib.__version__}}
    prefix.with_suffix(".data-check.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT.parent / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out")
    a = ap.parse_args()
    draw(a.data_dir, a.out_dir)
