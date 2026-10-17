"""Supplementary Fig. 2 - Pairwise discrimination by the multilabel model in intensive care: AUC of each output for
patients with the row condition but not the column condition vs the reverse, matched 1:1 on sex and age
(19_train_multilabel.py).

Input: data/figS2_multilabel.csv
Usage: python draw_figS2.py [--data-dir DIR] [--out-dir DIR]
"""
from pathlib import Path
import argparse, hashlib, json, platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
INK, MUTED, FRAME, BAND = "#1a1a1a", "#666666", "#4d4d4d", "#f1f1f1"
ICU = "#2f6b85"
CMAP = LinearSegmentedColormap.from_list("seq", ["#f4f6f7", "#9fbccb", ICU, "#173b4b"])
VMIN, VMAX = 0.5, 0.85
ORDER = ["Cirrhosis", "HF", "Sepsis", "ESRD"]
LABEL = {"Cirrhosis": "Cirrhosis", "HF": "Heart failure", "Sepsis": "Sepsis", "ESRD": "ESRD"}
FS = 6.4
WIDTH_MM, HEIGHT_MM = 120, 76
FONTS = ["Arial", "Liberation Sans", "DejaVu Sans"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONTS,
    "font.size": FS, "axes.labelsize": FS + 0.4, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "axes.linewidth": 0.7, "axes.edgecolor": FRAME, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
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


def lum(c):
    v = np.asarray(to_rgb(c)); v = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return float(v @ [0.2126, 0.7152, 0.0722])


def read_data(d):
    name = "figS2_multilabel.csv"
    M = pd.read_csv(d / name, encoding="utf-8-sig")
    assert len(M) == 12 and set(M.model_output) == set(ORDER) and set(M.comparator) == set(ORDER)
    assert (M.model_output != M.comparator).all() and ((M.lo <= M.auc) & (M.auc <= M.hi)).all()
    return M.set_index(["model_output", "comparator"]), [name]


def draw(data_dir, out_dir):
    font = font_in_use()
    if font not in ("Arial", "Liberation Sans"):
        print(f"[warning] Arial not found; using {font}. Text is wider than intended.")
    M, names = read_data(data_dir)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    cell = 15.0
    x0, top = 30.0, 12.0
    ax = fig.add_axes([x0 / WIDTH_MM, 1 - (top + 4 * cell) / HEIGHT_MM, 4 * cell / WIDTH_MM, 4 * cell / HEIGHT_MM])
    norm = Normalize(VMIN, VMAX)
    for i, r in enumerate(ORDER):
        for j, c in enumerate(ORDER):
            if r == c:
                ax.add_patch(Rectangle((j, i), 1, 1, fc=BAND, ec="white", lw=1.2))
                ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", color=MUTED, fontsize=FS)
                continue
            v = M.loc[(r, c)]
            col = CMAP(norm(v.auc))
            ax.add_patch(Rectangle((j, i), 1, 1, fc=col, ec="white", lw=1.2))
            tc = "white" if lum(col) < 0.25 else INK
            ax.text(j + 0.5, i + 0.42, f"{v.auc:.3f}", ha="center", va="center", color=tc, fontsize=FS + 0.4,
                    fontweight="bold")
            ax.text(j + 0.5, i + 0.68, f"{v.lo:.3f}–{v.hi:.3f}", ha="center", va="center", color=tc, fontsize=FS - 1.2)
    ax.add_patch(Rectangle((0.02, 0.02), 3.96, 0.96, fc="none", ec=INK, lw=1.0, zorder=5))
    ax.set(xlim=(0, 4), ylim=(4, 0))
    ax.set_xticks(np.arange(4) + 0.5); ax.set_xticklabels([LABEL[c] for c in ORDER])
    ax.set_yticks(np.arange(4) + 0.5); ax.set_yticklabels([LABEL[c] for c in ORDER])
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.tick_params(length=0, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Comparator condition", labelpad=5, fontweight="bold")
    ax.set_ylabel("Model output (condition detected)", labelpad=5, fontweight="bold")
    cax = fig.add_axes([(x0 + 4 * cell + 6) / WIDTH_MM, 1 - (top + 3.2 * cell) / HEIGHT_MM, 2.2 / WIDTH_MM, 2.4 * cell / HEIGHT_MM])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, ticks=[0.5, 0.6, 0.7, 0.8])
    cb.outline.set_linewidth(0.5); cb.ax.tick_params(labelsize=FS - 0.4, length=2, width=0.5)
    cb.set_label("AUC", fontsize=FS, labelpad=3)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "figS2_multilabel"
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out_dir / "figS2_multilabel_preview.png", dpi=300, facecolor="white")
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
