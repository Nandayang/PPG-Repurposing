"""Fig. 6 - Deep model and handcrafted features of the same waveform: a AUC of handcrafted feature sets and of the deep
model; b handcrafted, deep and combined models, internally and frozen in the other system; c AUC with only the first
1, 2, 4 or 10 retained 30-s windows per patient, or all windows.

Inputs (data/): figS3_inputs.csv, figS3b_combined.csv and figS3c_spotcheck.csv (31_deep_vs_handcrafted.py)
Usage: python draw_fig6.py [--data-dir DIR] [--out-dir DIR]
"""
from pathlib import Path
import argparse, hashlib, json, platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parent
COL = {"MIMIC": "#2f6b85", "MC-MED": "#c0643c"}
MK = {"MIMIC": "o", "MC-MED": "s"}
MS = {"MIMIC": 2.8, "MC-MED": 2.5}
NAME = {"MIMIC": "ICU (MIMIC)", "MC-MED": "ED (MC-MED)"}
INK, MUTED, FRAME, REF, BAND = "#1a1a1a", "#666666", "#4d4d4d", "#8c8c8c", "#f1f1f1"
HAND2 = "#b0b0b0"
COHORTS = ("MIMIC", "MC-MED")
SETS_A = [("Pulse rate", "Pulse rate\n(1)"), ("Pulse-rate variability", "Pulse-rate\nvariability (7)"),
          ("Handcrafted morphology", "Pulse\ncontour (11)"), ("All handcrafted features", "All hand-\ncrafted (21)"),
          ("Deep model (beat matrix, SSL)", "Deep\nmodel")]
WINDOWS = {"1": "30 s", "2": "1 min", "4": "2 min", "10": "5 min", "all": "All"}
FS = 6.4
WIDTH_MM, HEIGHT_MM = 183, 106
FONTS = ["Arial", "Liberation Sans", "DejaVu Sans"]

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


def shade(hex_, t):
    c = np.array(to_rgb(hex_))
    return tuple(c + (1 - c) * (-t)) if t < 0 else tuple(c * (1 - t))


def read_data(d):
    names = ("figS3_inputs.csv", "figS3b_combined.csv", "figS3c_spotcheck.csv")
    A, B, S = [pd.read_csv(d / n, encoding="utf-8-sig") for n in names]
    S["n"] = S.n.astype(str)
    S.loc[~S.n.str.isdigit(), "n"] = "all"
    for f, c in ((A, ["lo", "auc", "hi"]), (S, ["lo", "auc", "hi"])):
        v = f[c].to_numpy(float); assert np.isfinite(v).all() and (np.diff(v, axis=1) >= 0).all()
    g = B[B.row == "gain"]; assert len(g) == 8 and (g.lo <= g.value).all() and (g.value <= g.hi).all()
    assert len(B[B.row == "auc"]) == 12 and len(S) == 30 and set(S.n) == set(WINDOWS)
    for c in COHORTS:
        for s, _ in SETS_A:
            m = "Deep" if s.startswith("Deep") else "Logistic regression"
            assert len(A[(A.cohort == c) & (A.set == s) & (A.model == m)]) == 1, (c, s)
        a = A[(A.cohort == c) & (A.model == "Deep")].auc.iloc[0]
        b = B[(B.cohort == c) & (B.kind == "int") & (B.model == "Deep model")].value.iloc[0]
        s = S[(S.cohort == c) & (S.model == "Deep model") & (S.n == "all")].auc.iloc[0]
        assert abs(a - b) < 1e-9 and abs(a - s) < 5e-5
    return A, B, S, names


def framed(ax):
    for s in ax.spines.values():
        s.set_visible(True)


def plain(x):
    return f"{x:.3f}".replace("-", "−")


def bar(ax, x, h, w, color, label=None, lo=None, hi=None, base=0.5):
    ax.bar(x, h - base, w, bottom=base, color=color, edgecolor="white", linewidth=0.6, zorder=2)
    top = h
    if lo is not None and np.isfinite(lo):
        ax.plot([x, x], [lo, hi], color=INK, lw=0.6, zorder=3); top = hi
    if label:
        ax.text(x, top + 0.006, label, ha="center", va="bottom", fontsize=FS - 1.4, color=INK)


def draw(data_dir, out_dir):
    font = font_in_use()
    if font not in ("Arial", "Liberation Sans"):
        print(f"[warning] Arial not found; using {font}. Text is wider than intended.")
    A, B, S, names = read_data(data_dir)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    T = lambda mm: 1 - mm / HEIGHT_MM
    H = lambda mm: mm / HEIGHT_MM
    L = lambda s, x, y: fig.text(x, y, s, fontsize=9, fontweight="bold", va="top")
    TOP, ROW_H = 9.0, 40.0
    X0, GAP = 0.075, 0.085
    AX_W = (0.985 - X0 - GAP) / 2
    X1 = X0 + AX_W + GAP

    L("a", 0.012, T(1.5))
    ax = fig.add_axes([X0, T(TOP + ROW_H), AX_W, H(ROW_H)]); framed(ax)
    w = 0.36
    for k, (s_, lab) in enumerate(SETS_A):
        deep = s_.startswith("Deep")
        for c, off in (("MIMIC", -0.19), ("MC-MED", 0.19)):
            r = A[(A.cohort == c) & (A.set == s_) & (A.model == ("Deep" if deep else "Logistic regression"))].iloc[0]
            bar(ax, k + off, r.auc, w, COL[c] if deep else shade(COL[c], -0.55), plain(r.auc), r.lo, r.hi)
    ax.axvline(len(SETS_A) - 1.5, color="#d0d0d0", lw=0.6, zorder=0)
    ax.set(xlim=(-0.55, len(SETS_A) - 0.45), ylim=(0.5, 0.86), yticks=[0.5, 0.6, 0.7, 0.8])
    ax.set_xticks(range(len(SETS_A))); ax.set_xticklabels([lab for _, lab in SETS_A], fontsize=FS - 0.4)
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_ylabel("AUC (95% CI)", labelpad=2)
    ax.legend(handles=[Patch(fc=COL[c], ec="none", label=NAME[c]) for c in COHORTS], loc="upper left", ncol=2,
              fontsize=FS - 0.6, handlelength=1.1, handleheight=0.8, handletextpad=0.35, columnspacing=1.2,
              borderaxespad=0.4)
    ax.text(0.02, 0.84, "Light, handcrafted features; solid, deep model", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS - 0.7, color=MUTED)

    L("b", X1 - 0.06, T(1.5))
    ax = fig.add_axes([X1, T(TOP + ROW_H), AX_W, H(ROW_H)]); framed(ax)
    GROUPS = [("MIMIC", "int"), ("MC-MED", "int"), ("MIMIC", "ext"), ("MC-MED", "ext")]
    MODELS = [("Handcrafted", -0.55), ("Deep model", 0.0), ("Deep + handcrafted", 0.38)]
    w = 0.27
    ax.add_patch(Rectangle((1.5, 0.5), 2.0, 0.36, fc=BAND, ec="none", zorder=0))
    for g, (c, k) in enumerate(GROUPS):
        for m_i, (m, t) in enumerate(MODELS):
            r = B[(B.cohort == c) & (B.kind == k) & (B.row == "auc") & (B.model == m)].iloc[0]
            bar(ax, g + (m_i - 1) * (w + 0.02), r.value, w, shade(COL[c], t), plain(r.value))
    ax.text(0.5, 0.845, "Internal", ha="center", va="top", fontsize=FS - 0.4, color=MUTED)
    ax.text(2.5, 0.845, "Frozen model, other system", ha="center", va="top", fontsize=FS - 0.4, color=MUTED)
    ax.set(xlim=(-0.5, 3.5), ylim=(0.5, 0.86), yticks=[0.5, 0.6, 0.7, 0.8])
    ax.set_xticks(range(4)); ax.set_xticklabels([NAME[c] for c, _ in GROUPS]); ax.tick_params(axis="x", length=0, pad=2.5)
    for lbl, c in zip(ax.get_xticklabels(), [c for c, _ in GROUPS]):
        lbl.set_color(COL[c])
    ax.set_ylabel("AUC", labelpad=2)
    ax.legend(handles=[Patch(fc=shade("#6a6a6a", t), ec="none", label=m) for m, t in MODELS],
              loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, fontsize=FS - 0.6, handlelength=1.1,
              handleheight=0.8, handletextpad=0.35, columnspacing=1.2, borderaxespad=0)

    C_TOP = 65.0
    L("c", 0.012, T(C_TOP - 4.0))
    SER = [("Deep model", "Deep model", None), ("Handcrafted (patient level)", "Handcrafted, patient level", "#6f6f6f"),
           ("Handcrafted (window level)", "Handcrafted, window level", HAND2)]
    order = ["1", "2", "4", "10", "all"]
    for j, c in enumerate(COHORTS):
        ax = fig.add_axes([(X0, X1)[j], T(C_TOP + 32), AX_W, H(32)]); framed(ax)
        for m, lab, col in SER:
            col = col or COL[c]
            d = S[(S.cohort == c) & (S.model == m)].set_index("n").loc[order]
            x = np.arange(5)
            ax.fill_between(x, d.lo, d.hi, color=col, alpha=0.13, lw=0, zorder=1)
            ax.plot(x, d.auc, color=col, lw=1.0, marker=MK[c], ms=MS[c], zorder=3, label=lab)
        ax.set(xlim=(-0.25, 4.25), ylim=(0.56, 0.83), yticks=[0.6, 0.7, 0.8])
        ax.set_xticks(range(5)); ax.set_xticklabels([WINDOWS[k] for k in order]); ax.tick_params(axis="x", length=0, pad=2.5)
        ax.set_xlabel("Signal per patient (first 30-s windows)", labelpad=2)
        ax.set_ylabel("AUC (95% CI)", labelpad=2)
        ax.set_title(NAME[c] + ", internal", fontsize=FS + 0.2, fontweight="bold", color=COL[c], loc="left", pad=3)
        ax.legend(loc="upper left", fontsize=FS - 0.6, handlelength=1.6, borderaxespad=0.35, labelspacing=0.25)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "fig6_handcrafted"
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out_dir / "fig6_handcrafted_preview.png", dpi=300, facecolor="white")
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
