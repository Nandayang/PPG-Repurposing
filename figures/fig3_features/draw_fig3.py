"""Fig. 3 - Pulse-contour and beat-to-beat features in both health systems: a Cliff's delta (cases vs controls) for 11
pulse-contour descriptors, pulse rate and pulse-rate variability (ns: BH-adjusted P >= 0.05); b concordance of the
effect sizes between systems.

Input: data/figS2a_effects.csv (from 29_morph_effects.py)
Usage: python draw_fig3.py [--data-dir DIR] [--out-dir DIR]
"""
from pathlib import Path
import argparse, hashlib, json, platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
COL = {"MIMIC": "#004E85", "MC-MED": "#E96A6A"}
NAME = {"MIMIC": "ICU (MIMIC)", "MC-MED": "ED (MC-MED)"}
INK, MUTED, FRAME, REF = "#1a1a1a", "#666666", "#4d4d4d", "#8c8c8c"
GREY = "#4a4a4a"
COHORTS = ("MIMIC", "MC-MED")
CMAP = LinearSegmentedColormap.from_list("div", [COL["MIMIC"], "#f1f1f1", COL["MC-MED"]])
VMAX = 0.40
DOMAINS = ["Morphology", "Rate", "Variability"]
DOMAIN_LABEL = {"Morphology": "Pulse contour", "Rate": "Pulse rate", "Variability": "Beat-to-beat variability"}
RENAME = {"Heart rate": "Pulse rate", "IBI CV": "Interval CV"}
FS = 6.4
WIDTH_MM, HEIGHT_MM = 183, 87
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


def lum(c):
    v = np.asarray(to_rgb(c)); v = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return float(v @ [0.2126, 0.7152, 0.0722])


def read_data(d):
    name = "figS2a_effects.csv"
    E = pd.read_csv(d / name, encoding="utf-8-sig")
    assert len(E) == 40 and set(E.cohort) == set(COHORTS) and set(E.domain) == set(DOMAINS)
    assert (E.groupby("feature").cohort.nunique() == 2).all()
    v = E[["lo", "delta", "hi"]].to_numpy(float); assert np.isfinite(v).all() and (np.diff(v, axis=1) >= 0).all()
    E["label"] = E.label.replace(RENAME)
    W = E.pivot(index=["feature", "label", "domain"], columns="cohort", values=["delta", "q"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index()
    W["dorder"] = W.domain.map({k: i for i, k in enumerate(DOMAINS)})
    W = W.sort_values(["dorder", "delta_MIMIC"], ascending=[True, False]).reset_index(drop=True)
    return W, [name]


def draw(data_dir, out_dir):
    font = font_in_use()
    if font not in ("Arial", "Liberation Sans"):
        print(f"[warning] Arial not found; using {font}. Text is wider than intended.")
    W, names = read_data(data_dir)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    T = lambda mm: 1 - mm / HEIGHT_MM
    H = lambda mm: mm / HEIGHT_MM
    L = lambda s, x, y: fig.text(x, y, s, fontsize=9, fontweight="bold", va="top")
    norm = Normalize(-VMAX, VMAX)

    L("a", 0.012, T(1.5))
    GAP = 0.45
    y, ys, prev = 0.0, [], None
    for dom in W.domain:
        if prev is not None and dom != prev:
            y += GAP
        ys.append(y); y += 1; prev = dom
    n_rows = y
    top_mm, row_mm = 10.0, 3.2
    ax = fig.add_axes([0.215, T(top_mm + n_rows * row_mm), 0.25, H(n_rows * row_mm)])
    for i, (_, r) in enumerate(W.iterrows()):
        for j, c in enumerate(COHORTS):
            d, q = r[f"delta_{c}"], r[f"q_{c}"]
            fc = CMAP(norm(d))
            ax.add_patch(plt.Rectangle((j + 0.015, ys[i] + 0.04), 0.97, 0.92, fc=fc, ec="none"))
            tc = "white" if lum(fc) < 0.30 else (INK if q < 0.05 else MUTED)
            ax.text(j + 0.5, ys[i] + 0.5, f"{d:+.2f}".replace("-", "−") + ("  ns" if q >= 0.05 else ""),
                    ha="center", va="center", fontsize=FS - 0.3, color=tc)
        ax.text(-0.06, ys[i] + 0.5, r.label, ha="right", va="center", fontsize=FS)
    ax.set_xlim(0, 2); ax.set_ylim(n_rows, 0); ax.axis("off")
    for j, c in enumerate(COHORTS):
        ax.text(j + 0.5, -0.55, NAME[c], ha="center", va="bottom", fontsize=FS + 0.2, fontweight="bold", color=INK)
    for dom in DOMAINS:
        idx = [ys[i] for i in range(len(W)) if W.domain.iloc[i] == dom]
        if len(idx) > 1:
            ax.text(-1.38, (min(idx) + max(idx) + 1) / 2, DOMAIN_LABEL[dom], rotation=90, ha="center", va="center",
                    fontsize=FS - 0.2, color=MUTED)
            ax.plot([-1.3, -1.3], [min(idx) + 0.1, max(idx) + 0.9], color="#bdbdbd", lw=0.7, clip_on=False)
    cax = fig.add_axes([0.485, T(top_mm + n_rows * row_mm * 0.72), 0.010, H(n_rows * row_mm * 0.44)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, ticks=[-0.4, -0.2, 0, 0.2, 0.4])
    cb.outline.set_visible(False)
    cb.ax.set_yticklabels(["−0.4", "−0.2", "0", "0.2", "0.4"])
    cb.ax.tick_params(labelsize=FS - 0.6, length=1.8, width=0.5, pad=1.5)
    cb.set_label("Cliff's δ (cases − controls)", fontsize=FS - 0.2, labelpad=3)

    L("b", 0.610, T(1.5))
    side_mm = 56.0
    ax = fig.add_axes([0.665, T(top_mm + side_mm), side_mm / WIDTH_MM, H(side_mm)])
    for s in ax.spines.values():
        s.set_visible(True)
    lim = (-0.42, 0.42)
    ax.plot(lim, lim, color=REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color="#d0d0d0", lw=0.6, zorder=0); ax.axvline(0, color="#d0d0d0", lw=0.6, zorder=0)
    sty = {"Morphology": dict(marker="o", mfc=GREY), "Rate": dict(marker="^", mfc=GREY),
           "Variability": dict(marker="o", mfc="white")}
    for dom in DOMAINS:
        d = W[W.domain == dom]
        ax.plot(d["delta_MIMIC"], d["delta_MC-MED"], ls="", ms=3.4, color=GREY, mec=GREY, mew=0.8, zorder=3,
                label=DOMAIN_LABEL[dom], **sty[dom])
    P = W.set_index("label")
    ann = {"Inflection point area": ("Inflection point area,\nnotch position", (-4, 7), "right"),
           "Reflection index": ("Reflection index", (5, -2), "left"),
           "SD2": ("SD2", (5, -3), "left"), "SDNN": ("SDNN", (5, 0), "left")}
    for lab, (t, xy, ha) in ann.items():
        ax.annotate(t, (P.loc[lab, "delta_MIMIC"], P.loc[lab, "delta_MC-MED"]), xytext=xy, textcoords="offset points",
                    fontsize=FS - 0.6, color=MUTED, ha=ha, va="center", linespacing=1.2)
    rho_all = spearmanr(W["delta_MIMIC"], W["delta_MC-MED"])
    m = W.domain == "Morphology"
    rho_m = spearmanr(W.loc[m, "delta_MIMIC"], W.loc[m, "delta_MC-MED"])
    ax.text(0.04, 0.96, f"Spearman ρ\nPulse contour ({int(m.sum())}): {rho_m.statistic:.2f}\n"
            f"All {len(W)} features: {rho_all.statistic:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS - 0.3, linespacing=1.35)
    tl = ["−0.4", "−0.2", "0", "0.2", "0.4"]
    ax.set(xlim=lim, ylim=lim, xticks=[-0.4, -0.2, 0, 0.2, 0.4], yticks=[-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xticklabels(tl); ax.set_yticklabels(tl)
    ax.set_xlabel("Cliff's δ, ICU (MIMIC)", labelpad=2)
    ax.set_ylabel("Cliff's δ, ED (MC-MED)", labelpad=1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=FS - 0.4, handletextpad=0.2,
              columnspacing=1.0, borderaxespad=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "fig3_features"
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out_dir / "fig3_features_preview.png", dpi=300, facecolor="white")
    plt.close(fig)
    meta = {"canvas_mm": [WIDTH_MM, HEIGHT_MM], "font": font,
            "source_hashes": {n: hashlib.sha256((data_dir / n).read_bytes()).hexdigest() for n in names},
            "features": int(len(W)), "spearman_all": [rho_all.statistic, rho_all.pvalue],
            "spearman_pulse_contour": [rho_m.statistic, rho_m.pvalue],
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "matplotlib": matplotlib.__version__}}
    prefix.with_suffix(".data-check.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_dir} (rho all {rho_all.statistic:.2f}, pulse contour {rho_m.statistic:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT.parent / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out")
    a = ap.parse_args()
    draw(a.data_dir, a.out_dir)
