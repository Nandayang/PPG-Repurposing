"""Fig. 5 - PPG scores across the spectrum of liver disease: a by MELD category within cirrhosis (internal models);
b, c ICU controls, non-cirrhotic chronic liver disease without and with portal-hypertension or decompensation codes,
compensated and decompensated cirrhosis (b internal model; c ED-trained model applied frozen, converted to
percentiles of the cirrhosis cohort).

Inputs (data/): fig_spectrum_summary.csv, fig_spectrum_points.csv (33_highrisk.py),
fig_spectrum_meld_summary.csv, fig_spectrum_meld_points.csv (make_meld_panel_data.py)
Usage: python draw_fig5.py [--data-dir DIR] [--out-dir DIR]
"""
from pathlib import Path
import argparse, hashlib, json, platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
COL = {"MIMIC": "#004E85", "MC-MED": "#E96A6A"}
MARKER = {"MIMIC": "o", "MC-MED": "o"}
INK, MUTED, FRAME, REF = "#1a1a1a", "#666666", "#4d4d4d", "#8c8c8c"
DASH = (0, (3, 1.6))
FS = 6.4
WIDTH_MM, HEIGHT_MM = 183, 84
FONTS = ["Arial", "Liberation Sans", "DejaVu Sans"]
COLS = ["#8a8a8a", "#77b0e7", "#568ec3", "#356da0", "#004E85"]
LABEL = ["Controls", "CLD, no PH", "CLD, PH", "Compensated", "Decompensated"]
PANELS = [("b", "internal", "ICU, internal model"), ("c", "frozen", "ICU, ED-trained model, frozen")]
MAX_DOTS = 300
YLIM, YTICKS = (-0.03, 1.47), [0, 0.25, 0.5, 0.75, 1.0]
BR0, BR_STEP, BR_TICK = 1.06, 0.095, 0.025
C_GAP = 0.9
X0, CW, AW, GAP_AB, GAP_BC, TOP, AH = 13.0, 60.0, 44.5, 11.0, 8.0, 8.0, 56.0

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONTS,
    "font.size": FS, "axes.labelsize": FS + 0.6, "xtick.labelsize": FS, "ytick.labelsize": FS,
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
    names = ["fig_spectrum_summary.csv", "fig_spectrum_points.csv", "fig_spectrum_meld_summary.csv",
             "fig_spectrum_meld_points.csv"]
    miss = [n for n in names if not (d / n).exists()]
    if miss:
        raise SystemExit(f"[!] missing in {d}: {', '.join(miss)}")
    S, P, MS, MP = (pd.read_csv(d / n, encoding="utf-8-sig") for n in names)
    assert len(S) == 10 and set(S.model) == {"internal", "frozen"} and sorted(S.order.unique()) == list(range(5))
    assert S[S.order > 0].auc_crude.between(0, 1).all() and S[S.order == 0].auc_crude.isna().all()
    assert P.score.between(0, 1).all() and MP.score.between(0, 1).all()
    for r in S.itertuples():
        assert ((P.model == r.model) & (P.order == r.order)).sum() == r.n, (r.model, r.group)
    assert len(MS) == 8 and set(MS.cohort) == {"MIMIC", "MC-MED"}
    for r in MS.itertuples():
        assert ((MP.cohort == r.cohort) & (MP.order == r.order)).sum() == r.n, (r.cohort, r.category)
    return S, P, MS, MP, names


def lighten(c, f):
    c = np.array(matplotlib.colors.to_rgb(c)); return tuple(1 - (1 - c) * f)


def dot_alpha(c):
    r, g, b = (x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in matplotlib.colors.to_rgb(c))
    return 0.30 + 0.4 * (0.2126 * r + 0.7152 * g + 0.0722 * b)


def draw_box(ax, x, v, col, rng, frozen=False, marker="o", W=0.27):
    q05, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
    d = v if len(v) <= MAX_DOTS else rng.choice(v, MAX_DOTS, replace=False)
    ax.scatter(x + rng.uniform(-1, 1, len(d)) * (W + 0.03), d, s=3.0 if marker == "o" else 2.6, marker=marker,
               color=col, alpha=dot_alpha(col), lw=0, zorder=3)
    ax.add_patch(Rectangle((x - W, q25), 2 * W, q75 - q25, fc=lighten(col, 0.18), ec=col, lw=0.9,
                           ls=DASH if frozen else "-", zorder=2))
    for a, b in ((q05, q25), (q75, q95)):
        ax.plot([x, x], [a, b], color=col, lw=0.9, zorder=4)
    for yy in (q05, q95):
        ax.plot([x - 0.09, x + 0.09], [yy, yy], color=col, lw=0.9, zorder=4)
    ax.plot([x - W, x + W], [q50, q50], color=col, lw=2.4, zorder=5, solid_capstyle="butt")
    return q50


def bracket(ax, x1, x2, yb, text):
    ax.plot([x1, x1, x2, x2], [yb - BR_TICK, yb, yb, yb - BR_TICK], color=INK, lw=0.6, zorder=6, clip_on=False)
    ax.text((x1 + x2) / 2, yb + BR_TICK * 0.35, text, ha="center", va="bottom", fontsize=FS - 0.6, zorder=6)


def frame(ax, ylabel=None):
    ax.set(ylim=YLIM, yticks=YTICKS); ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.tick_params(axis="x", length=2.5, pad=2)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=3)
    for side in ax.spines.values():
        side.set_visible(True)


def ptxt(p):
    return "P < 0.001" if p < 0.001 else f"P = {p:.3f}"


def draw(data_dir, out_dir):
    font = font_in_use()
    if font not in ("Arial", "Liberation Sans"):
        print(f"[warning] Arial not found; using {font}. Text is wider than intended.")
    S, P, MS, MP, names = read_data(data_dir)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    T = lambda mm: 1 - mm / HEIGHT_MM
    H = lambda mm: mm / HEIGHT_MM

    for j, (letter, model, title) in enumerate(PANELS):
        rng = np.random.default_rng(0)
        x0 = X0 + CW + GAP_AB + j * (AW + GAP_BC)
        ax = fig.add_axes([x0 / WIDTH_MM, T(TOP + AH), AW / WIDTH_MM, H(AH)])
        fig.text((x0 - 9.5) / WIDTH_MM, T(1.0), letter, fontsize=9, fontweight="bold", va="top")
        ax.set_title(title, fontsize=FS + 0.6, pad=4)
        sm = S[S.model == model].sort_values("order"); meds = []
        for k, s in enumerate(sm.itertuples()):
            v = P[(P.model == model) & (P.order == k)].score.to_numpy(float)
            meds.append(draw_box(ax, k, v, COLS[k], rng, frozen=(model == "frozen")))
            if k > 0:
                bracket(ax, 0, k, BR0 + (k - 1) * BR_STEP, f"AUC = {s.auc_crude:.3f}")
        ax.axhline(meds[0], color="#c8c8c8", lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(-0.6, 4.6); ax.set_xticks(range(5))
        ax.set_xticklabels([f"{lab} ({int(n):,})" for lab, n in zip(LABEL, sm.n)], rotation=40, ha="right",
                           rotation_mode="anchor", fontsize=FS)
        frame(ax)

    rng = np.random.default_rng(0)
    x0 = X0
    ax = fig.add_axes([x0 / WIDTH_MM, T(TOP + AH), CW / WIDTH_MM, H(AH)])
    fig.text((x0 - 11.5) / WIDTH_MM, T(1.0), "a", fontsize=9, fontweight="bold", va="top")
    ax.set_title("Cirrhosis by MELD score, internal models", fontsize=FS + 0.6, pad=4)
    ticks, labels = [], []
    for g, (coh, lab) in enumerate((("MIMIC", "ICU (MIMIC)"), ("MC-MED", "ED (MC-MED)"))):
        ms = MS[MS.cohort == coh].sort_values("order"); off = g * (4 + C_GAP)
        for r in ms.itertuples():
            v = MP[(MP.cohort == coh) & (MP.order == r.order)].score.to_numpy(float)
            draw_box(ax, off + r.order, v, COL[coh], rng, marker=MARKER[coh], W=0.29)
            ticks.append(off + r.order); labels.append(f"{r.category} ({r.n})")
        c50 = ms.control_median.iloc[0]
        ax.plot([off - 0.5, off + 3.5], [c50, c50], color="#c8c8c8", lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(off + 1.5, 1.31, lab, ha="center", va="bottom", fontsize=FS, color=COL[coh], fontweight="bold")
        ax.text(off + 1.5, 1.20, f"ρ = {ms.spearman_rho.iloc[0]:.2f}, {ptxt(ms.spearman_p.iloc[0])}", ha="center",
                va="bottom", fontsize=FS - 0.6)
    ax.axvline(3.5 + C_GAP / 2, color="#d9d9d9", lw=0.6, zorder=1)
    ax.set_xlim(-0.6, 7.6 + C_GAP); ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=40, ha="right", rotation_mode="anchor", fontsize=FS)
    frame(ax, "PPG score (percentile)")

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "fig5_liver_spectrum"
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out_dir / "fig5_liver_spectrum_preview.png", dpi=300, facecolor="white")
    plt.close(fig)
    meta = {"canvas_mm": [WIDTH_MM, HEIGHT_MM], "font": font, "max_dots_per_group": MAX_DOTS,
            "auc_vs_controls": {m: dict(zip(S[(S.model == m) & (S.order > 0)].sort_values("order").group,
                                            S[(S.model == m) & (S.order > 0)].sort_values("order").auc_crude.round(3)))
                                for m in ("internal", "frozen")},
            "meld_n": {c: MS[MS.cohort == c].sort_values("order").n.tolist() for c in ("MIMIC", "MC-MED")},
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
