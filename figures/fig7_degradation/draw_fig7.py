"""Fig. 7 - Discrimination under emulated consumer-oximeter degradation (conditions C0-C5, first ten windows per
patient): a AUC with 95% CIs and fold- or model-level AUCs; b C0, C4 and C5; c change in AUC from C0 with paired CIs;
d AUC table shaded by the change from C0.

Inputs: data/fig4c_hardware.csv (make_plot_data.py); optional fold_auc_22b.csv and delta_22b.csv (22b_spot_folds.py)
Usage: python draw_fig7.py [--data-dir DIR] [--out-dir DIR]
"""
from pathlib import Path
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
SPOT = Path(os.environ.get("PPG_ROOT", ".")) / "results" / "spot"
COL = {"MIMIC": "#004E85", "MC-MED": "#E96A6A"}
MK = {"MIMIC": "o", "MC-MED": "s"}
NAME = {"MIMIC": "ICU", "MC-MED": "ED"}
LONG = {"MIMIC": "ICU (MIMIC)", "MC-MED": "ED (MC-MED)"}
KIND = {"int": "internal", "ext": "external"}
INK, MUTED, AXIS, GRID = "#1a1a1a", "#6b6b6b", "#8a8a8a", "#ebebeb"
DASH = (0, (3, 1.6))
FS = 6.4
STEP = ["C0\noriginal", "C1\n60 Hz", "C2\n5.5-bit", "C3\nclipping", "C4\ngain", "C5\nstress"]
SETTINGS = [("MIMIC", "int"), ("MIMIC", "ext"), ("MC-MED", "int"), ("MC-MED", "ext")]
PANEL_MM = (88, 62)
SHEET_MM = (183, 122)
FONTS = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONTS,
    "font.size": FS, "axes.labelsize": FS + 0.2, "xtick.labelsize": FS - 0.6, "ytick.labelsize": FS - 0.4,
    "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "xtick.color": "#444444", "ytick.color": "#444444",
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none", "text.hinting": "no_hinting"})


def tint(c, t):
    v = np.array(to_rgb(c)); return tuple(v + (1 - v) * t)


def hgrid(ax, ys):
    ax.set_yticks(ys)
    for y in ys:
        ax.axhline(y, color=GRID, lw=0.5, zorder=0)


def font_in_use():
    from matplotlib import font_manager as fm
    for f in FONTS:
        try:
            fm.findfont(fm.FontProperties(family=f), fallback_to_default=False); return f
        except ValueError:
            continue
    return "matplotlib default"


def save(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def optional(data_dir, name):
    for base in (data_dir, SPOT):
        if (base / name).exists():
            return pd.read_csv(base / name, encoding="utf-8-sig")
    return None


def load(data_dir):
    hw = pd.read_csv(data_dir / "fig4c_hardware.csv", encoding="utf-8-sig")
    assert len(hw) == 24
    v = hw[["lo", "auc", "hi"]].to_numpy(float); assert np.isfinite(v).all() and (np.diff(v, axis=1) >= 0).all()
    get = lambda c, k: hw[(hw.test == c) & (hw.kind == k)].sort_values("step").reset_index(drop=True)
    D = {s: get(*s) for s in SETTINGS}
    for s in SETTINGS:
        assert list(D[s].step) == list(range(6))
    folds, delta = optional(data_dir, "fold_auc_22b.csv"), optional(data_dir, "delta_22b.csv")
    if folds is not None:
        assert len(folds) == 120 and set(folds.groupby(["test", "kind", "step"]).size()) == {5}
    if delta is not None:
        assert len(delta) == 20
    return D, folds, delta


def legend_settings(fig, y=0.005):
    h = [Line2D([0], [0], marker=MK[c], ls="-" if k == "int" else DASH, lw=0.9, ms=3, color=COL[c],
                mfc=COL[c] if k == "int" else "white", label=f"{NAME[c]} {KIND[k]}") for c, k in SETTINGS]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=4, fontsize=FS - 0.8,
               handlelength=2.0, columnspacing=1.2, handletextpad=0.4, borderaxespad=0)


def panel_a(fig, D, folds=None):
    xs = np.array([0, 1, 2, 3, 4, 5.4]); bw = 0.36
    jit = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
    top = 0.80 if folds is None else max(0.80, float(np.ceil((folds.auc.max() + 0.01) * 20) / 20))
    bottom = 0.5 if folds is None else min(0.5, float(np.floor((folds.auc.min() - 0.01) * 20) / 20))
    for j, c in enumerate(("MIMIC", "MC-MED")):
        ax = fig.add_axes([0.085 + j * 0.465, 0.17, 0.415, 0.75])
        hgrid(ax, [t for t in (0.5, 0.6, 0.7, 0.8, 0.9) if bottom - 1e-9 <= t <= top + 1e-9])
        ax.set_ylim(bottom, top)
        for k, off in (("int", -0.2), ("ext", 0.2)):
            d = D[(c, k)]
            ax.bar(xs + off, d.auc - 0.5, bw, bottom=0.5, color=tint(COL[c], 0.55 if k == "int" else 0.82),
                   edgecolor="none", zorder=2)
            ax.errorbar(xs + off, d.auc, yerr=np.vstack([d.auc - d.lo, d.hi - d.auc]), fmt="none", ecolor=INK,
                        elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=4)
            if folds is not None:
                for i, st in enumerate(range(6)):
                    v = folds[(folds.test == c) & (folds.kind == k) & (folds.step == st)].sort_values("unit").auc
                    ax.scatter(xs[i] + off + jit, v, s=6, marker="o", zorder=5, linewidths=0.5,
                               facecolors=COL[c] if k == "int" else "white", edgecolors=COL[c])
        ax.set_xticks(xs); ax.set_xticklabels([s.split("\n")[0] for s in STEP]); ax.set_xlim(-0.6, 6.0)
        ax.set_title(LONG[c], fontsize=FS + 0.2, fontweight="bold", loc="left", pad=3)
        if j == 0:
            ax.set_ylabel("AUC (95% CI)", labelpad=2)
        else:
            ax.tick_params(axis="y", labelleft=False)
    h = [Patch(fc=tint("#555555", 0.55), label="Internal"), Patch(fc=tint("#555555", 0.82), label="External")]
    if folds is not None:
        h.append(Line2D([0], [0], marker="o", ls="", ms=2.6, color="#555555",
                        label="Fold (internal) / model (external)"))
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=len(h), fontsize=FS - 0.6,
               handlelength=1.2, handleheight=0.8, columnspacing=1.6)


def spread(vals, gap):
    order = np.argsort(vals); pos = np.array(vals, float)[order].copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos += np.array(vals, float)[order].mean() - pos.mean()
    out = np.empty_like(pos); out[order] = pos; return out


def panel_b(fig, D):
    ax = fig.add_axes([0.15, 0.15, 0.63, 0.83])
    hgrid(ax, [0.6, 0.7]); ax.set_ylim(0.545, 0.765)
    idx, xs = [0, 4, 5], [0, 1, 2]
    ends = {0: [], 2: []}
    for c, k in SETTINGS:
        d = D[(c, k)].iloc[idx]
        ax.plot(xs, d.auc, color=COL[c], lw=1.0, ls="-" if k == "int" else DASH, zorder=2)
        ax.errorbar(xs, d.auc, yerr=np.vstack([d.auc - d.lo, d.hi - d.auc]), fmt="none", ecolor=COL[c],
                    elinewidth=0.6, alpha=0.55, capsize=0, zorder=1)
        ax.plot(xs, d.auc, ls="", marker=MK[c], ms=3.2, color=COL[c], mec=COL[c], mew=0.8,
                mfc=COL[c] if k == "int" else "white", zorder=3)
        ends[0].append(d.auc.iloc[0]); ends[2].append(d.auc.iloc[2])
    for x, ha, dx in ((0, "right", -0.09), (2, "left", 0.09)):
        for (c, k), v, y in zip(SETTINGS, ends[x], spread(ends[x], 0.0115)):
            lab = f"{v:.3f}" if x == 0 else f"{v:.3f}  {NAME[c]} {KIND[k]}"
            ax.text(x + dx, y, lab, ha=ha, va="center", fontsize=FS - 0.9, color=INK if k == "int" else MUTED)
    ax.set_xticks(xs); ax.set_xticklabels(["Original\n(C0)", "Consumer\noximeter (C4)", "Stress\ntest (C5)"])
    ax.set_xlim(-0.35, 2.15); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", labelleft=False)
    fig.text(0.03, 0.565, "AUC (95% CI)", rotation=90, ha="center", va="center", fontsize=FS + 0.2)


def panel_c(fig, D, delta_ci=None):
    ax = fig.add_axes([0.15, 0.22, 0.83, 0.76])
    lo_min = -0.125 if delta_ci is None else min(-0.125, float(delta_ci.lo.min()) - 0.008)
    hi_max = 0.012 if delta_ci is None else max(0.012, float(delta_ci.hi.max()) + 0.004)
    hgrid(ax, [-0.10, -0.05, 0]); ax.set_ylim(lo_min, hi_max)
    ax.axhline(0, color="#9a9a9a", lw=0.6, zorder=1)
    xs = np.array([1, 2, 3, 4, 5.4])
    offs = dict(zip(SETTINGS, (-0.27, -0.09, 0.09, 0.27)))
    for c, k in SETTINGS:
        d = D[(c, k)]; delta = (d.auc - d.auc.iloc[0]).to_numpy()[1:]
        x = xs + offs[(c, k)]
        ax.vlines(x, 0, delta, color=COL[c], lw=0.9, linestyles="-" if k == "int" else (0, (2, 1.2)), zorder=2,
                  alpha=0.45 if delta_ci is not None else 1.0)
        if delta_ci is not None:
            q = delta_ci[(delta_ci.test == c) & (delta_ci.kind == k)].sort_values("step")
            assert np.allclose(q.delta.to_numpy(), delta, atol=0.0025)
            ax.vlines(x, q.lo, q.hi, color=COL[c], lw=0.8, zorder=2.5)
        ax.plot(x, delta, ls="", marker=MK[c], ms=3.2, color=COL[c], mec=COL[c], mew=0.8,
                mfc=COL[c] if k == "int" else "white", zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(STEP[1:], linespacing=1.1); ax.set_xlim(0.5, 5.9)
    ax.set_yticklabels(["−0.10", "−0.05", "0"])
    ax.set_ylabel("Change in AUC vs C0", labelpad=2)
    legend_settings(fig)


def panel_d(fig, D):
    ax = fig.add_axes([0.15, 0.005, 0.705, 0.975]); ax.axis("off")
    cmap = LinearSegmentedColormap.from_list("loss", ["#F6F7F8", "#BFD3E6", "#6E9CC4", "#2B5C8A"])
    norm = Normalize(0, 0.12)
    for j, (c, k) in enumerate(SETTINGS):
        d = D[(c, k)]
        for i in range(6):
            delta = d.auc.iloc[i] - d.auc.iloc[0]
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.04), 0.96, 0.92, fc=cmap(norm(max(0.0, -delta))), ec="none"))
            dark = norm(-delta) > 0.62
            ax.text(j + 0.5, i + 0.38, f"{d.auc.iloc[i]:.3f}", ha="center", va="center", fontsize=FS - 0.2,
                    color="white" if dark else INK, fontweight="bold" if i in (0, 4) else "normal")
            ax.text(j + 0.5, i + 0.70, f"{d.lo.iloc[i]:.3f}–{d.hi.iloc[i]:.3f}", ha="center", va="center",
                    fontsize=FS - 1.6, color="white" if dark else MUTED)
        ax.plot([j + 0.30], [-0.35], marker=MK[c], ms=3.2, color=COL[c], mfc=COL[c] if k == "int" else "white",
                mec=COL[c], mew=0.8, clip_on=False)
        ax.text(j + 0.40, -0.35, f"{NAME[c]}\n{KIND[k]}", ha="left", va="center", fontsize=FS - 0.8, linespacing=1.05)
    for i, s in enumerate(STEP):
        ax.text(-0.08, i + 0.5, s.replace("\n", "  "), ha="right", va="center", fontsize=FS - 0.6,
                fontweight="bold" if i == 4 else "normal")
    ax.set_xlim(0, 4); ax.set_ylim(6, -0.72)
    cax = fig.add_axes([0.875, 0.22, 0.014, 0.50])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[0, 0.05, 0.10])
    cb.outline.set_visible(False); cb.ax.set_yticklabels(["0", "−0.05", "−0.10"])
    cb.ax.tick_params(labelsize=FS - 1.2, length=1.5, width=0.4, pad=1)
    cb.set_label("Change in AUC vs C0", fontsize=FS - 1.0, labelpad=2)


def draw(data_dir, out_dir):
    font = font_in_use()
    if font not in ("Arial", "Liberation Sans"):
        print(f"[warning] Arial not found; using {font}. Text is wider than intended.")
    D, folds, delta = load(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [("a", lambda f: panel_a(f, D, folds)),
              ("b", lambda f: panel_b(f, D)),
              ("c", lambda f: panel_c(f, D, delta)),
              ("d", lambda f: panel_d(f, D))]

    for key, fn in panels:
        fig = plt.figure(figsize=(PANEL_MM[0] / 25.4, PANEL_MM[1] / 25.4))
        fn(fig); save(fig, out_dir / f"fig7{key}")

    sheet = plt.figure(figsize=(SHEET_MM[0] / 25.4, SHEET_MM[1] / 25.4))
    cells = sheet.subfigures(2, 2, wspace=0, hspace=0)
    for (key, fn), cell in zip(panels, cells.flat):
        fn(cell)
        cell.text(0.0, 1.0, key, fontsize=9, fontweight="bold", va="top", ha="left")
    save(sheet, out_dir / "fig7_degradation")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT.parent / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out")
    a = ap.parse_args()
    draw(a.data_dir, a.out_dir)
