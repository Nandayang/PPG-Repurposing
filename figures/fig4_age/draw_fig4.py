"""Fig. 4 - Age and the PPG signal: a AUC by age group; b odds ratio per SD of the score across age with the
interaction estimates; c Cliff's delta of four pulse-contour descriptors by age group; d mean phase-normalized beat,
cases minus controls, by age group.

Inputs (data/): fig6b_auc_by_age_45.csv, fig6a_or_by_age.csv, fig6a_interaction.csv, fig3c_feature_delta_by_age.csv,
fig3c_feature_age_interaction.csv, fig3d_beat_difference_by_age.csv
Usage: python draw_fig4.py [data_dir]
"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory as blend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = next(d for d in (sys.argv[1] if len(sys.argv) > 1 else "", os.path.join(HERE, "..", "data"),
                        "") if d and os.path.isdir(d))
rd = lambda f: pd.read_csv(os.path.join(DATA, f), encoding="utf-8-sig")
G = rd("fig6b_auc_by_age_45.csv")
O = rd("fig6a_or_by_age.csv")
I = rd("fig6a_interaction.csv")
assert (O[["odds_ratio", "lo", "hi"]].to_numpy(float) > 0).all(), "Log-scale odds ratios must be positive."
assert (I.p.to_numpy(float) > 0).all(), "Scientific-notation P values must be positive."
I["r"], I["rlo"], I["rhi"] = np.exp(I.beta_per_decade), np.exp(I.beta_per_decade - 1.96 * I.se), np.exp(I.beta_per_decade + 1.96 * I.se)
C = rd("fig3c_feature_delta_by_age.csv")
CI = rd("fig3c_feature_age_interaction.csv")
P = rd("fig3d_beat_difference_by_age.csv")

COL = {"MIMIC": "#004E85", "MC-MED": "#E96A6A"}
MK = {"MIMIC": "o", "MC-MED": "s"}
MS = {"MIMIC": 3.4, "MC-MED": 2.9}
NAME = {"MIMIC": "ICU (MIMIC)", "MC-MED": "ED (MC-MED)"}
INK, MUTED, FRAME, REF, BAND = "#1a1a1a", "#666666", "#4d4d4d", "#8c8c8c", "#f1f1f1"
GR = ["<45", "45\u201349", "50\u201359", "60\u201369", "\u226570"]
MM, FS = 1 / 25.4, 6.4
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS, "axes.labelsize": FS + 0.4, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "axes.linewidth": 0.7, "axes.edgecolor": FRAME, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.size": 2.5, "xtick.major.width": 0.6,
    "ytick.major.size": 2.5, "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "svg.fonttype": "none"})

WIDTH_MM = 183
H_MM = 179
LOWER_SHIFT_MM = 6
T = lambda mm: 1 - mm / H_MM
fig = plt.figure(figsize=(WIDTH_MM * MM, H_MM * MM))
def L(s, x, y): fig.text(x, y, s, fontsize=9, fontweight="bold", va="top")
ci = lambda a, lo, hi, d=3: f"{a:.{d}f} ({lo:.{d}f}\u2013{hi:.{d}f})"


def eb(ax, x, y, lo, hi, cohort, filled=True, horizontal=True, z=3):
    kw = dict(fmt=MK[cohort], ms=MS[cohort], color=COL[cohort], mfc=COL[cohort] if filled else "white",
              mec=COL[cohort], mew=0.8, elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=z)
    if horizontal: ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], **kw)
    else: ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], **kw)


def framed(ax):
    for s in ax.spines.values(): s.set_visible(True)


L("a", 0.012, T(1.5))
rows = []
for t in ("MIMIC", "MC-MED"):
    rows.append(dict(head=NAME[t], cohort=t))
    for gi, g in enumerate(GR):
        for k, kl in (("int", "Internal"), ("ext", "External")):
            r = G[(G.test == t) & (G.kind == k) & (G.age_group == g)].iloc[0]
            rows.append(dict(cohort=t, kind=k, g=g, gi=gi, val=kl, x=r.auc, lo=r.lo, hi=r.hi,
                             n=f"{int(r.n_pos):,} / {int(r.n_neg):,}", est=ci(r.auc, r.lo, r.hi)))
ax = fig.add_axes([0.335, T(89), 0.155, 77 / H_MM]); framed(ax); ax.patch.set_alpha(0)
n = len(rows); ax.set_ylim(n - 0.4, -0.9); ax.set_xlim(0.45, 0.95)
ax.xaxis.set_major_locator(FixedLocator([0.5, 0.6, 0.7, 0.8, 0.9])); ax.xaxis.set_minor_locator(NullLocator())
ax.set_yticks([])
ax.axvline(0.5, color=REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
tr = blend(fig.transFigure, ax.transData)
X_AGE, X_VAL, X_N, X_EST = 0.035, 0.125, 0.260, 0.505
for x, ha, h in ((X_AGE, "left", "Age (years)"), (X_VAL, "left", "Validation"), (X_N, "center", "Cases / controls"),
                 (X_EST, "left", "AUC (95% CI)")):
    fig.text(x, -1.6, h, transform=tr, ha=ha, va="center", fontsize=FS, fontweight="bold", fontstyle="italic")
for i, r in enumerate(rows):
    if "head" in r:
        fig.text(X_AGE, i, r["head"], transform=tr, ha="left", va="center", fontsize=FS + 0.2,
                 fontweight="bold", color=COL[r["cohort"]])
        continue
    if r["kind"] == "int":
        if r["gi"] % 2 == 1:
            fig.patches.append(Rectangle((0.03, i - 0.5), 0.6, 2, transform=tr, fc=BAND, ec="none", zorder=-5))
        fig.text(X_AGE + 0.012, i + 0.5, r["g"], transform=tr, ha="left", va="center", fontsize=FS)
        fig.text(X_N, i + 0.5, r["n"], transform=tr, ha="center", va="center", fontsize=FS)
    c = INK if r["kind"] == "int" else MUTED
    fig.text(X_VAL, i, r["val"], transform=tr, ha="left", va="center", fontsize=FS, color=c)
    fig.text(X_EST, i, r["est"], transform=tr, ha="left", va="center", fontsize=FS, color=c)
    eb(ax, r["x"], i, r["lo"], r["hi"], r["cohort"], filled=(r["kind"] == "int"))

L("b", 0.655, T(1.5))
ax = fig.add_axes([0.725, T(57), 0.26, 50 / H_MM]); framed(ax)
ax.set_yscale("log"); ax.set_xlim(35, 85); ax.set_ylim(0.85, 9)
ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8])); ax.yaxis.set_major_formatter(FixedFormatter(["1", "2", "4", "8"]))
ax.yaxis.set_minor_locator(NullLocator()); ax.set_xticks([35, 45, 55, 65, 75, 85])
ax.axvspan(35, 45, color=BAND, zorder=0, lw=0)
ax.text(40, 8.4, "<45", fontsize=FS - 0.4, color=MUTED, va="top", ha="center")
ax.axhline(1, color=REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
for t in ("MIMIC", "MC-MED"):
    for k in ("int", "ext"):
        d = O[(O.test == t) & (O.kind == k)].sort_values("age")
        if k == "int":
            ax.fill_between(d.age, d.lo, d.hi, color=COL[t], alpha=0.14, lw=0, zorder=2)
        ax.plot(d.age, d.odds_ratio, color=COL[t], lw=1.0, ls="-" if k == "int" else (0, (3, 1.6)), zorder=3)
ax.set_xlabel("Age (years)", labelpad=2); ax.set_ylabel("Odds ratio per SD of PPG score", labelpad=2)
y0, dy = T(70), 3.3 / H_MM
x1, x2, x3 = 0.675, 0.79, 0.985
for x, ha, h in ((x1, "left", "Setting"), (x2, "left", "Ratio per 10 years (95% CI)"), (x3, "right", "P")):
    fig.text(x, y0, h, fontsize=FS, fontweight="bold", fontstyle="italic", va="center", ha=ha)
fig.add_artist(Line2D([x1, x3], [y0 + 1.9 / H_MM] * 2, color=FRAME, lw=0.7))
fig.add_artist(Line2D([x1, x3], [y0 - 1.8 / H_MM] * 2, color=FRAME, lw=0.5))
for j, (t, k) in enumerate([("MIMIC", "int"), ("MIMIC", "ext"), ("MC-MED", "int"), ("MC-MED", "ext")]):
    r = I[(I.test == t) & (I.kind == k)].iloc[0]; y = y0 - (j + 1) * dy - 0.4 / H_MM
    e = int(np.floor(np.log10(r.p))); m = r.p / 10 ** e
    fig.text(x1, y, f"{NAME[t].split(' ')[0]}, {'internal' if k == 'int' else 'external'}", color=COL[t], va="center")
    fig.text(x2, y, ci(r.r, r.rlo, r.rhi, 2), va="center", color=INK if k == "int" else MUTED)
    fig.text(x3, y, f"{m:.1f}×10$^{{{e}}}$", fontsize=7.2, va="center", ha="right", color=INK if k == "int" else MUTED)
fig.add_artist(Line2D([x1, x3], [y0 - 4.5 * dy - 0.4 / H_MM] * 2, color=FRAME, lw=0.7))

L("c", 0.012, T(101 - LOWER_SHIFT_MM))
FEAT = [("notch_pos", "Notch position"), ("IPA", "Inflection point area"),
        ("dT", "Peak-to-peak time"), ("RI", "Reflection index")]
PW, PH = 0.205, 27 / H_MM
BOT = [T(135 - LOWER_SHIFT_MM), T(171 - LOWER_SHIFT_MM)]
qs = lambda v: "< 0.001" if v < 0.001 else f"= {v:.3f}"
for k, (f, nm) in enumerate(FEAT):
    ax = fig.add_axes([[0.07, 0.322][k % 2], BOT[k // 2], PW, PH]); framed(ax)
    ax.axhline(0, color=REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for t, off in (("MIMIC", -0.09), ("MC-MED", 0.09)):
        d = C[(C.cohort == t) & (C.feature == f)].sort_values("order")
        x = d.order.to_numpy(float) + off
        ax.plot(x, d.delta, color=COL[t], lw=0.9, zorder=2)
        ax.errorbar(x, d.delta, yerr=np.vstack([d.delta - d.lo, d.hi - d.delta]), fmt=MK[t], ms=MS[t],
                    color=COL[t], mfc=COL[t], mec=COL[t], mew=0.8, elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=3)
    ax.set_xlim(-0.45, 4.45); ax.set_xticks(range(5))
    ax.set_xticklabels(GR if k >= 2 else [""] * 5); ax.tick_params(axis="x", length=0, pad=2.5)
    neg = f == "RI"
    if neg: ax.set_ylim(-0.62, 0.12); ax.set_yticks([-0.6, -0.4, -0.2, 0]); ax.set_yticklabels(["−0.6", "−0.4", "−0.2", "0"])
    else: ax.set_ylim(-0.12, 0.66); ax.set_yticks([0, 0.2, 0.4, 0.6]); ax.set_yticklabels(["0", "0.2", "0.4", "0.6"])
    if k % 2 == 0: ax.set_ylabel("Cliff's δ (95% CI)", labelpad=2)
    if k >= 2: ax.set_xlabel("Age group (years)", labelpad=2)
    ax.set_title(nm + (" (lower in cases)" if neg else ""), fontsize=FS + 0.2, fontweight="bold", loc="left", pad=3)
    q = {t: CI[(CI.cohort == t) & (CI.feature == f)].iloc[0].q for t in ("MIMIC", "MC-MED")}
    lines = [("Age × case", MUTED), (f"ICU q {qs(q['MIMIC'])}", COL["MIMIC"]), (f"ED q {qs(q['MC-MED'])}", COL["MC-MED"])]
    if neg: lines = lines[::-1]
    for j, (txt, c) in enumerate(lines):
        yy = (0.94 - j * 0.105) if not neg else (0.05 + j * 0.105)
        ax.text(0.97, yy, txt, transform=ax.transAxes, fontsize=FS - 0.7, color=c, ha="right",
                va="top" if not neg else "bottom")

L("d", 0.548, T(101 - LOWER_SHIFT_MM))
def ramp(hex_, n):
    c = np.array(to_rgb(hex_)); w = np.ones(3)
    return [tuple(w * (1 - s) + c * s) for s in np.linspace(1.0, 0.32, n)]
YL = {"MIMIC": (-0.088, 0.082, [-0.05, 0, 0.05]), "MC-MED": (-0.094, 0.017, [-0.08, -0.04, 0])}
for k, t in enumerate(("MIMIC", "MC-MED")):
    ax = fig.add_axes([0.615, BOT[k], 0.37, PH]); framed(ax)
    ax.axhline(0, color=REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for g, col in zip(GR, ramp(COL[t], len(GR))):
        d = P[(P.cohort == t) & (P.age_group == g)].sort_values("phase")
        ax.plot(d.phase, d.diff_mean, color=col, lw=1.0, zorder=2 + (len(GR) - GR.index(g)) * 0.01)
    lo, hi, tk = YL[t]; ax.set_ylim(lo, hi); ax.set_yticks(tk)
    ax.set_yticklabels([f"{v:g}".replace("-", "−") for v in tk])
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"] if k == 1 else [""] * 6)
    ax.set_title(NAME[t], fontsize=FS + 0.2, fontweight="bold", color=COL[t], loc="left", pad=3)
    if k == 1: ax.set_xlabel("Phase of the beat (onset = 0, next onset = 1)", labelpad=2)
fig.text(0.565, (BOT[1] + BOT[0] + PH) / 2, "Mean beat, cases − controls (normalized amplitude)",
         rotation=90, rotation_mode="anchor", ha="center", va="center", fontsize=FS + 0.4)
from matplotlib.legend_handler import HandlerTuple
RA, RB = ramp(COL["MIMIC"], 5), ramp(COL["MC-MED"], 5)
hd = [(Line2D([0], [0], color=RA[i], lw=1.2), Line2D([0], [0], color=RB[i], lw=1.2)) for i in range(5)]
Y_LEG = T(102.8 - LOWER_SHIFT_MM)
lab = fig.text(0.607, Y_LEG, "Age (years)", ha="left", va="center", fontsize=FS - 0.4, fontweight="bold", fontstyle="italic")
bb = lab.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
LEG_D = fig.legend(hd, GR, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)}, ncol=5, loc="center left",
                   bbox_to_anchor=(bb.x1 + 0.010, Y_LEG), fontsize=FS - 0.6, handlelength=1.9, handletextpad=0.35,
                   columnspacing=0.8, borderaxespad=0, borderpad=0)

fig.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=3.4, color=COL["MIMIC"], label="ICU (MIMIC)"),
                    Line2D([0], [0], marker="s", ls="", ms=2.9, color=COL["MC-MED"], label="ED (MC-MED)"),
                    Line2D([0], [0], marker="o", ls="-", lw=0.9, ms=3.4, color=INK, label="Internal (cross-validated)"),
                    Line2D([0], [0], marker="o", ls=(0, (3, 1.6)), lw=0.9, ms=3.2, color=INK, mfc="white",
                           label="External (frozen model from the other system)")],
           loc="lower center", bbox_to_anchor=(0.5, 1 / H_MM), ncol=4, fontsize=FS - 0.2, handlelength=2.2, columnspacing=2.2)

OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig4_age.pdf"), facecolor="white")
fig.savefig(os.path.join(OUT, "fig4_age.svg"), facecolor="white")
fig.savefig(os.path.join(OUT, "fig4_age.png"), dpi=600, facecolor="white")
fig.savefig(os.path.join(OUT, "fig4_age.tiff"), dpi=600, facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(os.path.join(OUT, "fig4_age_preview.png"), dpi=300, facecolor="white")
print("[saved]", OUT)
plt.close(fig)
