"""Fig. 2 - Detection of cirrhosis: a ROC curves, b model vs baselines, c clinical subgroups and disease comparators
(dedicated models on age- and sex-matched samples).

Usage: python draw_fig2.py [--data-dir DIR] [--out-dir DIR]
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory as blend
from matplotlib.textpath import TextToPath

ROOT = Path(__file__).resolve().parent
COL = {"MIMIC": "#2f6b85", "MC-MED": "#c0643c"}
MARKER = {"MIMIC": "o", "MC-MED": "s"}
MARKER_SIZE = {"MIMIC": 2.6, "MC-MED": 2.3}
INK, MUTED, FRAME, REF = "#1a1a1a", "#666666", "#4d4d4d", "#8c8c8c"
FS = 6.4
WIDTH_MM, HEIGHT_MM = 183, 146
FOOTER_MM = 4
TOP_SIDE_MM = 46.5
B_LABEL_X, B_FRAME_X, B_EST_X = 0.415, 0.555, 0.822
B_MODELS = ("Age + sex", "Pulse rate", "Pulse-rate variability", "Vital signs*", "PPG model")
COHORTS = (("MIMIC", "ICU (MIMIC)"), ("MC-MED", "ED (MC-MED)"))
CLINICAL = (("All cirrhosis", "All cirrhosis"),
            ("Decompensated", "Decompensated"),
            ("Compensated", "Compensated"),
            ("ED discharged", "Discharged from the ED"))
COMPARATORS = (("Heart failure", "Heart failure"),
               ("Any critical illness", "HF / sepsis / ESRD"),
               ("Sepsis", "Sepsis"), ("ESRD", "ESRD"))
MODEL_NOTE = "* Disease-control rows use separately trained models and age/sex-matched samples."

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS, "axes.labelsize": FS + 0.4,
    "xtick.labelsize": FS, "ytick.labelsize": FS,
    "axes.linewidth": 0.7, "axes.edgecolor": FRAME,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.major.size": 2.5, "xtick.major.width": 0.6,
    "ytick.major.size": 2.5, "ytick.major.width": 0.6,
    "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def load_data(data_dir):
    names = ("fig2_estimates.csv", "fig2a_roc_curves.csv",
             "fig5a_specificity_source_data.csv", "fig5a_specificity.csv", "fig2b_prv.csv")
    summ, curves, dedicated, check, prv = [
        pd.read_csv(data_dir / name, encoding="utf-8-sig") for name in names]
    assert len(prv) == 2 and set(prv.panel) == {"b"} and set(prv.kind) == {"int"}
    assert set(prv.group) == {"Pulse-rate variability"} and set(prv.test) == {c for c, _ in COHORTS}
    for cohort, _ in COHORTS:
        ref = summ[(summ.panel == "b") & (summ.test == cohort) & (summ.group == "PPG model")].iloc[0]
        new = prv[prv.test == cohort].iloc[0]
        assert (new.n_pos, new.n_neg) == (ref.n_pos, ref.n_neg)
    summ = pd.concat([summ, prv], ignore_index=True)
    for data in (summ, dedicated):
        intervals = data[["lo", "auc", "hi"]].to_numpy(float)
        assert np.isfinite(intervals).all()
        assert (intervals >= 0).all() and (intervals <= 1).all()
        assert (np.diff(intervals, axis=1) >= 0).all()
        counts = data[["n_pos", "n_neg"]].to_numpy(float)
        assert (counts > 0).all() and (counts == np.floor(counts)).all()
    expected = {("MIMIC", "Heart failure"), ("MIMIC", "Any critical illness"),
                ("MIMIC", "Sepsis"), ("MC-MED", "Heart failure"), ("MC-MED", "ESRD")}
    assert len(dedicated) == 5
    assert set(zip(dedicated.test, dedicated.comparator)) == expected
    assert not dedicated.duplicated(["test", "comparator"]).any()
    assert (dedicated.n == dedicated.n_pos + dedicated.n_neg).all()
    merged = dedicated.merge(check, on=["test", "comparator"], validate="one_to_one",
                             suffixes=("", "_check"))
    for field in ("auc", "lo", "hi", "agesex_auc"):
        assert np.array_equal(merged[field], merged[field + "_check"])
    assert dedicated.agesex_auc.between(0, 1).all()
    assert not summ.duplicated(["panel", "test", "kind", "group"]).any()
    for cohort, _ in COHORTS:
        for kind in ("int", "ext"):
            points = curves[(curves.test == cohort) & (curves.kind == kind)]
            assert len(points) > 1
            assert points[["fpr", "tpr"]].apply(lambda x: x.between(0, 1).all()).all()
            assert (np.diff(points.fpr) >= 0).all() and (np.diff(points.tpr) >= 0).all()
    return summ, curves, dedicated, names


def get(summ, panel, cohort, kind, group):
    data = summ[(summ.panel == panel) & (summ.test == cohort)
                & (summ.kind == kind) & (summ.group == group)]
    assert len(data) <= 1
    return None if data.empty else data.iloc[0]


def forest(fig, box, rows, cols, xlim, xticks, est_x):
    ax = fig.add_axes(box)
    n = len(rows)
    ax.set_ylim(n - 0.4, -0.9)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_yticks([])
    ax.set_xlabel("AUC", labelpad=2.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.axvline(0.5, color=REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
    tr = blend(fig.transFigure, ax.transData)
    for key, x, ha, hdr in cols + [("est", est_x, "left", "AUC (95% CI)")]:
        fig.text(x, -1.55, hdr, transform=tr, ha=ha, va="center",
                 fontsize=FS, fontweight="bold", fontstyle="italic")
    i = 0
    while i < len(rows):
        if not rows[i].get("shade"):
            i += 1
            continue
        start = i
        while i < len(rows) and rows[i].get("shade"):
            i += 1
        fig.add_artist(Rectangle((cols[0][1] - 0.008, start - 0.5),
                                 est_x + 0.165 - cols[0][1] + 0.008, i - start,
                                 transform=tr, fc="#f1f1f1", ec="none", zorder=-10))
    ax.set_facecolor("none")
    for i, row in enumerate(rows):
        if "head" in row:
            fig.text(cols[0][1], i, row["head"], transform=tr, ha="left", va="center",
                     fontsize=FS + 0.2, fontweight="bold", color=row.get("hcolor", INK))
            continue
        for key, x, ha, hdr in cols:
            value = row.get(key, "")
            yy = i + 0.5 if key in row.get("span2", ()) else i
            if value:
                fig.text(x, yy, value, transform=tr, ha=ha, va="center",
                         fontsize=FS, color=row.get(key + "_c", INK))
        fig.text(est_x, i, f"{row['auc']:.3f} ({row['lo']:.3f}–{row['hi']:.3f})",
                 transform=tr, ha="left", va="center", fontsize=FS)
        colour = row["color"]
        ax.errorbar(row["auc"], i,
                    xerr=[[row["auc"] - row["lo"]], [row["hi"] - row["auc"]]],
                    fmt=row.get("marker", "o"), ms=row.get("ms", 2.6), color=colour,
                    mfc=colour if row.get("filled", True) else "white",
                    mec=colour, mew=0.8, elinewidth=0.8,
                    capsize=1.8, capthick=0.8, zorder=3)
        if "agesex_auc" in row:
            ax.plot(row["agesex_auc"], i, marker="|", ls="", ms=5.2,
                    color=REF, mew=1.0, zorder=2)
    return ax


def make_panel_c_rows(summ, dedicated):
    rows, source_rows = [], []
    for cohort, name in COHORTS:
        rows.append(dict(head=name, hcolor=COL[cohort]))
        group_index = 0
        for group, label in CLINICAL:
            ri, re = [get(summ, "c", cohort, kind, group) for kind in ("int", "ext")]
            if ri is None:
                assert re is None and cohort == "MIMIC" and group == "ED discharged"
                continue
            assert re is not None
            assert ri.n_pos == re.n_pos and ri.n_neg == re.n_neg
            shade = group_index % 2 == 1
            for kind, value, filled in (("int", ri, True), ("ext", re, False)):
                first = kind == "int"
                rows.append(dict(lab="   " + label if first else "",
                                 val="Internal" if first else "External",
                                 val_c=INK if first else MUTED,
                                 n=f"{int(value.n_pos):,} / {int(value.n_neg):,}" if first else "",
                                 auc=float(value.auc), lo=float(value.lo), hi=float(value.hi),
                                 color=COL[cohort], filled=filled, shade=shade,
                                 marker=MARKER[cohort], ms=MARKER_SIZE[cohort],
                                 span2=("lab", "n") if first else ()))
                source_rows.append(dict(test=cohort, comparison=group, model="Screening",
                                        validation=kind, auc=value.auc, lo=value.lo, hi=value.hi,
                                        n_pos=int(value.n_pos), n_neg=int(value.n_neg),
                                        agesex_auc=np.nan))
            group_index += 1
        for comparator, label in COMPARATORS:
            data = dedicated[(dedicated.test == cohort) & (dedicated.comparator == comparator)]
            if data.empty:
                continue
            value = data.iloc[0]
            rows.append(dict(lab="   " + label, val="dedicated*",
                             n=f"{int(value.n_pos):,} / {int(value.n_neg):,}",
                             auc=float(value.auc), lo=float(value.lo), hi=float(value.hi),
                             agesex_auc=float(value.agesex_auc), marker=MARKER[cohort],
                             ms=MARKER_SIZE[cohort],
                             color=COL[cohort], filled=True, shade=group_index % 2 == 1))
            source_rows.append(dict(test=cohort, comparison=comparator, model="Dedicated",
                                    validation="int", auc=value.auc, lo=value.lo, hi=value.hi,
                                    n_pos=int(value.n_pos), n_neg=int(value.n_neg),
                                    agesex_auc=value.agesex_auc))
            group_index += 1
    source = pd.DataFrame(source_rows)
    assert len(rows) == 21 and len(source) == 19
    assert (source.model == "Screening").sum() == 14
    assert (source.model == "Dedicated").sum() == 5
    assert not source.comparison.str.contains("age <45", regex=False).any()
    return rows, source


def draw(data_dir, out_dir):
    summ, curves, dedicated, names = load_data(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    Y = lambda mm: mm / HEIGHT_MM
    T = lambda mm: 1 - mm / HEIGHT_MM
    for label, x, y in (("a", 0.012, T(1.5)), ("b", B_LABEL_X - 0.020, T(1.5)),
                         ("c", 0.012, Y(83 + FOOTER_MM))):
        fig.text(x, y, label, fontsize=9, fontweight="bold", va="top")

    top_width = TOP_SIDE_MM / WIDTH_MM
    top_height = TOP_SIDE_MM / HEIGHT_MM
    ax = fig.add_axes([0.06, T(53.25), top_width, top_height])
    ax_a = ax
    ax.plot([0, 1], [0, 1], color="#d0d0d0", lw=0.7, zorder=0)
    for cohort, kind in (("MIMIC", "int"), ("MC-MED", "int"),
                         ("MIMIC", "ext"), ("MC-MED", "ext")):
        points = curves[(curves.test == cohort) & (curves.kind == kind)]
        auc = get(summ, "c", cohort, kind, "All cirrhosis").auc
        ax.plot(points.fpr, points.tpr, color=COL[cohort], lw=1,
                ls="-" if kind == "int" else (0, (3, 1.6)),
                label=f"{'ICU' if cohort == 'MIMIC' else 'ED'}, "
                      f"{'internal' if kind == 'int' else 'external'}  {auc:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, .5, 1], yticks=[0, .5, 1],
           xlabel="1 − Specificity", ylabel="Sensitivity")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=FS - 0.6, handlelength=2,
              borderaxespad=0.3, labelspacing=0.35, title="AUC", title_fontsize=FS - 0.4)

    rows_b = []
    for cohort, name in COHORTS:
        rows_b.append(dict(head=name, hcolor=COL[cohort]))
        for model in B_MODELS:
            value = get(summ, "b", cohort, "int", model)
            assert value is not None
            rows_b.append(dict(lab="   " + model, auc=value.auc, lo=value.lo, hi=value.hi,
                               marker=MARKER[cohort], ms=MARKER_SIZE[cohort],
                               color=COL[cohort] if model == "PPG model" else "#7a7a7a"))
    ax_b = forest(fig, [B_FRAME_X, T(53.25), top_width, top_height], rows_b,
                  [("lab", B_LABEL_X, "left", "Predictors")],
                  (0.4, 0.9), [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], B_EST_X)

    rows_c, source_c = make_panel_c_rows(summ, dedicated)
    forest(fig, [0.455, Y(12.75 + FOOTER_MM), 0.365, 64.5 / HEIGHT_MM], rows_c,
           [("lab", 0.035, "left", "Subgroup / disease control"),
            ("val", 0.255, "left", "Validation"),
            ("n", 0.425, "right", "Cases / controls")],
           (0.40, 0.95), [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 0.835)
    handles = [
        Line2D([0], [0], marker="o", ls="", ms=3.4, color=COL["MIMIC"], label="ICU (MIMIC)"),
        Line2D([0], [0], marker="s", ls="", ms=2.9, color=COL["MC-MED"], label="ED (MC-MED)"),
        Line2D([0], [0], marker="o", ls="-", lw=0.9, ms=2.6, color=INK,
               label="Internal (cross-validated)"),
        Line2D([0], [0], marker="o", ls=(0, (3, 1.6)), lw=0.9, ms=2.6, color=INK,
               mfc="white", label="External (frozen model from the other system)"),
        Line2D([0], [0], marker="|", ls="", ms=5.2, color=REF,
               mew=1, label="Age + sex (disease controls)"),
    ]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, Y(4.4)),
                        ncol=5, fontsize=FS - 0.2, handlelength=1.8, handletextpad=0.3,
                        columnspacing=1.2, borderpad=0.2, borderaxespad=0)
    fig.text(0.54, Y(1.3), MODEL_NOTE, fontsize=5.7, color=MUTED, ha="center", va="bottom")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    assert legend_box.x0 >= 0 and legend_box.x1 <= 1
    b_labels = [t for t in fig.texts if t.get_position()[0] == B_LABEL_X]
    assert all(t.get_ha() == "left" for t in b_labels)
    text_mm = lambda t: TextToPath().get_text_width_height_descent(
        t.get_text(), t.get_fontproperties(), ismath=False)[0] / 72 * 25.4
    b_label_gap_mm = (B_FRAME_X - B_LABEL_X) * WIDTH_MM - max(text_mm(t) for t in b_labels)
    assert b_label_gap_mm >= 2.0, b_label_gap_mm
    a_right = ax_a.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).x1
    assert a_right < B_LABEL_X - 0.020
    box_a, box_b = ax_a.get_position(), ax_b.get_position()
    assert np.allclose([box_a.y0, box_a.width, box_a.height],
                       [box_b.y0, box_b.width, box_b.height], rtol=0, atol=1e-12)

    prefix = out_dir / "fig2_detection"
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out_dir / "fig2_preview.png", dpi=300, facecolor="white")
    plt.close(fig)

    panel_height_mm = 86 + FOOTER_MM
    fig_c = plt.figure(figsize=(WIDTH_MM / 25.4, panel_height_mm / 25.4))
    fig_c.text(0.012, (83 + FOOTER_MM) / panel_height_mm, "c", fontsize=9, fontweight="bold", va="top")
    forest(fig_c, [0.455, (12.75 + FOOTER_MM) / panel_height_mm, 0.365, 64.5 / panel_height_mm], rows_c,
           [("lab", 0.035, "left", "Subgroup / disease control"),
            ("val", 0.255, "left", "Validation"),
            ("n", 0.425, "right", "Cases / controls")],
           (0.40, 0.95), [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 0.835)
    fig_c.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 4.4 / panel_height_mm),
                 ncol=5, fontsize=FS - 0.2, handlelength=1.8, handletextpad=0.3,
                 columnspacing=1.2, borderpad=0.2, borderaxespad=0)
    fig_c.text(0.54, 1.3 / panel_height_mm, MODEL_NOTE, fontsize=5.7,
               color=MUTED, ha="center", va="bottom")
    fig_c.savefig(out_dir / "fig2c.pdf", facecolor="white")
    fig_c.savefig(out_dir / "fig2c.svg", facecolor="white")
    fig_c.savefig(out_dir / "fig2c_preview.png", dpi=300, facecolor="white")
    plt.close(fig_c)

    source_c.to_csv(out_dir / "fig2c_source_data.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "canvas_mm": [WIDTH_MM, HEIGHT_MM],
        "source_hashes": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                          for name in names},
    }
    (out_dir / "fig2_data_check.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out")
    args = parser.parse_args()
    draw(args.data_dir, args.out_dir)
