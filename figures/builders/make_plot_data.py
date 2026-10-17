"""Summary tables for Figs. 2, 4, 5, 7 and Supplementary Figs. 1-2 (ROC curves, AUCs with CIs, calibration bins,
recording duration, emulated degradation, age, MELD), written to figures/data/. No patient-level rows.

Usage: python figures/builders/make_plot_data.py [--boot 1000] [--only 2,3,4,5,6,7]
"""
import os, json, argparse, warnings, importlib.util
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

ROOT = os.environ.get("PPG_ROOT", ".")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

spec = importlib.util.spec_from_file_location("m24", os.path.join(HERE, "..", "..", "pipeline", "24_figure_inputs.py"))
m24 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m24)
auc_ci, logit, NAME = m24.auc_ci, m24.logit, m24.NAME
KEYS = [("MIMIC", "int"), ("MIMIC", "ext"), ("MC-MED", "int"), ("MC-MED", "ext")]


def out(df, name):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object: df[c] = df[c].map(lambda v: v.replace("\n", "\\n") if isinstance(v, str) else v)
    df.to_csv(os.path.join(DATA, name), index=False, encoding="utf-8-sig", float_format="%.6g")
    print(f"[saved] {name}  ({len(df)} rows)", flush=True)


def kcols(k):
    return dict(test=k[0], kind=k[1], setting=NAME[k])


def fig2(S):
    rows, aucs = [], []
    for k in [("MIMIC", "int"), ("MC-MED", "int"), ("MIMIC", "ext"), ("MC-MED", "ext")]:
        D = S[k]; f, t, _ = roc_curve(D.y, D.score); a = auc_ci(D.y, D.score)
        rows += [dict(**kcols(k), fpr=x, tpr=y) for x, y in zip(f, t)]
        aucs.append(dict(**kcols(k), auc=a[0], lo=a[1], hi=a[2]))
    out(pd.DataFrame(rows), "fig2a_roc_curves.csv"); out(pd.DataFrame(aucs), "fig2a_roc_auc.csv")

    rows = []
    mi, mc = S[("MIMIC", "int")], S[("MC-MED", "int")]
    for nm, D, base in [("MIMIC", mi, [("Age + sex", ["age", "male"]), ("PPG pulse rate", ["ppg_hr"]),
                                       ("Vital signs*", ["age", "male", "hr", "sbp", "dbp", "rr"])]),
                        ("MC-MED", mc, [("Age + sex", ["age", "male"]), ("PPG pulse rate", ["ppg_hr"]),
                                        ("Vital signs*", ["age", "male", "Triage_HR", "Triage_RR", "Triage_SpO2",
                                                          "Triage_SBP", "Triage_DBP"])])]:
        for lab, cols in base:
            a = auc_ci(D.y, m24.fold_baseline(D, cols)); rows.append(dict(test=nm, model=lab, is_ppg=0, auc=a[0], lo=a[1], hi=a[2]))
        a = auc_ci(D.y, D.score); rows.append(dict(test=nm, model="PPG model", is_ppg=1, auc=a[0], lo=a[1], hi=a[2]))
    out(pd.DataFrame(rows), "fig2b_baselines.csv")

    subs = [("All cirrhosis", lambda D: np.ones(len(D), bool)),
            ("Decompensated", lambda D: D.dec.to_numpy() == 1),
            ("Compensated", lambda D: D.com.to_numpy() == 1),
            ("Compensated, age < 60", lambda D: (D.com.to_numpy() == 1) & (D.age.to_numpy() < 60)),
            ("ED discharged", None)]
    rows = []
    for i, (lab, f) in enumerate(subs):
        for k in KEYS:
            D = S[k]; ctl = D.y.to_numpy() == 0
            if f is None:
                if k[0] != "MC-MED": continue
                m = D.ED_dispo.astype(str).eq("Discharge").to_numpy()
            elif "age < 60" in lab:
                m = (ctl & (D.age.to_numpy() < 60)) | ((D.y.to_numpy() == 1) & f(D))
            else:
                m = ctl | ((D.y.to_numpy() == 1) & f(D))
            q = D[m]
            if q.y.sum() < 20: continue
            a = auc_ci(q.y, q.score)
            rows.append(dict(subgroup=lab, order=i, **kcols(k), auc=a[0], lo=a[1], hi=a[2],
                             n_pos=int(q.y.sum()), n_neg=int((1 - q.y).sum())))
    out(pd.DataFrame(rows), "fig2c_subgroups.csv")


def fig3(S):
    M = [("MIMIC", "MIMIC", ("MIMIC", "int")), ("MC-MED", "MC-MED", ("MC-MED", "int")),
         ("MC-MED", "MIMIC", ("MIMIC", "ext")), ("MIMIC", "MC-MED", ("MC-MED", "ext"))]
    rows = []
    for tr, te, k in M:
        a = auc_ci(S[k].y, S[k].score); rows.append(dict(trained_on=tr, tested_on=te, auc=a[0], lo=a[1], hi=a[2]))
    out(pd.DataFrame(rows), "fig3a_matrix.csv")

    rows = []
    for pi_, (k, lab) in enumerate([(("MC-MED", "ext"), "MIMIC model in MC-MED"), (("MIMIC", "ext"), "MC-MED model in MIMIC")]):
        D = S[k]; y = D.y.to_numpy().astype(int); p = np.clip(D.score.to_numpy(), 1e-6, 1 - 1e-6); z = logit(p)
        pr = np.zeros(len(y))
        for trn, tst in StratifiedKFold(5, shuffle=True, random_state=0).split(z, y):
            m = LogisticRegression(C=1e6, max_iter=1000).fit(z[trn, None], y[trn]); pr[tst] = m.predict_proba(z[tst, None])[:, 1]
        for q, ver in [(p, "Frozen"), (pr, "Locally recalibrated")]:
            bins = np.quantile(q, np.linspace(0, 1, 11)); b = np.clip(np.digitize(q, bins[1:-1]), 0, 9)
            sl = LogisticRegression(C=1e6, max_iter=1000).fit(logit(q)[:, None], y).coef_[0][0]
            for i in range(10):
                rows.append(dict(panel="bc"[pi_], title=lab, test=k[0], version=ver, slope=sl, decile=i + 1,
                                 predicted=q[b == i].mean(), observed=y[b == i].mean(), n=int((b == i).sum())))
    out(pd.DataFrame(rows), "fig3bc_calibration.csv")


def fig4():
    R = m24.load_spot()
    xs_map = {"1": 0.5, "2": 1, "4": 2, "6": 3, "10": 5, "20": 10, "50": 25}
    rows, rows_b = [], []
    for data, test in [("mimic", "MIMIC"), ("mcmed", "MC-MED")]:
        for r in R.get(f"A_{data}", []):
            for kind, fld in [("int", "internal"), ("ext", "external")]:
                rows.append(dict(test=test, kind=kind, n_windows=r["n"], minutes=xs_map.get(str(r["n"]), 60),
                                 auc=r[fld][0], lo=r[fld][1], hi=r[fld][2], auc_spread=r.get(f"{fld}_rand")))
    out(pd.DataFrame(rows), "fig4ab_recording_length.csv")
    labels = ["Original", "60 Hz", "+5.5 bit", "+Clipping", "+Gain jumps\n≈CMS50E", "Stress"]
    for data, test in [("mimic", "MIMIC"), ("mcmed", "MC-MED")]:
        for i, r in enumerate(R.get(f"B_{data}", [])):
            for kind, fld in [("int", "internal"), ("ext", "external")]:
                rows_b.append(dict(test=test, kind=kind, step=i, condition=r["cond"], label=labels[i] if i < len(labels) else r["cond"],
                                   auc=r[fld][0], lo=r[fld][1], hi=r[fld][2], usable_fraction=r.get("ok")))
    if rows_b: out(pd.DataFrame(rows_b), "fig4c_hardware.csv")


def fig5():
    items = [("Heart failure", "mimic", "spec_hf"), ("Heart failure", "mcmed", "spec_hf"), ("ESRD", "mcmed", "spec_esrd"),
             ("Any critical illness", "mimic", "spec_any"), ("Sepsis", "mimic", "spec_sepsis")]
    rows = []
    for lab, data, task in items:
        f = os.path.join(m24.RES, "es", f"result_{data}_{task}_ssl.json")
        if not os.path.exists(f): continue
        j = json.load(open(f, encoding="utf-8"))
        rows.append(dict(comparator=lab, test="MIMIC" if data == "mimic" else "MC-MED", auc=j["auc"], lo=j["ci"][0], hi=j["ci"][1],
                         agesex_auc=j.get("baselines", {}).get("age+sex")))
    out(pd.DataFrame(rows), "fig5a_specificity.csv")
    f = os.path.join(m24.RES, "ml", "result_mimic_ssl_w100.json")
    if os.path.exists(f):
        j = json.load(open(f, encoding="utf-8"))["spec_matrix"]
        dz = {"cirrhosis": "Cirrhosis", "hf": "HF", "sepsis": "Sepsis", "esrd": "ESRD"}
        rows = [dict(model_output=dz[a], comparator=dz[b], auc=(j.get(f"{a}_vs_{b}", [np.nan])[0] if a != b else np.nan))
                for a in dz for b in dz]
        out(pd.DataFrame(rows), "fig5b_multilabel_matrix.csv")


def fig6(S):
    ages = np.linspace(35, 85, 51); rows, inter = [], []
    for k in KEYS:
        D = S[k].dropna(subset=["score", "age"]); y = D.y.to_numpy().astype(float)
        s = logit(np.clip(D.score.to_numpy(), 0.5 / len(D), 1 - 0.5 / len(D))); s = (s - s.mean()) / s.std()
        a = (D.age.clip(upper=90).to_numpy() - 60) / 10
        X = np.column_stack([np.ones_like(s), s, a, D.male.to_numpy().astype(float), s * a])
        b, V = m24.logistic_mle(X, y)
        t = (ages - 60) / 10; g = np.column_stack([np.ones_like(t), t])
        est = b[1] + b[4] * t; se = np.sqrt(np.einsum("ij,jk,ik->i", g, V[np.ix_([1, 4], [1, 4])], g))
        rows += [dict(**kcols(k), age=x, odds_ratio=np.exp(e), lo=np.exp(e - 1.96 * s_), hi=np.exp(e + 1.96 * s_))
                 for x, e, s_ in zip(ages, est, se)]
        z = b[4] / np.sqrt(V[4, 4]); inter.append(dict(**kcols(k), beta_per_decade=b[4], se=np.sqrt(V[4, 4]), p=2 * norm.sf(abs(z))))
    out(pd.DataFrame(rows), "fig6a_or_by_age.csv"); out(pd.DataFrame(inter), "fig6a_interaction.csv")
    bins = [(0, 50, "<50"), (50, 60, "50–59"), (60, 70, "60–69"), (70, 200, "≥70")]
    rows = []
    for k in KEYS:
        D = S[k]
        for i, (lo, hi, lab) in enumerate(bins):
            q = D[(D.age >= lo) & (D.age < hi)]; a = auc_ci(q.y, q.score)
            rows.append(dict(**kcols(k), age_group=lab, order=i, auc=a[0], lo=a[1], hi=a[2], n_pos=int(q.y.sum()), n_neg=int((1 - q.y).sum())))
    out(pd.DataFrame(rows), "fig6b_auc_by_age.csv")


def fig7(S):
    Lm = pd.read_csv(os.path.join(m24.DERIVED, "labs_median.csv")).merge(pd.read_csv(os.path.join(m24.DERIVED, "labs_meld_mimic.csv")),
                                                                         on="subject_id", how="outer").rename(columns={"subject_id": "pid"})
    Lc = pd.read_csv(os.path.join(m24.DERIVED, "mcmed_labs.csv")).rename(columns={"CSN": "pid"})
    cats = [(0, 10, "<10"), (10, 15, "10–14"), (15, 20, "15–19"), (20, 99, "≥20")]
    rows = []
    for k, L in [(("MIMIC", "int"), Lm), (("MC-MED", "int"), Lc)]:
        D = S[k]; c = D[D.y == 1].merge(L, on="pid", how="left"); c["meld"] = m24.meld(c)
        ctl = D[D.y == 0].score.median()
        for i, (lo, hi, lab) in enumerate(cats):
            q = c[(c.meld >= lo) & (c.meld < hi)].score
            q1, md, q3 = q.quantile([.25, .5, .75]) if len(q) else (np.nan,) * 3
            rows.append(dict(test=k[0], meld_group=lab, order=i, n=len(q), q1=q1, median=md, q3=q3, control_median=ctl))
    out(pd.DataFrame(rows), "fig7a_score_by_meld.csv")

    rows = []
    f = os.path.join(m24.RES, "es", "result_mimic_decomp_ssl.json")
    if os.path.exists(f):
        j = json.load(open(f, encoding="utf-8"))
        rows.append(dict(label="ICU: decompensated\nvs compensated†", test="MIMIC", kind="int", auc=j["auc"], lo=j["ci"][0], hi=j["ci"][1]))
    f = os.path.join(m24.RES, "acute", "result_23.json")
    if os.path.exists(f):
        j = json.load(open(f, encoding="utf-8"))
        kh, ka, kv = [[k] if k in j else [] for k in ("history", "A", "A_vs_verified")]
        for lab, kk in [("ED: history of\ndecompensation", kh), ("ED: decompensated\nat this visit‡", ka),
                        ("ED: this visit vs imaging-\nverified compensated‡", kv)]:
            if kk:
                for kind, fld in [("int", "cirr_int"), ("ext", "cirr_ext")]:
                    v = j[kk[0]][fld]; rows.append(dict(label=lab, test="MC-MED", kind=kind, auc=v[0], lo=v[1], hi=v[2]))
    out(pd.DataFrame(rows), "fig7b_decompensation.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--only", default="2,3,4,5,6,7")
    a = ap.parse_args(); m24.BOOT = a.boot; todo = a.only.split(",")
    warnings.filterwarnings("ignore"); os.makedirs(DATA, exist_ok=True)
    S = m24.load_sets()
    for k, f in [("2", lambda: fig2(S)), ("3", lambda: fig3(S)), ("4", fig4), ("5", fig5), ("6", lambda: fig6(S)), ("7", lambda: fig7(S))]:
        if k in todo: f()


if __name__ == "__main__":
    main()
