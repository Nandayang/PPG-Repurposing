"""Supplementary Tables 2b and 3: AUC by age group with case-mix descriptors, age gradient after excluding comorbid
controls or by ED disposition, and ED checks (signal-quality tertiles, cases with cirrhosis on imaging).
2,000 patient-level bootstrap resamples.

Usage: python 34_supplementary_tables.py [PPG_ROOT]
Output: results/paper/supp_tables.json
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

R = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PPG_ROOT", "."))
OUT = R / "results" / "paper" / "supp_tables.json"; OUT.parent.mkdir(parents=True, exist_ok=True)
AGES = [("<45", 0, 45), ("45–49", 45, 50), ("50–59", 50, 60), ("60–69", 60, 70), ("≥70", 70, 200)]


def auc_ci(y, p, n=2000, seed=0):
    y, p = np.asarray(y), np.asarray(p, float); ok = np.isfinite(p); y, p = y[ok], p[ok]
    a = roc_auc_score(y, p); rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            bs.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(auc=float(a), lo=float(lo), hi=float(hi), n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()))


def agegrp(a):
    for lab, lo, hi in AGES:
        if lo <= a < hi:
            return lab


mi = pd.read_csv(R / "results/es/patient_oof_mimic_cirrhosis_ssl.csv")
co = pd.read_csv(R / "derived/cohort_patients.csv")
mi = mi.merge(co[["subject_id", "lbl_hf", "lbl_sepsis", "lbl_esrd", "lbl_cirr_decomp"]],
              left_on="pid", right_on="subject_id", how="left")
mc = pd.read_csv(R / "results/es/patient_oof_mcmed_cirrhosis_ssl.csv")
tb = lambda s: s.astype(str).isin(["True", "1", "1.0"])
for c in ["dx_decomp", "dx_hf", "bb_ever", "rad_cirr"]:
    mc[c] = tb(mc[c])
ext = pd.read_csv(R / "results/mcmed/patient_preds.csv")
ext = ext[(ext.n_win.fillna(0) >= 10) & ((ext.in_cirrhosis == 1) | (ext.in_control == 1))].copy()
ext["y"] = ext.in_cirrhosis.astype(int); ext["score"] = ext.s_es
for c in ["rad_cirr"]:
    ext[c] = tb(ext[c])

T = {}
rows = []
for name, D in (("MIMIC", mi), ("MC-MED", mc)):
    D = D.copy(); D["ag"] = D.age.map(agegrp)
    for lab, _, _ in AGES:
        q = D[D.ag == lab]; ca, ct = q[q.y == 1], q[q.y == 0]
        r = dict(cohort=name, age=lab, **auc_ci(q.y, q.score),
                 ctrl_median=float(ct.score.median()), case_median=float(ca.score.median()))
        if name == "MIMIC":
            r.update(decomp=float(ca.lbl_cirr_decomp.mean() * 100), ctrl_sepsis=float(ct.lbl_sepsis.mean() * 100),
                     ctrl_hf=float(ct.lbl_hf.mean() * 100), ctrl_esrd=float(ct.lbl_esrd.mean() * 100))
        else:
            dbp = auc_ci(q.y, -q.Triage_DBP)
            r.update(decomp=float(ca.dx_decomp.mean() * 100),
                     ctrl_admitted=float(ct.ED_dispo.isin(["Inpatient", "ICU", "Observation"]).mean() * 100),
                     ctrl_hf=float(ct.dx_hf.mean() * 100), ctrl_bb=float(ct.bb_ever.mean() * 100),
                     dbp_auc=dbp["auc"], dbp_lo=dbp["lo"], dbp_hi=dbp["hi"])
        rows.append(r)
T["age_a"] = rows

rows = []
q0 = mi[(mi.lbl_hf != 1) & (mi.lbl_sepsis != 1) & (mi.lbl_esrd != 1)].copy(); q0["ag"] = q0.age.map(agegrp)
for lab, _, _ in AGES:
    rows.append(dict(setting="ICU, excluding HF, sepsis and ESRD", age=lab, **auc_ci(q0[q0.ag == lab].y, q0[q0.ag == lab].score)))
for nm, codes in (("ED, admitted or under observation", ["Inpatient", "ICU", "Observation"]), ("ED, discharged", ["Discharge"])):
    q = mc[mc.ED_dispo.isin(codes)].copy(); q["ag"] = q.age.map(agegrp)
    for lab, _, _ in AGES:
        rows.append(dict(setting=nm, age=lab, **auc_ci(q[q.ag == lab].y, q[q.ag == lab].score)))
T["age_b"] = rows

rows = []
for nm, D in (("Internal", mc), ("Frozen ICU model", ext)):
    D = D.copy()
    D["tert"] = pd.qcut(D.frac_good.rank(method="first"), 3, labels=["Lowest", "Middle", "Highest"])
    for t in ["Lowest", "Middle", "Highest"]:
        q = D[D.tert == t]
        rows.append(dict(model=nm, check=f"Signal quality, {t.lower()} tertile",
                         frac=[float(q.frac_good.min()), float(q.frac_good.max())], **auc_ci(q.y, q.score)))
    q = D[(D.y == 0) | ((D.y == 1) & D.rad_cirr)]
    rows.append(dict(model=nm, check="Cases with cirrhosis described on imaging", **auc_ci(q.y, q.score)))
T["ed_checks"] = rows

OUT.write_text(json.dumps(T, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in T.items():
    print("==", k)
    for r in v:
        print({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in r.items()})
