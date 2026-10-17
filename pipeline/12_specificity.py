"""Comparator labels for the ICU cohort: cirrhosis vs controls with heart failure, sepsis, end-stage renal disease or any
of these, matched 1:1 on sex and age (+/- 3 years) without replacement.

Without --apply the matching is only reported. With --apply, lbl_spec_<k>_pos / _neg columns are added to
cache/index.csv and derived/cohort_patients.csv.
"""
import os, shutil, argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

CACHE   = os.path.join(os.environ.get("PPG_ROOT", "."), "cache")
DERIVED = os.path.join(os.environ.get("PPG_ROOT", "."), "derived")
CTRLS = {"hf": ("lbl_hf", "heart failure"), "sepsis": ("lbl_sepsis", "sepsis"),
         "esrd": ("lbl_esrd", "ESRD"), "any": (None, "any")}

def cliffs(a, b):
    u = mannwhitneyu(a, b)
    return (2*u.statistic/(len(a)*len(b))) - 1, u.pvalue

def hm_ci(auc, n1, n2):
    q1 = auc/(2-auc); q2 = 2*auc*auc/(1+auc)
    return 1.96*np.sqrt((auc*(1-auc)+(n1-1)*(q1-auc**2)+(n2-1)*(q2-auc**2))/(n1*n2))

def match(cases, ctrls, ratio, tol, seed):
    rng = np.random.default_rng(seed); used = set(); keep = []
    for sid, r in cases.sample(frac=1, random_state=seed).iterrows():
        cand = ctrls[(ctrls.g == r.g) & (abs(ctrls.age - r.age) <= tol)
                     & (~ctrls.index.isin(used))]
        if len(cand) < ratio: continue
        pick = cand.sample(n=ratio, random_state=int(rng.integers(1e9))).index
        used.update(pick); keep.append(sid)
    return keep, sorted(used)

def _oof(d, folds=10):
    oy, op = [], []
    for f in range(folds):
        tr, te = d[d.fold != f], d[d.fold == f]
        if te.y.nunique() < 2 or tr.y.nunique() < 2: continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
        m.fit(tr[["age", "male"]], tr.y)
        oy += list(te.y); op += list(m.predict_proba(te[["age", "male"]])[:, 1])
    return roc_auc_score(oy, op) if len(set(oy)) > 1 else float("nan")

def age_sex_auc(pt, pos, neg, folds=10, nperm=0, seed=0):
    d = pt.loc[pos + neg].copy()
    d["y"] = [1]*len(pos) + [0]*len(neg)
    d["male"] = (d.g == "M").astype(int)
    mi = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    mi.fit(d[["age", "male"]], d.y)
    ins = roc_auc_score(d.y, mi.predict_proba(d[["age", "male"]])[:, 1])
    ofa = _oof(d, folds)
    hi = float("nan")
    if nperm:
        rng = np.random.default_rng(seed); nulls = []
        for _ in range(nperm):
            dd = d.copy(); dd["y"] = rng.permutation(dd.y.to_numpy())
            v = _oof(dd, folds)
            if not np.isnan(v): nulls.append(v)
        hi = float(np.percentile(nulls, 97.5)) if nulls else float("nan")
    return ins, ofa, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--ratio", type=int, default=1, help="controls per case")
    ap.add_argument("--tol", type=int, default=3, help="age tolerance (years)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nperm", type=int, default=100,
                    help="permutations for the null distribution of the age+sex AUC")
    a = ap.parse_args()

    ip = os.path.join(CACHE, "index.csv")
    idx = pd.read_csv(ip)
    pt = idx.groupby("subject_id").agg(
        cirr=("lbl_cirrhosis","max"), ctrl=("lbl_control","max"),
        hf=("lbl_hf","max"), sepsis=("lbl_sepsis","max"), esrd=("lbl_esrd","max"),
        age=("age","median"), g=("gender","first"), fold=("strat_fold","first"),
        nseg=("age","size"))
    cir = pt[pt.cirr == 1]
    print("="*76); print(f" Comparators, 1:{a.ratio} matching (age +/- {a.tol} years, sex)")
    print("="*76)
    print(f"[cases] cirrhosis {len(cir)}, median age {cir.age.median():.0f}, "
          f"male {(cir.g=='M').mean():.0%}")

    print("\n--- Before matching ---")
    for k, (col, nm) in CTRLS.items():
        p = pt[(pt.ctrl == 1) & ((pt[["hf","sepsis","esrd"]].max(1) == 1) if col is None
                                 else (pt[k] == 1))]
        d, pv = cliffs(cir.age, p.age)
        ins, _, _ = age_sex_auc(pt, list(cir.index), list(p.index))
        print(f"  cirrhosis vs {nm:<14}(n={len(p):>4})  median age {cir.age.median():.0f} vs "
              f"{p.age.median():.0f}   delta={d:+.3f}   age+sex AUC {ins:.3f}")

    out = {}
    print("\n--- After matching ---")
    print(f"{'comparator':<16}{'cases':>6}{'ctrls':>6}{'age d':>9}{'p':>7}"
          f"{'age+sex in':>12}{'OOF':>8}{'null 97.5%':>12}{'CI half':>9}")
    for k, (col, nm) in CTRLS.items():
        pool = pt[(pt.ctrl == 1) & ((pt[["hf","sepsis","esrd"]].max(1) == 1) if col is None
                                    else (pt[k] == 1))]
        pos, neg = match(cir, pool, a.ratio, a.tol, a.seed)
        if len(pos) < 50:
            print(f"  {nm:<14} only {len(pos)} matched cases, skipped"); continue
        ca, co = pt.loc[pos], pt.loc[neg]
        d, pv = cliffs(ca.age, co.age)
        ins, ofa, hi = age_sex_auc(pt, pos, neg, nperm=a.nperm, seed=a.seed)
        print(f"  {nm:<14}{len(pos):>6}{len(neg):>6}{d:>+9.3f}{pv:>7.2f}"
              f"{ins:>12.3f}{ofa:>8.3f}{hi:>12.3f}{hm_ci(0.75,len(pos),len(neg)):>9.3f}")
        out[k] = (pos, neg)

    if not a.apply:
        print("\n(preview only; add --apply to write the labels)"); return

    for path in [os.path.join(DERIVED, "cohort_patients.csv"), ip]:
        if not os.path.exists(path): continue
        d = pd.read_csv(path); key = d.subject_id.astype(int)
        bak = path + ".bak"
        if not os.path.exists(bak): shutil.copy2(path, bak)
        for k, (pos, neg) in out.items():
            d[f"lbl_spec_{k}_pos"] = key.isin(pos).astype(int)
            d[f"lbl_spec_{k}_neg"] = key.isin(neg).astype(int)
        d.to_csv(path, index=False)
        print(f"[out] {os.path.basename(path)}: {2*len(out)} columns added")

if __name__ == "__main__":
    warnings.filterwarnings("ignore"); main()
