"""MIMIC laboratory values (AST, ALT, platelets, albumin, bilirubin, INR) as medians within each patient's ICU stays,
FIB-4 / APRI, and the cross-validated logistic model used for combined scores (imported by 21_paper_analyses.py).

Output: derived/labs_median.csv (run before 21_paper_analyses.py)
"""
import os, argparse, warnings, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT    = os.environ.get("PPG_ROOT", ".")
CACHE   = os.path.join(ROOT, "cache")
DERIVED = os.path.join(ROOT, "derived")
MIMIC   = os.environ.get("MIMIC_DIR", "data/mimic-iii-clinical-database-1.4")

ITEMS = {50878: "ast", 50861: "alt", 51265: "plt",
         50862: "alb", 50885: "bili", 51237: "inr"}
RANGE = {"ast": (1, 20000), "alt": (1, 20000), "plt": (1, 2000),
         "alb": (0.5, 7), "bili": (0.05, 60), "inr": (0.5, 20)}

def _find(name):
    for p in (os.path.join(MIMIC, name), os.path.join(MIMIC, name + ".gz")):
        if os.path.exists(p): return p
    raise SystemExit(f"[!] {name} not found in {MIMIC}")

def extract_labs(sids, agg="median", chunk=2_000_000):
    icu = pd.read_csv(_find("ICUSTAYS.csv"), usecols=["SUBJECT_ID", "INTIME", "OUTTIME"],
                      parse_dates=["INTIME", "OUTTIME"])
    icu.columns = [c.lower() for c in icu.columns]
    icu = icu[icu.subject_id.isin(sids)]
    win = icu.groupby("subject_id").agg(t0=("intime", "min"), t1=("outtime", "max"))
    print(f"[labs] ICU stays of {len(win):,} patients")

    path = _find("LABEVENTS.csv"); keep = []
    t0 = time.time(); n = 0
    for ch in pd.read_csv(path, usecols=["SUBJECT_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
                          parse_dates=["CHARTTIME"], chunksize=chunk):
        ch.columns = [c.lower() for c in ch.columns]
        n += len(ch)
        ch = ch[ch.itemid.isin(ITEMS) & ch.subject_id.isin(sids) & ch.valuenum.notna()]
        if len(ch): keep.append(ch)
        print(f"   {n/1e6:.1f}M rows  {time.time()-t0:.0f}s", end="\r", flush=True)
    lab = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    print(f"\n[labs] {len(lab):,} values")
    lab["name"] = lab.itemid.map(ITEMS)
    for k, (lo, hi) in RANGE.items():
        bad = (lab.name == k) & ~lab.valuenum.between(lo, hi)
        lab = lab[~bad]
    lab = lab.merge(win, left_on="subject_id", right_index=True, how="left")
    lab["in_icu"] = (lab.charttime >= lab.t0 - pd.Timedelta("6h")) & (lab.charttime <= lab.t1)

    def take(df):
        if agg == "first":
            df = df.sort_values("charttime")
            return df.groupby(["subject_id", "name"]).valuenum.first()
        return df.groupby(["subject_id", "name"]).valuenum.median()
    v = take(lab[lab.in_icu])
    vall = take(lab)
    v = v.reindex(vall.index).fillna(vall)
    w = v.unstack("name")
    for c in ITEMS.values():
        if c not in w.columns: w[c] = np.nan
    w = w.reset_index()
    w.to_csv(os.path.join(DERIVED, f"labs_{agg}.csv"), index=False)
    print(f"[labs] derived/labs_{agg}.csv   {len(w):,} patients")
    return w

def add_scores(d):
    d = d.copy()
    d["age_fib"] = d.age.clip(18, 90)
    ok = d.ast.notna() & d.alt.notna() & d.plt.notna() & d.age_fib.notna()
    d["fib4"] = np.where(ok & (d.plt > 0) & (d.alt > 0),
                         d.age_fib * d.ast / (d.plt * np.sqrt(d.alt.clip(lower=1e-9))), np.nan)
    d["apri"] = np.where(d.ast.notna() & (d.plt > 0), (d.ast/40) / d.plt * 100, np.nan)
    d["aar"]  = np.where(d.ast.notna() & (d.alt > 0), d.ast / d.alt, np.nan)
    for c in ["fib4", "apri", "ast", "alt", "plt", "bili", "inr"]:
        if c in d.columns:
            d["log_" + c] = np.log(pd.to_numeric(d[c], errors="coerce").clip(lower=1e-3))
    return d

def cv_logit(d, feats, y, fold, folds=10):
    p = np.full(len(d), np.nan)
    for f in range(folds):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or len(np.unique(y[tr])) < 2: continue
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=3000, C=0.5))
        m.fit(d.loc[tr, feats], y[tr])
        p[te] = m.predict_proba(d.loc[te, feats])[:, 1]
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="median", choices=["median", "first"])
    a = ap.parse_args()
    sids = set(pd.read_csv(os.path.join(CACHE, "index.csv"), usecols=["subject_id"]).subject_id)
    extract_labs(sids, a.agg)

if __name__ == "__main__":
    warnings.filterwarnings("ignore"); main()
