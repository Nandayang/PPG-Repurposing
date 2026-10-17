"""Patient-level predictions of the four settings and helper functions used by figures/builders/make_plot_data.py.
"""
import os, json, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT    = os.environ.get("PPG_ROOT", ".")
DERIVED = os.path.join(ROOT, "derived")
RES     = os.path.join(ROOT, "results")
BOOT    = 1000


def auc_ci(y, s, n=None, seed=0):
    n = n or BOOT
    y, s = np.asarray(y).astype(int), np.asarray(s, float); ok = np.isfinite(s); y, s = y[ok], s[ok]
    a = roc_auc_score(y, s); rng = np.random.default_rng(seed); b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): b.append(roc_auc_score(y[i], s[i]))
    return a, np.percentile(b, 2.5), np.percentile(b, 97.5)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def fold_baseline(D, cols, fold_col="fold"):
    cols = [c for c in cols if c in D.columns and D[c].notna().any()]
    X = D[cols].apply(pd.to_numeric, errors="coerce"); p = np.full(len(D), np.nan)
    for k in sorted(D[fold_col].dropna().unique()):
        tr, te = (D[fold_col] != k).to_numpy(), (D[fold_col] == k).to_numpy()
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
        m.fit(X[tr], D.y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p


def logistic_mle(X, y, it=100):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        H = (X * W[:, None]).T @ X + 1e-8 * np.eye(X.shape[1]); st = np.linalg.solve(H, X.T @ (y - p)); b += st
        if np.abs(st).max() < 1e-8: break
    return b, np.linalg.inv(H)


def tb(s):
    return s.astype(str).isin(["True", "1", "1.0"])


def load_sets():
    C = pd.read_csv(os.path.join(DERIVED, "cohort_patients.csv"),
                    usecols=["subject_id", "lbl_cirr_decomp", "lbl_cirr_comp"]).rename(columns={"subject_id": "pid"})
    mi = pd.read_csv(os.path.join(RES, "es", "patient_oof_mimic_cirrhosis_ssl.csv")).merge(C, on="pid", how="left")
    mi["dec"] = mi.lbl_cirr_decomp.fillna(0).astype(int); mi["com"] = mi.lbl_cirr_comp.fillna(0).astype(int)
    mc = pd.read_csv(os.path.join(RES, "es", "patient_oof_mcmed_cirrhosis_ssl.csv")); mc["pid"] = mc.CSN
    mc["dec"] = tb(mc.dx_decomp).astype(int); mc["com"] = tb(mc.dx_comp).astype(int)
    rv = pd.read_csv(os.path.join(RES, "reverse", "patient_preds_es_mcmed_cirrhosis_ssl.csv")).drop(columns=["dec", "com"], errors="ignore").merge(C, on="pid", how="left")
    rv["dec"] = rv.lbl_cirr_decomp.fillna(0).astype(int); rv["com"] = rv.lbl_cirr_comp.fillna(0).astype(int)
    fw = pd.read_csv(os.path.join(RES, "mcmed", "patient_preds.csv"))
    fw = fw[(fw.n_win.fillna(0) >= 10) & ((fw.in_cirrhosis == 1) | (fw.in_control == 1))].copy()
    fw["y"] = fw.in_cirrhosis.astype(int); fw["score"] = fw.s_es; fw["pid"] = fw.CSN
    fw["age"] = fw.Age; fw["male"] = (fw.Gender == "M").astype(int)
    fw["dec"] = tb(fw.dx_decomp).astype(int); fw["com"] = tb(fw.dx_comp).astype(int)
    return {
        ("MIMIC", "int"): mi, ("MC-MED", "int"): mc, ("MIMIC", "ext"): rv, ("MC-MED", "ext"): fw.dropna(subset=["score"])}


NAME = {("MIMIC", "int"): "MIMIC internal", ("MC-MED", "int"): "MC-MED internal",
        ("MIMIC", "ext"): "MC-MED → MIMIC (frozen)", ("MC-MED", "ext"): "MIMIC → MC-MED (frozen)"}


def load_spot():
    R = {}
    for f in [os.path.join(RES, "spot", "result_22.json")] + sorted(glob.glob(os.path.join(RES, "spot", "result_22_*.json"))):
        if os.path.exists(f):
            R.update(json.load(open(f, encoding="utf-8")))
    return R


def meld(L):
    b = L.bili.clip(lower=1); i = L.inr.clip(lower=1); c = L.creat.clip(lower=1, upper=4)
    return (3.78 * np.log(b) + 11.2 * np.log(i) + 9.57 * np.log(c) + 6.43).round(0)
