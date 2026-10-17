"""Gain in AUC from N randomly drawn over the first N retained 30-s windows, with paired patient-level bootstrap CIs
(2,000 resamples; same window predictions and random draws as part A of 22_emulated_spotcheck.py).

Usage: python 22c_sampling_gains.py [PPG_ROOT] [out_dir]
Output: fig4b_sampling_gain_ci.csv
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import rankdata

R = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PPG_ROOT", ".")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "results", "paper")
NS = [1, 2, 4, 6, 10, 20, 50]
NAME = {"mimic": "MIMIC", "mcmed": "MC-MED"}
B, SEED = 2000, 0


def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def windows(db):
    z = np.load(os.path.join(R, "results", "spot", f"winpred_{db}.npz"))
    W = pd.DataFrame(dict(row=z["row"], p_in=z["p_in"], p_ex=z["p_ex"]))
    m = pd.read_csv(os.path.join(R, "derived", f"morph_hrv_window_{db}.csv.gz"), usecols=["row", "pid"])
    W = W.merge(m, on="row", how="left", sort=False)
    miss = W.pid.isna()
    W["pid"] = W.pid.ffill()
    assert (W.pid.bfill() == W.pid)[miss].all(), "missing window lies between two patients"
    W["pid"] = W.pid.astype(np.int64)
    assert (np.diff(W.pid.to_numpy()) >= 0).all() and W.groupby("pid").row.apply(lambda r: r.is_monotonic_increasing).all()
    O = pd.read_csv(os.path.join(R, "results", "es", f"patient_oof_{db}_cirrhosis_ssl.csv"))
    if db == "mcmed": O["pid"] = O.CSN
    S = pd.read_csv(os.path.join(R, "derived", f"split712_{db}_cirrhosis.csv"))[["pid", "test_fold"]]
    W = W.merge(O[["pid", "y"]], on="pid", how="left").merge(S, on="pid", how="left").rename(columns={"test_fold": "fold"})
    assert W.y.notna().all() and W.fold.notna().all()
    W["k"] = W.groupby("pid").cumcount()
    return W, int(miss.sum())


def patient_scores(sel, col, internal):
    g = sel.groupby("pid").agg(y=("y", "first"), fold=("fold", "first"), s=(col, "mean"))
    if internal:
        g["s"] = g.groupby("fold").s.rank(pct=True)
    return g


rows = []
for db in ["mimic", "mcmed"]:
    W, nmiss = windows(db)
    pids = np.sort(W.pid.unique()); P = len(pids)
    u = [np.random.default_rng(sd).random(len(W)) for sd in range(3)]
    rk = [W.assign(_u=x).groupby("pid")._u.rank(method="first").to_numpy() for x in u]
    rng = np.random.default_rng(SEED); boot = [rng.integers(0, P, P) for _ in range(B)]
    for col, kind in (("p_in", "int"), ("p_ex", "ext")):
        internal = kind == "int"
        for n in NS:
            gf = patient_scores(W[W.k < n], col, internal).reindex(pids)
            gr = [patient_scores(W[r <= n], col, internal).reindex(pids) for r in rk]
            y = gf.y.to_numpy().astype(int); sf = gf.s.to_numpy(); sr = [g.s.to_numpy() for g in gr]
            a_f = auc(y, sf); a_r = np.mean([auc(y, s) for s in sr])
            d = []
            for i in boot:
                yi = y[i]
                if 0 < yi.sum() < len(yi):
                    d.append(np.mean([auc(yi, s[i]) for s in sr]) - auc(yi, sf[i]))
            d = np.asarray(d)
            lo, hi = np.percentile(d, [2.5, 97.5]); p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            rows.append(dict(test=NAME[db], kind=kind, n_windows=n, minutes=n * 0.5, auc_first=a_f, auc_random=a_r,
                             gain=a_r - a_f, lo=lo, hi=hi, p_boot=p, n_boot=len(d), n_patients=P))
            print(f"{NAME[db]:7s} {kind} N={n:>2}  first {a_f:.6f}  random {a_r:.6f}  gain {a_r - a_f:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}", flush=True)
    print(f"  ({db}: {nmiss} window(s) without morphology row, pid taken from neighbours)")
os.makedirs(OUT, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "fig4b_sampling_gain_ci.csv"), index=False, encoding="utf-8-sig")
