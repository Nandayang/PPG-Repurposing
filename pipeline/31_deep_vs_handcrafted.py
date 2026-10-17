"""Deep model vs the 21 handcrafted features of 30_input_ablation.py.

A. Increments: handcrafted + deep score vs either alone (logistic regression on the same test folds).
B. Short recordings: first N retained 30-s windows per patient (N = 1, 2, 4, 10, all); handcrafted models fitted at
   the patient level (full recordings, applied to medians of the first N windows) or to individual windows.
B'. As B, with models fitted in one system and applied to the other.
C. Frozen fusion: [deep score + handcrafted] fitted in one system, evaluated in the other.
Deep-model AUCs use within-fold ranks, handcrafted models pooled probabilities; 2,000 patient-level bootstrap resamples.

Outputs: results/paper/report_31_deep_vs_hand.txt, result_31_deep_vs_hand.json; Fig. 6b-c tables
(figS3b_combined.csv, figS3c_spotcheck.csv) in --figdata (default results/paper/31_fig_data)
"""
import os, json, argparse, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("PPG_ROOT", "."))
ap.add_argument("--out", default=None, help="default: <root>/results/paper")
ap.add_argument("--figdata", default=None, help="default: <out>/31_fig_data")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
a.out = a.out or os.path.join(a.root, "results", "paper")
a.figdata = a.figdata or os.path.join(a.out, "31_fig_data")
os.makedirs(a.out, exist_ok=True); os.makedirs(a.figdata, exist_ok=True)
LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(str(s))
def head(t):
    say(); say("=" * 78); say(t); say("=" * 78)

MORPH = ["crest", "notch_pos", "notch_found", "notch_depth", "RI", "IPA", "dT", "width50", "b_a", "d_a", "sys_area"]
HRV = ["sdnn", "rmssd", "pnn50", "sd1", "sd2", "sd1_sd2", "ibi_cv"]
VAR = ["amp_cv", "beat_consistency"]
ALLP = MORPH + ["ppg_hr"] + HRV + VAR
MED = [c for c in ALLP if c != "sd1_sd2"]
NS = [1, 2, 4, 10, 10 ** 6]


def boot_idx(y, n, seed=0):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): out.append(i)
    return out


def auc_ci(y, p, B):
    return roc_auc_score(y, p), *np.percentile([roc_auc_score(y[i], p[i]) for i in B], [2.5, 97.5])


def delta(y, p1, p2, B):
    d = np.array([roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]) for i in B])
    return (roc_auc_score(y, p1) - roc_auc_score(y, p2), *np.percentile(d, [2.5, 97.5]),
            max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(B)))


def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000, C=0.5))


def logit_rank(s):
    r = pd.Series(s).rank(pct=True).to_numpy(); r = np.clip(r, 0.5 / len(r), 1 - 0.5 / len(r))
    return np.log(r / (1 - r))


def frank(p, fold):
    return pd.Series(p).groupby(np.asarray(fold)).rank(pct=True).to_numpy()


def num(D, cols):
    return D[cols].apply(pd.to_numeric, errors="coerce")


def ratio(F):
    F = F.copy(); F["sd1_sd2"] = F.sd1 / F.sd2
    return F.replace([np.inf, -np.inf], np.nan)


def load(db):
    R = a.root
    O = pd.read_csv(os.path.join(R, "results", "es", f"patient_oof_{db}_cirrhosis_ssl.csv"))
    if db == "mcmed": O["pid"] = O.CSN
    S = pd.read_csv(os.path.join(R, "derived", f"split712_{db}_cirrhosis.csv"))
    F = pd.read_csv(os.path.join(R, "derived", f"morph_hrv_patient_{db}.csv")).drop(columns=["y"], errors="ignore")
    F = F.rename(columns={"hr": "hr_beat"})
    P = O[["pid", "y", "fold", "score", "ppg_hr"]].merge(S[["pid", "test_fold"]], on="pid").merge(
        F[["pid"] + [c for c in MORPH + HRV + VAR if c in F.columns and c != "sd1_sd2"]], on="pid", how="left")
    P = ratio(P)
    assert (P.fold == P.test_fold).all()
    Wf = pd.read_csv(os.path.join(R, "derived", f"morph_hrv_window_{db}.csv.gz"))
    Wf = Wf.drop(columns=["hr"], errors="ignore")
    if db == "mimic":
        hr = np.load(os.path.join(R, "cache", "ppg_hr.npy"))
    else:
        hr = pd.read_csv(os.path.join(R, "cache", "mcmed_index.csv"), usecols=["ppg_hr"]).ppg_hr.to_numpy()
    Wf["ppg_hr"] = hr[Wf.row.to_numpy()]
    z = np.load(os.path.join(R, "results", "spot", f"winpred_{db}.npz"))
    Wp = pd.DataFrame({"row": z["row"], "p_in": z["p_in"], "p_ex": z["p_ex"]})
    W = Wp.merge(Wf[["row", "pid"] + [c for c in MED if c != "ppg_hr"] + ["ppg_hr"]], on="row", how="left")
    miss = W.pid.isna().mean()
    W = W.dropna(subset=["pid"]).copy(); W["pid"] = W.pid.astype(P.pid.dtype)
    W = W.merge(P[["pid", "y", "test_fold"]], on="pid").sort_values(["pid", "row"])
    W["k"] = W.groupby("pid").cumcount()
    W = ratio(W)
    say(f"  {db}: patients {len(P):,} ({int(P.y.sum())}+)  windows {len(W):,} (without features {miss:.2%})")
    return P, W


def first_n(W, n, order):
    F = W[W.k < n].groupby("pid")[MED].median()
    return ratio(F).reindex(order)


def analyse(db):
    head(db)
    P, W = load(db)
    y = P.y.to_numpy().astype(int); B = boot_idx(y, a.boot)
    f = P.test_fold.to_numpy()
    R = {}

    def cv(cols):
        X = num(P, cols); p = np.full(len(P), np.nan)
        for k in range(5):
            tr, te = f != k, f == k
            p[te] = lr().fit(X[tr], P.y[tr]).predict_proba(X[te])[:, 1]
        return p

    p_h = cv(ALLP); p_hd = cv(ALLP + ["score"]); p_d = P.score.to_numpy()
    say("\n  A. Increments (logistic regression, same test folds)")
    rh, rd, rhd = auc_ci(y, p_h, B), auc_ci(y, p_d, B), auc_ci(y, p_hd, B)
    for nm, r in [("handcrafted", rh), ("deep", rd), ("handcrafted + deep", rhd)]:
        say(f"    {nm:<20} {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]")
    d1 = delta(y, p_hd, p_h, B); d2 = delta(y, p_hd, p_d, B); d0 = delta(y, p_d, p_h, B)
    say(f"    deep - handcrafted     {d0[0]:+.3f} [{d0[1]:+.3f}, {d0[2]:+.3f}]  p={d0[3]:.3g}")
    say(f"    both - handcrafted     {d1[0]:+.3f} [{d1[1]:+.3f}, {d1[2]:+.3f}]  p={d1[3]:.3g}")
    say(f"    both - deep            {d2[0]:+.3f} [{d2[1]:+.3f}, {d2[2]:+.3f}]  p={d2[3]:.3g}")
    R["A"] = dict(hand=float(rh[0]), deep=float(rd[0]), both=float(rhd[0]),
                  deep_minus_hand=[float(x) for x in d0],
                  both_minus_hand=[float(x) for x in d1], both_minus_deep=[float(x) for x in d2])

    say("\n  B. First N windows per patient (internal)")
    say(f"    {'N':>5}  {'deep':>22}  {'patient-level':>22}  {'window-level':>22}  {'deep - patient':>24}  {'deep - window':>24}")
    Xp = num(P, ALLP)
    models_p = {k: lr().fit(Xp[f != k], P.y[f != k]) for k in range(5)}
    WW = ["ppg_hr"] + [c for c in ALLP if c != "ppg_hr"]
    models_w = {k: lr().fit(W.loc[W.test_fold != k, WW], W.loc[W.test_fold != k, "y"]) for k in range(5)}
    W["pw"] = np.nan
    for k in range(5):
        te = (W.test_fold == k).to_numpy(); W.loc[te, "pw"] = models_w[k].predict_proba(W.loc[te, WW])[:, 1]
    R["B"] = []
    for n in NS:
        sub = W[W.k < n]
        pw = sub.groupby("pid").pw.mean().reindex(P.pid).to_numpy()
        if n > 1000:
            deep, pp = p_d, p_h
        else:
            deep = frank(sub.groupby("pid").p_in.mean().reindex(P.pid).to_numpy(), f)
            Fn = first_n(W, n, P.pid)
            pp = np.full(len(P), np.nan)
            for k in range(5):
                m = f == k; pp[m] = models_p[k].predict_proba(num(Fn.iloc[np.flatnonzero(m)].reset_index(drop=True), ALLP))[:, 1]
        assert np.isfinite(deep).all() and np.isfinite(pp).all() and np.isfinite(pw).all()
        rd_, rp, rw = auc_ci(y, deep, B), auc_ci(y, pp, B), auc_ci(y, pw, B)
        dp, dw = delta(y, deep, pp, B), delta(y, deep, pw, B)
        lab = "all" if n > 1000 else str(n)
        say(f"    {lab:>5}  {rd_[0]:.3f} [{rd_[1]:.3f}, {rd_[2]:.3f}]  {rp[0]:.3f} [{rp[1]:.3f}, {rp[2]:.3f}]  "
            f"{rw[0]:.3f} [{rw[1]:.3f}, {rw[2]:.3f}]  {dp[0]:+.3f} [{dp[1]:+.3f}, {dp[2]:+.3f}]  {dw[0]:+.3f} [{dw[1]:+.3f}, {dw[2]:+.3f}]")
        R["B"].append(dict(n=lab, deep=[float(x) for x in rd_], hand_patient=[float(x) for x in rp],
                           hand_window=[float(x) for x in rw], d_patient=[float(x) for x in dp], d_window=[float(x) for x in dw]))
    return R, P, W


warnings.filterwarnings("ignore")
RES, PP, WW_ = {}, {}, {}
for db in ["mimic", "mcmed"]:
    RES[db], PP[db], WW_[db] = analyse(db)

head("B'. First N windows, models from the other system; C. frozen fusion")
OTH = {"mimic": "mcmed", "mcmed": "mimic"}
WCOLS = ["ppg_hr"] + [c for c in ALLP if c != "ppg_hr"]
for src in ["mimic", "mcmed"]:
    tgt = OTH[src]; Ps, Ws, Pt, Wt = PP[src], WW_[src], PP[tgt], WW_[tgt].copy()
    say(f"\n  {src} -> {tgt}")
    mp = lr().fit(num(Ps, ALLP), Ps.y)
    mw = lr().fit(Ws[WCOLS], Ws.y)
    Wt["pw"] = mw.predict_proba(Wt[WCOLS])[:, 1]
    yy = Pt.y.to_numpy().astype(int); Bn = boot_idx(yy, a.boot, 2)
    deep_all = Wt.groupby("pid").p_ex.mean().reindex(Pt.pid).to_numpy()
    rows = []
    for n in NS:
        sub = Wt[Wt.k < n]
        deep = sub.groupby("pid").p_ex.mean().reindex(Pt.pid).to_numpy()
        pw = sub.groupby("pid").pw.mean().reindex(Pt.pid).to_numpy()
        Fn = num(Pt, ALLP) if n > 1000 else num(first_n(Wt, n, Pt.pid).reset_index(drop=True), ALLP)
        pp = mp.predict_proba(Fn)[:, 1]
        rd, rp, rw = (auc_ci(yy, p, Bn) for p in (deep, pp, pw))
        dp = delta(yy, deep, pp, Bn); dw = delta(yy, deep, pw, Bn)
        lab = "all" if n > 1000 else str(n)
        say(f"    N={lab:>4}  deep {rd[0]:.3f} [{rd[1]:.3f}, {rd[2]:.3f}]  patient-level {rp[0]:.3f}  window-level {rw[0]:.3f}"
            f"   deep - patient {dp[0]:+.3f} [{dp[1]:+.3f}, {dp[2]:+.3f}]  deep - window {dw[0]:+.3f} [{dw[1]:+.3f}, {dw[2]:+.3f}]")
        rows.append(dict(n=lab, deep=[float(x) for x in rd], hand_patient=[float(x) for x in rp],
                         hand_window=[float(x) for x in rw], d_patient=[float(x) for x in dp], d_window=[float(x) for x in dw]))
    RES[f"Bext_{src}_{tgt}"] = rows
    Ps2 = Ps.copy(); Ps2["dl"] = logit_rank(Ps2.score)
    mf = lr().fit(num(Ps2, ALLP + ["dl"]), Ps2.y)
    Xt = num(Pt, ALLP).copy(); Xt["dl"] = logit_rank(deep_all)
    pf = mf.predict_proba(Xt[ALLP + ["dl"]])[:, 1]
    ph = mp.predict_proba(num(Pt, ALLP))[:, 1]
    Bc = boot_idx(yy, a.boot, 3)
    rf, rd, rh = auc_ci(yy, pf, Bc), auc_ci(yy, deep_all, Bc), auc_ci(yy, ph, Bc)
    d1 = delta(yy, pf, deep_all, Bc); d2 = delta(yy, pf, ph, Bc)
    say(f"    C. fusion {rf[0]:.3f} [{rf[1]:.3f}, {rf[2]:.3f}]   deep {rd[0]:.3f}   handcrafted {rh[0]:.3f}"
        f"   fusion - deep {d1[0]:+.3f} [{d1[1]:+.3f}, {d1[2]:+.3f}]  fusion - handcrafted {d2[0]:+.3f} [{d2[1]:+.3f}, {d2[2]:+.3f}]")
    RES[f"C_{src}_{tgt}"] = dict(fusion=[float(x) for x in rf], deep=[float(x) for x in rd], hand=[float(x) for x in rh],
                                 fusion_minus_deep=[float(x) for x in d1], fusion_minus_hand=[float(x) for x in d2])

json.dump(RES, open(os.path.join(a.out, "result_31_deep_vs_hand.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(os.path.join(a.out, "report_31_deep_vs_hand.txt"), "w", encoding="utf-8").write("\n".join(LOG))

rows = []
for cohort, key in (("MIMIC", "mimic"), ("MC-MED", "mcmed")):
    A = RES[key]["A"]
    for model, v in (("Handcrafted", A["hand"]), ("Deep model", A["deep"]), ("Deep + handcrafted", A["both"])):
        rows.append(dict(cohort=cohort, kind="int", row="auc", model=model, value=v, lo=None, hi=None, p=None))
    for what, v in (("Deep model added", A["both_minus_hand"]), ("Handcrafted added", A["both_minus_deep"])):
        rows.append(dict(cohort=cohort, kind="int", row="gain", model=what, value=v[0], lo=v[1], hi=v[2], p=v[3]))
for cohort, key in (("MC-MED", "C_mimic_mcmed"), ("MIMIC", "C_mcmed_mimic")):
    C = RES[key]
    for model, v in (("Handcrafted", C["hand"]), ("Deep model", C["deep"]), ("Deep + handcrafted", C["fusion"])):
        rows.append(dict(cohort=cohort, kind="ext", row="auc", model=model, value=v[0], lo=v[1], hi=v[2], p=None))
    for what, v in (("Deep model added", C["fusion_minus_hand"]), ("Handcrafted added", C["fusion_minus_deep"])):
        rows.append(dict(cohort=cohort, kind="ext", row="gain", model=what, value=v[0], lo=v[1], hi=v[2], p=v[3]))
pd.DataFrame(rows).to_csv(os.path.join(a.figdata, "figS3b_combined.csv"), index=False, encoding="utf-8-sig")
rows = []
for cohort, key in (("MIMIC", "mimic"), ("MC-MED", "mcmed")):
    for order, r in enumerate(RES[key]["B"]):
        for model, k in (("Deep model", "deep"), ("Handcrafted (patient level)", "hand_patient"),
                         ("Handcrafted (window level)", "hand_window")):
            rows.append(dict(cohort=cohort, n=r["n"], order=order, model=model, auc=r[k][0], lo=r[k][1], hi=r[k][2]))
pd.DataFrame(rows).to_csv(os.path.join(a.figdata, "figS3c_spotcheck.csv"), index=False, encoding="utf-8")
print(f"\n[saved] {a.out}, {a.figdata}")
