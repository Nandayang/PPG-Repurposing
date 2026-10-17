"""Input ablation on the same 7:1:2 splits and patients: age and sex, spectral pulse rate, pulse-rate variability,
pulse-contour descriptors and all 21 handcrafted features (logistic regression fitted on train + validation;
XGBoost with early stopping on validation) compared with the deep model; increments over the deep score; and
handcrafted models transferred between health systems without refitting.

Outputs (results/paper/): report_30_inputs.txt, result_30_inputs.json
"""
import os, json, argparse, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("PPG_ROOT", "."))
ap.add_argument("--feat", default=None, help="folder with the outputs of 27_morph_hrv.py (default <root>/derived)")
ap.add_argument("--out", default=None, help="default: <root>/results/paper")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--dbs", default="mimic,mcmed")
a = ap.parse_args()
a.feat = getattr(a, "feat", None) or os.path.join(a.root, "derived")
a.out = a.out or os.path.join(a.root, "results", "paper")
os.makedirs(a.out, exist_ok=True)
LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(str(s))
def head(t):
    say(); say("=" * 78); say(t); say("=" * 78)

MORPH = ["crest", "notch_pos", "notch_found", "notch_depth", "RI", "IPA", "dT", "width50",
         "b_a", "d_a", "sys_area"]
HRV = ["sdnn", "rmssd", "pnn50", "sd1", "sd2", "sd1_sd2", "ibi_cv"]
VAR = ["amp_cv", "beat_consistency"]


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
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=5000, C=0.5))


def cv_lr(D, cols):
    X = D[cols].apply(pd.to_numeric, errors="coerce"); p = np.full(len(D), np.nan)
    for k in range(5):
        tr, te = (D.test_fold != k).to_numpy(), (D.test_fold == k).to_numpy()
        m = lr().fit(X[tr], D.y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p


def cv_xgb(D, cols, seed=0):
    import xgboost as xgb
    X = D[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float); y = D.y.to_numpy()
    p = np.full(len(D), np.nan); best = []
    for k in range(5):
        role = D[f"role{k}"].to_numpy()
        tr, va, te = role == "train", role == "val", role == "test"
        spw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
        m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03, max_depth=3, min_child_weight=5,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
                              eval_metric="auc", early_stopping_rounds=100, random_state=seed + k,
                              n_jobs=2, tree_method="hist")
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        p[te] = m.predict_proba(X[te])[:, 1]; best.append(int(m.best_iteration))
    return p, best


def rank_within_fold(D, p):
    return pd.Series(p).groupby(D.test_fold.to_numpy()).rank(pct=True).to_numpy()


def load(db):
    R = a.root
    O = pd.read_csv(os.path.join(R, "results", "es", f"patient_oof_{db}_cirrhosis_ssl.csv"))
    if db == "mcmed": O["pid"] = O.CSN
    S = pd.read_csv(os.path.join(R, "derived", f"split712_{db}_cirrhosis.csv"))
    F = pd.read_csv(os.path.join(a.feat, f"morph_hrv_patient_{db}.csv")).drop(columns=["y"], errors="ignore")
    F = F.rename(columns={"hr": "hr_beat"})
    keep = ["pid", "y", "age", "male", "ppg_hr", "score", "fold"]
    D = O[keep].merge(S.drop(columns=["y"]), on="pid").merge(F, on="pid", how="left")
    if {"sd1", "sd2"} <= set(D.columns): D["sd1_sd2"] = D.sd1 / D.sd2
    assert (D.fold == D.test_fold).all()
    return D


def analyse(db, label):
    head(label)
    D = load(db); y = D.y.to_numpy().astype(int); B = boot_idx(y, a.boot)
    hrv = [c for c in HRV if c in D.columns and D[c].notna().mean() > 0.5]
    var = [c for c in VAR if c in D.columns and D[c].notna().mean() > 0.5]
    cov = D[MORPH].notna().all(axis=1).mean()
    say(f"  patients {len(D):,} ({y.sum()}+)  with contour features {cov:.1%}  HRV {hrv}  other {var}")
    SETS = [("age+sex", ["age", "male"]),
            ("pulse rate", ["ppg_hr"]),
            ("HRV", hrv),
            ("pulse rate + HRV", ["ppg_hr"] + hrv),
            ("beat consistency", ["beat_consistency"] if "beat_consistency" in D.columns else []),
            ("beat consistency + HRV", (["beat_consistency"] + hrv) if "beat_consistency" in D.columns else []),
            ("pulse contour (11)", MORPH),
            ("all handcrafted", MORPH + ["ppg_hr"] + hrv + var)]
    P = {}; R = {"sets": {}}
    say(f"\n  {'features':<34}{'logistic':>26}{'XGBoost':>28}")
    for nm, cols in SETS:
        if not cols: continue
        pl = cv_lr(D, cols); rl = auc_ci(y, pl, B); P[f"LR {nm}"] = pl
        txt = f"  {nm:<34}{rl[0]:>8.3f} [{rl[1]:.3f}, {rl[2]:.3f}]"
        R["sets"][nm] = {"lr": [float(x) for x in rl], "n_feat": len(cols)}
        if len(cols) >= 3:
            px, best = cv_xgb(D, cols); rx = auc_ci(y, px, B); P[f"XGB {nm}"] = px
            txt += f"   {rx[0]:>8.3f} [{rx[1]:.3f}, {rx[2]:.3f}]  (trees {best})"
            R["sets"][nm]["xgb"] = [float(x) for x in rx]
        say(txt)
    sc = D.score.to_numpy(); r = auc_ci(y, sc, B)
    say(f"  {'deep model':<34}{r[0]:>8.3f} [{r[1]:.3f}, {r[2]:.3f}]")
    R["deep"] = [float(x) for x in r]

    say("\n  paired differences")
    hc = "all handcrafted"
    pairs = [("deep", sc, f"LR {hc}"), ("deep", sc, f"XGB {hc}"), ("deep", sc, "LR HRV"),
             ("deep", sc, "LR pulse contour (11)")]
    R["delta"] = {}
    for n1, p1, n2 in pairs:
        if n2 not in P: continue
        d = delta(y, p1, P[n2], B); R["delta"][f"{n1} − {n2}"] = [float(x) for x in d]
        say(f"    {n1} − {n2:<40} {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]  p={d[3]:.3g}")
    for n1, n2 in [("LR pulse rate + HRV", "LR pulse rate"), ("LR beat consistency + HRV", "LR beat consistency"), (f"LR {hc}", "LR pulse contour (11)")]:
        if n1 in P and n2 in P:
            d = delta(y, P[n1], P[n2], B); R["delta"][f"{n1} − {n2}"] = [float(x) for x in d]
            say(f"    {n1} − {n2:<26} {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]  p={d[3]:.3g}")

    say("\n  increments over the deep score (logistic regression, same folds)")
    R["incr"] = {}
    for nm, cols in [("deep + HRV", ["score"] + hrv), ("deep + pulse rate + HRV", ["score", "ppg_hr"] + hrv),
                     ("deep + all handcrafted", ["score"] + MORPH + ["ppg_hr"] + hrv + var)]:
        p2 = cv_lr(D, cols); p1 = cv_lr(D, ["score"])
        d = delta(y, p2, p1, B); R["incr"][nm] = [float(x) for x in d]
        say(f"    {nm:<26} {roc_auc_score(y, p2):.3f}  vs deep {roc_auc_score(y, p1):.3f}   {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]  p={d[3]:.3g}")

    say("\n  single-feature AUC; Spearman rho with beat consistency in controls")
    R["single"] = {}
    for c in ["ppg_hr"] + hrv + var:
        m = D[c].notna(); au = roc_auc_score(y[m], D[c][m]); r_bc = spearmanr(D[c][(D.y == 0) & m], D.beat_consistency[(D.y == 0) & m]).correlation if "beat_consistency" in D.columns else np.nan
        R["single"][c] = [float(au), float(r_bc)]
        say(f"    {c:<18} AUC {au:.3f}   rho {r_bc:+.2f}")
    if db == "mcmed":
        say("\n  ED disposition strata (within-stratum AUC; pooled = weighted by case-control pairs)")
        O2 = pd.read_csv(os.path.join(a.root, "results", "es", "patient_oof_mcmed_cirrhosis_ssl.csv"))[["CSN", "ED_dispo"]]
        g = D[["pid"]].merge(O2.rename(columns={"CSN": "pid"}), on="pid", how="left").ED_dispo.astype(str).str.lower()
        D["g"] = np.where(g.str.contains("discharge"), "discharged", np.where(g.str.contains("observ"), "observation", "admitted"))
        R["dispo"] = {}
        for nm, p in [("deep", sc), ("HRV", P.get("LR HRV")), ("pulse contour", P.get("LR pulse contour (11)")),
                      ("all handcrafted", P.get(f"LR {hc}"))]:
            if p is None: continue
            txt = f"    {nm:<8}"; num = den = 0; row = {}
            for gg in ["discharged", "observation", "admitted"]:
                m = (D.g == gg).to_numpy(); yy, pp = y[m], np.asarray(p)[m]
                au = roc_auc_score(yy, pp); row[gg] = float(au); txt += f"  {gg} {au:.3f}"
                w = yy.sum() * (len(yy) - yy.sum()); num += au * w; den += w
            row["pooled"] = float(num / den); R["dispo"][nm] = row
            say(txt + f"   pooled {num / den:.3f}")
        for gg in ["discharged", "admitted"]:
            m = (D.g == gg).to_numpy(); yy = y[m]; Bg = boot_idx(yy, a.boot, 3)
            d1 = delta(yy, sc[m], P["LR HRV"][m], Bg); d2 = delta(yy, sc[m], P[f"LR {hc}"][m], Bg)
            R["dispo"][f"{gg}_delta"] = {"deep - HRV": [float(x) for x in d1], "deep - all handcrafted": [float(x) for x in d2]}
            say(f"    {gg}: deep - HRV {d1[0]:+.3f} [{d1[1]:+.3f}, {d1[2]:+.3f}]   deep - all handcrafted {d2[0]:+.3f} [{d2[1]:+.3f}, {d2[2]:+.3f}]")
        say("    median HRV-model probability in controls: " + "  ".join(
            f"{gg} {np.median(np.asarray(P['LR HRV'])[(D.g == gg).to_numpy() & (y == 0)]):.3f}" for gg in ["discharged", "observation", "admitted"]))

    say("\n  controls: Spearman rho of the deep score with pulse rate and HRV")
    c0 = D.y == 0; R["corr_controls"] = {}
    for c in ["ppg_hr"] + hrv + var:
        m = c0 & D[c].notna()
        rho = spearmanr(D.score[m], D[c][m]).correlation; R["corr_controls"][c] = float(rho)
    say("    " + "  ".join(f"{k} {v:+.2f}" for k, v in R["corr_controls"].items()))
    return R, D


warnings.filterwarnings("ignore")
RES, DD = {}, {}
for db in a.dbs.split(","):
    lab = {"mimic": "MIMIC (ICU)", "mcmed": "MC-MED (ED)"}[db]
    RES[db], DD[db] = analyse(db, lab)

if len(DD) == 2:
    head("Handcrafted logistic models fitted in one system and applied to the other")
    hrv = [c for c in HRV if all(c in d.columns and d[c].notna().mean() > 0.5 for d in DD.values())]
    var = [c for c in VAR if all(c in d.columns for d in DD.values())]
    RES["frozen"] = {}
    for nm, cols in [("pulse contour (11)", MORPH), ("HRV", hrv), ("all handcrafted", MORPH + ["ppg_hr"] + hrv + var)]:
        for s_, t_ in [("mimic", "mcmed"), ("mcmed", "mimic")]:
            S_, T_ = DD[s_], DD[t_]
            m = lr().fit(S_[cols].apply(pd.to_numeric, errors="coerce"), S_.y)
            p = m.predict_proba(T_[cols].apply(pd.to_numeric, errors="coerce"))[:, 1]
            yt = T_.y.to_numpy().astype(int); r = auc_ci(yt, p, boot_idx(yt, a.boot, 7))
            RES["frozen"][f"{nm} {s_}->{t_}"] = [float(x) for x in r]
            say(f"  {nm:<18} {s_:>5} -> {t_:<5}  {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]")

json.dump(RES, open(os.path.join(a.out, "result_30_inputs.json"), "w"), indent=1, ensure_ascii=False)
open(os.path.join(a.out, "report_30_inputs.txt"), "w", encoding="utf-8").write("\n".join(LOG))
