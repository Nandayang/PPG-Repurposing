"""FIB-4 as clinical reference: PPG score vs FIB-4 and platelets in patients with FIB-4 available, by age group,
age interaction, FIB-4 zones (low < 1.3, or < 2.0 at age >= 65; high > 2.67), FIB-4-or-PPG referral rules, and
increments (delta AUC, continuous NRI, IDI). Combined models are cross-validated on the folds of 16_train_es.py;
2,000 patient-level bootstrap resamples.

Outputs (results/paper/): report_28_fib4.txt, result_28_fib4.json
"""
import os, json, argparse, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("PPG_ROOT", "."))
ap.add_argument("--out", default=None, help="default: <root>/results/paper")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
a.feat = getattr(a, "feat", None) or os.path.join(a.root, "derived")
a.out = a.out or os.path.join(a.root, "results", "paper")
ROOT, OUT, BOOT = a.root, a.out, a.boot
os.makedirs(OUT, exist_ok=True)
EPS = 1e-6
LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(str(s))
def head(t):
    say(); say("=" * 78); say(t); say("=" * 78)
def sub(t):
    say(); say("-" * 78); say(" " + t); say("-" * 78)


def add_scores(d):
    d = d.copy()
    d["age_fib"] = d.age.clip(18, 90)
    ok = d.ast.notna() & d.alt.notna() & d.plt.notna() & d.age_fib.notna()
    d["fib4"] = np.where(ok & (d.plt > 0) & (d.alt > 0),
                         d.age_fib * d.ast / (d.plt * np.sqrt(d.alt.clip(lower=1e-9))), np.nan)
    d["apri"] = np.where(d.ast.notna() & (d.plt > 0), (d.ast / 40) / d.plt * 100, np.nan)
    for c in ["fib4", "apri", "ast", "alt", "plt", "bili", "inr"]:
        if c in d.columns:
            d["log_" + c] = np.log(pd.to_numeric(d[c], errors="coerce").clip(lower=1e-3))
    return d


def cv_logit(d, feats, y, fold, folds=5):
    p = np.full(len(d), np.nan)
    for f in range(folds):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or len(np.unique(y[tr])) < 2: continue
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=3000, C=0.5))
        m.fit(d.loc[tr, feats], y[tr]); p[te] = m.predict_proba(d.loc[te, feats])[:, 1]
    return p


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); return np.log(p / (1 - p))


def boot_idx(y, n, seed=0):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): out.append(i)
    return out


def auc_ci(y, p, B):
    y, p = np.asarray(y), np.asarray(p)
    a0 = roc_auc_score(y, p)
    bs = [roc_auc_score(y[i], p[i]) for i in B]
    return a0, *np.percentile(bs, [2.5, 97.5])


def delta_ci(y, p1, p2, B):
    y = np.asarray(y)
    d0 = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    d = np.array([roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]) for i in B])
    pv = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(B))
    return d0, *np.percentile(d, [2.5, 97.5]), pv


def nri_idi(y, p_old, p_new):
    y = np.asarray(y).astype(bool)
    up, dn = p_new > p_old, p_new < p_old
    nri_e = up[y].mean() - dn[y].mean(); nri_ne = dn[~y].mean() - up[~y].mean()
    idi = (p_new[y].mean() - p_old[y].mean()) - (p_new[~y].mean() - p_old[~y].mean())
    return nri_e + nri_ne, nri_e, nri_ne, idi


def incr_block(name, y, p_old, p_new, B):
    y = np.asarray(y); p_old, p_new = np.asarray(p_old), np.asarray(p_new)
    dA = delta_ci(y, p_new, p_old, B)
    r0 = nri_idi(y, p_old, p_new)
    bs = np.array([nri_idi(y[i], p_old[i], p_new[i]) for i in B])
    lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
    say(f"  {name}")
    say(f"    AUC {roc_auc_score(y, p_old):.3f} -> {roc_auc_score(y, p_new):.3f}   "
        f"ΔAUC {dA[0]:+.3f} [{dA[1]:+.3f}, {dA[2]:+.3f}]  p={dA[3]:.3g}")
    say(f"    continuous NRI {r0[0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}]"
        f" (events {r0[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}], non-events {r0[2]:+.3f} [{lo[2]:+.3f}, {hi[2]:+.3f}])")
    say(f"    IDI {r0[3]:+.3f} [{lo[3]:+.3f}, {hi[3]:+.3f}]")
    return dict(auc_old=float(roc_auc_score(y, p_old)), auc_new=float(roc_auc_score(y, p_new)),
                dauc=[float(x) for x in dA], nri=[float(r0[0]), float(lo[0]), float(hi[0])],
                nri_events=[float(r0[1]), float(lo[1]), float(hi[1])],
                nri_nonevents=[float(r0[2]), float(lo[2]), float(hi[2])],
                idi=[float(r0[3]), float(lo[3]), float(hi[3])])


def baseline(D, cols, folds=5):
    X = D[cols].apply(pd.to_numeric, errors="coerce"); p = np.full(len(D), np.nan)
    for k in range(folds):
        tr, te = (D.test_fold != k).to_numpy(), (D.test_fold == k).to_numpy()
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=3000, C=0.5))
        m.fit(X[tr], D.y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p


def load(db):
    O = pd.read_csv(os.path.join(ROOT, "results", "es", f"patient_oof_{db}_cirrhosis_ssl.csv"))
    if db == "mcmed":
        O["pid"] = O.CSN
        L = pd.read_csv(os.path.join(ROOT, "derived", "mcmed_labs.csv")).rename(columns={"CSN": "pid"})
    else:
        L = pd.read_csv(os.path.join(ROOT, "derived", "labs_median.csv")).rename(columns={"subject_id": "pid"})
    S = pd.read_csv(os.path.join(ROOT, "derived", f"split712_{db}_cirrhosis.csv"))[["pid", "test_fold"]]
    d = O.merge(L[[c for c in L.columns if c in ["pid", "alb", "alt", "ast", "bili", "inr", "plt"]]],
                on="pid", how="left").merge(S, on="pid", how="left")
    assert (d.fold == d.test_fold).all(), "OOF folds do not match split712"
    d["age"] = d.age.clip(upper=90)
    d = add_scores(d).reset_index(drop=True)
    r = np.clip(d.score.to_numpy(), 0.5 / len(d), 1 - 0.5 / len(d))
    d["ppg_l"] = logit(r)
    return d


AGE_G = [("<45", 0, 45), ("45–49", 45, 50), ("50–59", 50, 60), ("60–69", 60, 70), ("≥70", 70, 200)]


def fib4_zone(d):
    low = np.where(d.age >= 65, 2.0, 1.3)
    z = np.where(d.fib4 < low, "low", np.where(d.fib4 > 2.67, "high", "indeterminate"))
    return pd.Series(np.where(d.fib4.isna(), "na", z), index=d.index)


def analyse(db, label):
    R = {}
    head(f"{label}")
    d = load(db)
    say(f"  all {len(d):,} ({int(d.y.sum())}+)  PPG AUC {roc_auc_score(d.y, d.score):.3f}")
    s = d[d.fib4.notna()].reset_index(drop=True)
    y, f = s.y.to_numpy().astype(int), s.fold.to_numpy()
    s["p_fib4"] = cv_logit(s, ["log_fib4"], y, f)
    s["p_fus"] = cv_logit(s, ["ppg_l", "log_fib4"], y, f)
    s["p_plt"] = cv_logit(s, ["log_plt"], y, f)
    s["p_fusplt"] = cv_logit(s, ["ppg_l", "log_plt"], y, f)
    s["p_ppg"] = cv_logit(s, ["ppg_l"], y, f)
    B = boot_idx(y, BOOT)

    sub(f"A. Patients with FIB-4: {len(s):,} ({y.sum()}+ / {len(s)-y.sum()}-)")
    M = {"PPG": s.ppg_l.to_numpy(), "FIB-4": s.fib4.to_numpy(), "platelets": -s.plt.to_numpy(),
         "PPG+FIB-4": s.p_fus.to_numpy(), "PPG+platelets": s.p_fusplt.to_numpy()}
    R["overall"] = {}
    for k, v in M.items():
        r = auc_ci(y, v, B); R["overall"][k] = [float(x) for x in r]
        say(f"    {k:<14}{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]")
    for x, w in [("PPG", "FIB-4"), ("PPG+FIB-4", "FIB-4")]:
        dd = delta_ci(y, M[x], M[w], B); R["overall"][f"{x} − {w}"] = [float(t) for t in dd]
        say(f"    {x} − {w}: {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]  p={dd[3]:.3g}")

    sub("B. By age")
    say(f"  {'age':<7}{'pos/neg':>10}  {'PPG':>20}  {'FIB-4':>20}  {'platelets':>20}  {'PPG+FIB-4':>20}"
        f"  {'PPG - FIB-4':>24}  {'combined - FIB-4':>24}")
    R["age"] = []
    for g, lo, hi in AGE_G:
        m = ((s.age >= lo) & (s.age < hi)).to_numpy()
        yy = y[m]
        if yy.sum() < 10 or (len(yy) - yy.sum()) < 10:
            say(f"  {g:<7} too few"); continue
        Bg = boot_idx(yy, BOOT, seed=1)
        row = dict(group=g, n_pos=int(yy.sum()), n_neg=int(len(yy) - yy.sum()))
        txt = f"  {g:<7}{yy.sum():>4}/{len(yy)-yy.sum():<5}"
        for k in ["PPG", "FIB-4", "platelets", "PPG+FIB-4"]:
            r = auc_ci(yy, M[k][m], Bg); row[k] = [float(x) for x in r]
            txt += f"  {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]"
        for x, w in [("PPG", "FIB-4"), ("PPG+FIB-4", "FIB-4")]:
            dd = delta_ci(yy, M[x][m], M[w][m], Bg); row[f"{x} − {w}"] = [float(t) for t in dd]
            txt += f"  {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]"
        say(txt); R["age"].append(row)
    for g, lo, hi in [("<60", 0, 60), ("≥60", 60, 200)]:
        m = ((s.age >= lo) & (s.age < hi)).to_numpy(); yy = y[m]; Bg = boot_idx(yy, BOOT, seed=2)
        dd = delta_ci(yy, M["PPG"][m], M["FIB-4"][m], Bg)
        say(f"  {g:<7}{yy.sum():>4}/{len(yy)-yy.sum():<5}  PPG {roc_auc_score(yy, M['PPG'][m]):.3f}  "
            f"FIB-4 {roc_auc_score(yy, M['FIB-4'][m]):.3f}  PPG − FIB-4 {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]")
        R[f"age_{g}"] = dict(n_pos=int(yy.sum()), n_neg=int(len(yy) - yy.sum()),
                             ppg=float(roc_auc_score(yy, M['PPG'][m])), fib4=float(roc_auc_score(yy, M['FIB-4'][m])),
                             delta=[float(t) for t in dd])
    def dgrp(lo, hi, seed):
        m = ((s.age >= lo) & (s.age < hi)).to_numpy(); yy = y[m]
        Bg = boot_idx(yy, BOOT, seed=seed)
        a0 = roc_auc_score(yy, M["PPG"][m]) - roc_auc_score(yy, M["FIB-4"][m])
        bs = np.array([roc_auc_score(yy[i], M["PPG"][m][i]) - roc_auc_score(yy[i], M["FIB-4"][m][i]) for i in Bg])
        return a0, bs
    a_y, b_y = dgrp(0, 50, 11); a_o, b_o = dgrp(70, 200, 12)
    dd = b_y - b_o
    pv = max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1 / len(dd))
    say(f"  (PPG - FIB-4), age < 50 minus age >= 70: {a_y - a_o:+.3f} [{np.percentile(dd, 2.5):+.3f}, "
        f"{np.percentile(dd, 97.5):+.3f}]  p={pv:.3g}")
    R["delta_young_minus_old"] = [float(a_y - a_o), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)), float(pv)]

    sub("C. Age interaction (logistic regression; standardized markers; age centred at 60, per 10 years)")
    t = s.copy()
    t["age10"] = (t.age - 60) / 10
    t["z_ppg"] = (t.ppg_l - t.ppg_l.mean()) / t.ppg_l.std()
    t["z_fib4"] = (t.log_fib4 - t.log_fib4.mean()) / t.log_fib4.std()
    t["z_plt"] = -(t.log_plt - t.log_plt.mean()) / t.log_plt.std()
    R["interaction"] = {}
    for nm, z in [("PPG", "z_ppg"), ("FIB-4", "z_fib4"), ("platelets", "z_plt")]:
        X = sm.add_constant(pd.DataFrame({"z": t[z], "age10": t.age10, "male": t.male, "zx": t[z] * t.age10}))
        fit = sm.Logit(y, X).fit(disp=0)
        b, ci = fit.params, fit.conf_int()
        orr = np.exp(b["zx"]); lo_, hi_ = np.exp(ci.loc["zx"])
        or40 = np.exp(b["z"] + b["zx"] * (-2)); or70 = np.exp(b["z"] + b["zx"] * 1)
        say(f"  {nm:<14}ratio of OR per SD per 10 years {orr:.2f} [{lo_:.2f}, {hi_:.2f}]  p={fit.pvalues['zx']:.2g}"
            f"   OR per SD: age 40 {or40:.2f} / 60 {np.exp(b['z']):.2f} / 70 {or70:.2f}")
        R["interaction"][nm] = dict(ratio=float(orr), ci=[float(lo_), float(hi_)], p=float(fit.pvalues["zx"]),
                                    or40=float(or40), or60=float(np.exp(b["z"])), or70=float(or70))
    X = sm.add_constant(pd.DataFrame({"z_ppg": t.z_ppg, "z_fib4": t.z_fib4, "age10": t.age10, "male": t.male,
                                      "ppg_x_age": t.z_ppg * t.age10, "fib4_x_age": t.z_fib4 * t.age10}))
    fit = sm.Logit(y, X).fit(disp=0); ci = fit.conf_int()
    say("  joint model (PPG and FIB-4):")
    for k in ["z_ppg", "z_fib4", "ppg_x_age", "fib4_x_age"]:
        say(f"    {k:<11} OR {np.exp(fit.params[k]):.2f} [{np.exp(ci.loc[k, 0]):.2f}, {np.exp(ci.loc[k, 1]):.2f}]  p={fit.pvalues[k]:.2g}")
    R["interaction"]["joint"] = {k: [float(np.exp(fit.params[k])), float(np.exp(ci.loc[k, 0])),
                                     float(np.exp(ci.loc[k, 1])), float(fit.pvalues[k])]
                                 for k in ["z_ppg", "z_fib4", "ppg_x_age", "fib4_x_age"]}
    from scipy.stats import spearmanr
    for nm, col in [("PPG", "ppg_l"), ("FIB-4", "fib4"), ("platelets", "plt")]:
        rc = spearmanr(t.age[y == 0], t[col][y == 0]).correlation
        rp = spearmanr(t.age[y == 1], t[col][y == 1]).correlation
        say(f"    {nm:<10} vs age, Spearman: controls {rc:+.2f}  cases {rp:+.2f}")

    sub("D. FIB-4 zones")
    s["zone"] = fib4_zone(s)
    R["zone"] = {}
    say(f"  {'zone':<15}{'cases':>8}{'ctrls':>8}{'% cases':>10}   PPG AUC                  FIB-4 AUC")
    for z in ["low", "indeterminate", "high"]:
        m = (s.zone == z).to_numpy(); yy = y[m]
        txt = f"  {z:<15}{yy.sum():>8}{len(yy)-yy.sum():>8}{yy.mean():>10.1%}"
        row = dict(n_pos=int(yy.sum()), n_neg=int(len(yy) - yy.sum()))
        if yy.sum() >= 10 and (len(yy) - yy.sum()) >= 10:
            Bz = boot_idx(yy, BOOT, seed=3)
            r1 = auc_ci(yy, M["PPG"][m], Bz); r2 = auc_ci(yy, M["FIB-4"][m], Bz)
            txt += f"   {r1[0]:.3f} [{r1[1]:.3f}, {r1[2]:.3f}]   {r2[0]:.3f} [{r2[1]:.3f}, {r2[2]:.3f}]"
            row["ppg"] = [float(x) for x in r1]; row["fib4"] = [float(x) for x in r2]
        say(txt); R["zone"][z] = row
    yb = y.astype(bool)
    sens = (s.zone[yb] != "low").mean(); spec = (s.zone[~yb] == "low").mean()
    say(f"  FIB-4 above the low cut-off: sensitivity {sens:.3f}  specificity {spec:.3f}; cases in the low zone {(s.zone[yb]=='low').sum()}")
    for g, lo, hi in AGE_G:
        m = ((s.age >= lo) & (s.age < hi)).to_numpy() & yb
        if m.sum():
            say(f"    {g:<6} cases {m.sum():>4}: low {(s.zone[m]=='low').mean():.1%}  indeterminate {(s.zone[m]=='indeterminate').mean():.1%}  high {(s.zone[m]=='high').mean():.1%}")
    R["fib4_rule"] = dict(sens=float(sens), spec=float(spec))

    sub("F. Referral rules: FIB-4 above the low cut-off, or PPG above a control quantile (exploratory)")
    ctl = s.ppg_l[~yb]
    R["rule"] = {}
    for spec_t in [0.90, 0.95]:
        t_ = np.quantile(ctl, spec_t)
        say(f"  PPG threshold: {spec_t:.0%} quantile of controls")
        say(f"  {'age':<7}{'pos/neg':>10}  {'FIB-4 sens/spec':>20}  {'PPG sens/spec':>18}  "
            f"{'FIB-4 or PPG':>18}  {'low-zone cases flagged by PPG':>22}")
        rows = []
        for g, lo, hi in [("all", 0, 200)] + AGE_G:
            m = ((s.age >= lo) & (s.age < hi)).to_numpy()
            fz = (s.zone != "low").to_numpy(); pz = (s.ppg_l >= t_).to_numpy(); both = fz | pz
            yy = yb[m]
            r_ = dict(group=g, n_pos=int(yy.sum()), n_neg=int((~yy).sum()),
                      fib4=[float(fz[m][yy].mean()), float((~fz[m])[~yy].mean())],
                      ppg=[float(pz[m][yy].mean()), float((~pz[m])[~yy].mean())],
                      either=[float(both[m][yy].mean()), float((~both[m])[~yy].mean())],
                      rescued=[int((pz & ~fz)[m][yy].sum()), int((~fz)[m][yy].sum())])
            rows.append(r_)
            say(f"  {g:<7}{yy.sum():>4}/{(~yy).sum():<5}  {r_['fib4'][0]:>9.3f} / {r_['fib4'][1]:.3f}"
                f"  {r_['ppg'][0]:>9.3f} / {r_['ppg'][1]:.3f}  {r_['either'][0]:>9.3f} / {r_['either'][1]:.3f}"
                f"  {r_['rescued'][0]:>12} / {r_['rescued'][1]}")
        R["rule"][f"spec{int(spec_t*100)}"] = rows

    sub("E. Increments: delta AUC, continuous NRI, IDI")
    R["incr"] = {}
    R["incr"]["PPG+FIB-4 vs FIB-4"] = incr_block("PPG + FIB-4 vs FIB-4",
                                                  y, s.p_fib4.to_numpy(), s.p_fus.to_numpy(), B)
    R["incr"]["PPG+platelets vs platelets"] = incr_block("PPG + platelets vs platelets", y, s.p_plt.to_numpy(), s.p_fusplt.to_numpy(), B)
    D = d.copy(); D["test_fold"] = D.fold
    if db == "mimic":
        sets_ = [("age+sex+vitals", ["age", "male", "hr", "sbp", "dbp", "rr"])]
    else:
        V = ["Triage_HR", "Triage_RR", "Triage_SpO2", "Triage_SBP", "Triage_DBP"]
        sets_ = [("age+sex+triage vitals", ["age", "male"] + V),
                 ("all routine information", ["age", "male", "ppg_hr", "frac_good", "pi"] + V)]
    yA = D.y.to_numpy().astype(int); BA = boot_idx(yA, BOOT, seed=5)
    for nm, cols in sets_:
        p0 = baseline(D, cols); p1 = baseline(D, cols + ["score"])
        R["incr"][nm] = incr_block(f"{nm} + PPG vs {nm} (n={len(D):,})", yA, p0, p1, BA)
    return R


warnings.filterwarnings("ignore")
RES = {"mimic": analyse("mimic", "MIMIC (ICU), laboratory medians within the ICU stay"),
       "mcmed": analyse("mcmed", "MC-MED (ED), first laboratory values of the visit")}
json.dump(RES, open(os.path.join(OUT, "result_28_fib4.json"), "w"), indent=1, ensure_ascii=False)
open(os.path.join(OUT, "report_28_fib4.txt"), "w", encoding="utf-8").write("\n".join(LOG))
print("\n[saved]", OUT)
