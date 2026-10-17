"""Handcrafted features (27_morph_hrv.py): effect sizes (Cliff's delta, BH-FDR), concordance between health systems,
effects by age group and age x case interaction (rank-inverse-normal OLS, beta in SD per 10 years), partial
correlations with the deep score, and median beat curves by age group.

Outputs (results/paper/): 29_effects_<db>.csv, 29_age_effects_<db>.csv, 29_interaction_<db>.csv,
29_phase_curves_<db>.csv, 29_concordance.csv, report_29_morph.txt, result_29_morph.json
"""
import os, json, argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, norm, rankdata
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("PPG_ROOT", "."))
ap.add_argument("--feat", default=None, help="folder with the outputs of 27_morph_hrv.py (default <root>/derived)")
ap.add_argument("--out", default=None, help="default: <root>/results/paper")
ap.add_argument("--boot", type=int, default=1000)
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
OTHER = ["hr_beat", "beat_consistency", "amp_cv"]
HRV = ["sdnn", "rmssd", "pnn50", "sd1", "sd2", "ibi_cv"]
AGE_G = [("<45", 0, 45), ("45–49", 45, 50), ("50–59", 50, 60), ("60–69", 60, 70), ("≥70", 70, 200)]
EPS = 1e-6


def cliff(x1, x0):
    u, p = mannwhitneyu(x1, x0)
    return 2 * u / (len(x1) * len(x0)) - 1, p


def cliff_ci(x1, x0, n, seed=0):
    rng = np.random.default_rng(seed); x1, x0 = np.asarray(x1), np.asarray(x0); bs = []
    for _ in range(n):
        b1 = x1[rng.integers(0, len(x1), len(x1))]; b0 = x0[rng.integers(0, len(x0), len(x0))]
        u = mannwhitneyu(b1, b0).statistic; bs.append(2 * u / (len(b1) * len(b0)) - 1)
    return np.percentile(bs, [2.5, 97.5])


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); r = np.empty_like(p); m = len(p)
    q = p[o] * m / np.arange(1, m + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    r[o] = np.clip(q, 0, 1); return r


def rint(x):
    x = pd.Series(x, dtype=float); ok = x.notna(); out = pd.Series(np.nan, index=x.index)
    r = rankdata(x[ok]); out[ok] = norm.ppf((r - 0.5) / ok.sum()); return out


def pspear(x, y, Z):
    d = pd.DataFrame({"x": x, "y": y}).join(pd.DataFrame(Z)).dropna()
    rx, ry = rankdata(d.x), rankdata(d.y); Zc = sm.add_constant(d.drop(columns=["x", "y"]).to_numpy(float))
    ex = rx - Zc @ np.linalg.lstsq(Zc, rx, rcond=None)[0]; ey = ry - Zc @ np.linalg.lstsq(Zc, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]; n = len(d)
    t = r * np.sqrt((n - 2 - Zc.shape[1] + 1) / max(1 - r ** 2, 1e-12))
    from scipy.stats import t as tdist
    return r, 2 * tdist.sf(abs(t), n - 2 - Zc.shape[1] + 1)


def load(db):
    O = pd.read_csv(os.path.join(a.root, "results", "es", f"patient_oof_{db}_cirrhosis_ssl.csv"))
    if db == "mcmed": O["pid"] = O.CSN
    F = pd.read_csv(os.path.join(a.feat, f"morph_hrv_patient_{db}.csv")).drop(columns=["y"], errors="ignore")
    F = F.rename(columns={"hr": "hr_beat"})
    D = O[["pid", "y", "age", "male", "score"]].merge(F, on="pid", how="left")
    D["age"] = D.age.clip(upper=90)
    r = np.clip(D.score.to_numpy(), 0.5 / len(D), 1 - 0.5 / len(D)); D["score_l"] = np.log(r / (1 - r))
    return D


def feats_of(D):
    return [c for c in MORPH + OTHER + HRV if c in D.columns and D[c].notna().mean() > 0.5 and D[c].nunique() > 2]


def analyse(db):
    D = load(db); F = feats_of(D); R = {"features": F}
    head(f"{db}: cases {int(D.y.sum())} / controls {int((D.y == 0).sum())}; {len(F)} features")
    rows = []
    for c in F:
        x1, x0 = D.loc[D.y == 1, c].dropna(), D.loc[D.y == 0, c].dropna()
        d, p = cliff(x1, x0); lo, hi = cliff_ci(x1, x0, a.boot)
        rows.append(dict(feature=c, med_case=x1.median(), med_ctrl=x0.median(), delta=d, lo=lo, hi=hi, p=p))
    E = pd.DataFrame(rows); E["q"] = bh(E.p)
    say("\n  1. Cases vs controls (Cliff's delta, 95% CI; q = BH-FDR)")
    say(f"  {'feature':<17}{'cases':>10}{'controls':>10}{'delta':>8}{'95% CI':>18}{'q':>11}")
    for _, r in E.sort_values("delta", key=np.abs, ascending=False).iterrows():
        say(f"  {r.feature:<17}{r.med_case:>10.3f}{r.med_ctrl:>10.3f}{r.delta:>+8.3f}  [{r.lo:+.3f}, {r.hi:+.3f}]{r.q:>11.2g}")
    R["effects"] = E.to_dict("records")
    say("\n  3. Delta by age group; age x case interaction (rank-inverse-normal, beta in SD per 10 years)")
    say(f"  {'feature':<17}" + "".join(f"{g:>9}" for g, _, _ in AGE_G) + f"{'beta':>10}{'95% CI':>18}{'q':>9}"
        f"{'rho ctrl':>10}{'rho case':>10}")
    A, I, lines = [], [], []
    for c in F:
        txt = f"  {c:<17}"
        for g, lo_, hi_ in AGE_G:
            m = (D.age >= lo_) & (D.age < hi_)
            x1, x0 = D.loc[m & (D.y == 1), c].dropna(), D.loc[m & (D.y == 0), c].dropna()
            if len(x1) >= 10 and len(x0) >= 10:
                d, p = cliff(x1, x0); lo, hi = cliff_ci(x1, x0, max(200, a.boot // 5), seed=1)
                A.append(dict(feature=c, age_group=g, delta=d, lo=lo, hi=hi, n_case=len(x1), n_ctrl=len(x0)))
                txt += f"{d:>+9.3f}"
            else:
                txt += f"{'-':>9}"
        z = rint(D[c]); X = pd.DataFrame({"y": D.y, "age10": (D.age - 60) / 10, "male": D.male})
        X["yx"] = X.y * X.age10; ok = z.notna()
        fit = sm.OLS(z[ok], sm.add_constant(X[ok])).fit(cov_type="HC3")
        b, (l, h), p = fit.params["yx"], fit.conf_int().loc["yx"], fit.pvalues["yx"]
        r0 = spearmanr(D.age[(D.y == 0) & ok], D[c][(D.y == 0) & ok]).correlation
        r1 = spearmanr(D.age[(D.y == 1) & ok], D[c][(D.y == 1) & ok]).correlation
        I.append(dict(feature=c, beta=b, lo=l, hi=h, p=p, rho_age_ctrl=r0, rho_age_case=r1,
                      beta_case=fit.params["y"]))
        lines.append((c, txt + f"{b:>+10.3f}  [{l:+.3f}, {h:+.3f}]  ", f"{r0:>+10.2f}{r1:>+10.2f}"))
    I = pd.DataFrame(I); I["q"] = bh(I.p)
    qq = dict(zip(I.feature, I.q))
    for c, t1, t2 in lines:
        say(t1 + f"{qq[c]:>9.2g}" + t2)
    R["age_effects"] = A; R["interaction"] = I.to_dict("records")
    say("\n  4. Partial Spearman correlation of the deep score with each feature (age, sex)")
    say(f"  {'feature':<17}{'rho ctrl':>9}{'p':>10}{'rho case':>9}{'p':>10}")
    C = []
    for c in F:
        out = {}
        for nm, m in [("ctrl", D.y == 0), ("case", D.y == 1)]:
            r, p = pspear(D.loc[m, "score_l"], D.loc[m, c], D.loc[m, ["age", "male"]])
            out[nm] = (r, p)
        C.append(dict(feature=c, rho_ctrl=out["ctrl"][0], p_ctrl=out["ctrl"][1],
                      rho_case=out["case"][0], p_case=out["case"][1]))
        say(f"  {c:<17}{out['ctrl'][0]:>+9.2f}{out['ctrl'][1]:>10.2g}{out['case'][0]:>+9.2f}{out['case'][1]:>10.2g}")
    R["score_corr"] = C
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    for nm, sel in [("all", np.ones(len(D), bool)), ("controls", (D.y == 0).to_numpy()), ("cases", (D.y == 1).to_numpy())]:
        Xs, ys = D.loc[sel, F], D.loc[sel, "score_l"]
        pr = cross_val_predict(m, Xs, ys, cv=KFold(5, shuffle=True, random_state=0))
        r2 = 1 - np.sum((ys - pr) ** 2) / np.sum((ys - ys.mean()) ** 2)
        Xa = D.loc[sel, F + ["age", "male"]]
        pa = cross_val_predict(m, Xa, ys, cv=KFold(5, shuffle=True, random_state=0))
        r2a = 1 - np.sum((ys - pa) ** 2) / np.sum((ys - ys.mean()) ** 2)
        say(f"  cross-validated R2 of the deep score from the features ({nm}): {r2:.3f} (with age and sex {r2a:.3f})")
        R[f"r2_{nm}"] = [float(r2), float(r2a)]
    fp = os.path.join(a.feat, f"phase_curves_{db}.npz")
    if os.path.exists(fp):
        Z = np.load(fp); pc = pd.DataFrame(Z["curves"], index=Z["pid"])
        G = D.set_index("pid")[["y", "age"]].join(pc, how="inner")
        ph = np.linspace(0, 1, pc.shape[1]); out = []
        say("\n  5. Median beat, cases - controls")
        for g, lo_, hi_ in [("all", 0, 200)] + AGE_G:
            m = (G.age >= lo_) & (G.age < hi_)
            c1 = G.loc[m & (G.y == 1), pc.columns].to_numpy(); c0 = G.loc[m & (G.y == 0), pc.columns].to_numpy()
            if len(c1) < 10: continue
            m1, m0 = np.median(c1, 0), np.median(c0, 0); df = m1 - m0; j = int(np.argmax(np.abs(df)))
            say(f"    {g:<6} cases {len(c1):>4} / controls {len(c0):>4}: maximum {df[j]:+.4f} at phase {ph[j]:.2f}; "
                f"integral of |difference| {np.trapezoid(np.abs(df), ph):.4f}")
            for k in range(len(ph)):
                out.append(dict(age_group=g, phase=ph[k], case=m1[k], ctrl=m0[k], diff=df[k],
                                diff_mean=c1[:, k].mean() - c0[:, k].mean(),
                                diff_se=np.sqrt(c1[:, k].var(ddof=1) / len(c1) + c0[:, k].var(ddof=1) / len(c0)),
                                case_q25=np.percentile(c1[:, k], 25), case_q75=np.percentile(c1[:, k], 75),
                                ctrl_q25=np.percentile(c0[:, k], 25), ctrl_q75=np.percentile(c0[:, k], 75)))
        pd.DataFrame(out).to_csv(os.path.join(a.out, f"29_phase_curves_{db}.csv"), index=False)
    pd.DataFrame(R["effects"]).to_csv(os.path.join(a.out, f"29_effects_{db}.csv"), index=False)
    pd.DataFrame(A).to_csv(os.path.join(a.out, f"29_age_effects_{db}.csv"), index=False)
    I.to_csv(os.path.join(a.out, f"29_interaction_{db}.csv"), index=False)
    return R


warnings.filterwarnings("ignore")
RES = {db: analyse(db) for db in a.dbs.split(",")}
if len(RES) == 2:
    head("2. Concordance between health systems")
    E1 = pd.DataFrame(RES["mimic"]["effects"]).set_index("feature"); E2 = pd.DataFrame(RES["mcmed"]["effects"]).set_index("feature")
    J = E1[["delta", "q"]].join(E2[["delta", "q"]], lsuffix="_mimic", rsuffix="_mcmed", how="inner")
    rho, p = spearmanr(J.delta_mimic, J.delta_mcmed)
    sig = J[(J.q_mimic < 0.05) | (J.q_mcmed < 0.05)]
    both = J[(J.q_mimic < 0.05) & (J.q_mcmed < 0.05)]
    agree = int((np.sign(both.delta_mimic) == np.sign(both.delta_mcmed)).sum())
    say(f"  {len(J)} features: Spearman rho = {rho:.3f} (p = {p:.2g}); Pearson r = {np.corrcoef(J.delta_mimic, J.delta_mcmed)[0,1]:.3f}")
    say(f"  significant in both: {len(both)}, same direction {agree}; significant in either: {len(sig)}, "
        f"same direction {int((np.sign(sig.delta_mimic) == np.sign(sig.delta_mcmed)).sum())}")
    for f_, r in J.sort_values("delta_mimic").iterrows():
        say(f"    {f_:<17} MIMIC {r.delta_mimic:+.3f}  MC-MED {r.delta_mcmed:+.3f}"
            + ("" if np.sign(r.delta_mimic) == np.sign(r.delta_mcmed) else "   (opposite)"))
    RES["concordance"] = dict(rho=float(rho), p=float(p), n=int(len(J)), n_both_sig=int(len(both)), agree=agree)
    J.reset_index().to_csv(os.path.join(a.out, "29_concordance.csv"), index=False)
json.dump(RES, open(os.path.join(a.out, "result_29_morph.json"), "w"), indent=1, ensure_ascii=False, default=float)
open(os.path.join(a.out, "report_29_morph.txt"), "w", encoding="utf-8").write("\n".join(LOG))
