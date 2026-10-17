"""Fig. 3c-d data: Cliff's delta of four pulse-contour descriptors by age group (2,000 bootstrap resamples), the
age x case interaction (29_morph_effects.py) and the case-minus-control median beat by age group.

Usage: python make_fig3_mechanism_data.py [PPG_ROOT] [results_dir] [out_dir]
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
R = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PPG_ROOT", ".")
RES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "results", "paper")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)
KEY = ["notch_pos", "IPA", "dT", "RI"]
AGE_G = [("<45", 0, 45), ("45–49", 45, 50), ("50–59", 50, 60), ("60–69", 60, 70), ("≥70", 70, 200)]
COH = {"mimic": "MIMIC", "mcmed": "MC-MED"}
rows, inter, curves = [], [], []
for db in ["mimic", "mcmed"]:
    O = pd.read_csv(os.path.join(R, "results", "es", f"patient_oof_{db}_cirrhosis_ssl.csv"))
    if db == "mcmed": O["pid"] = O.CSN
    F = pd.read_csv(os.path.join(R, "derived", f"morph_hrv_patient_{db}.csv")).drop(columns=["y"])
    D = O[["pid", "y", "age"]].merge(F, on="pid")
    D["age"] = D.age.clip(upper=90)
    rng = np.random.default_rng(0)
    for f in KEY:
        for i, (g, lo_, hi_) in enumerate(AGE_G):
            m = (D.age >= lo_) & (D.age < hi_)
            x1 = D.loc[m & (D.y == 1), f].dropna().to_numpy(); x0 = D.loc[m & (D.y == 0), f].dropna().to_numpy()
            d = 2 * mannwhitneyu(x1, x0).statistic / (len(x1) * len(x0)) - 1
            bs = []
            for _ in range(2000):
                b1 = x1[rng.integers(0, len(x1), len(x1))]; b0 = x0[rng.integers(0, len(x0), len(x0))]
                bs.append(2 * mannwhitneyu(b1, b0).statistic / (len(b1) * len(b0)) - 1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(cohort=COH[db], feature=f, age_group=g, order=i, delta=d, lo=lo, hi=hi,
                             n_case=len(x1), n_ctrl=len(x0)))
    I = pd.read_csv(os.path.join(RES, f"29_interaction_{db}.csv"))
    for _, r in I[I.feature.isin(KEY)].iterrows():
        inter.append(dict(cohort=COH[db], feature=r.feature, beta=r.beta, lo=r.lo, hi=r.hi, p=r.p, q=r.q))
    P = pd.read_csv(os.path.join(RES, f"29_phase_curves_{db}.csv"))
    P = P[P.age_group.isin([g for g, _, _ in AGE_G])].copy(); P["cohort"] = COH[db]
    curves.append(P[["cohort", "age_group", "phase", "diff_mean", "diff_se"]])
pd.DataFrame(rows).to_csv(os.path.join(OUT, "fig3c_feature_delta_by_age.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(inter).to_csv(os.path.join(OUT, "fig3c_feature_age_interaction.csv"), index=False, encoding="utf-8-sig")
pd.concat(curves).to_csv(os.path.join(OUT, "fig3d_beat_difference_by_age.csv"), index=False, encoding="utf-8-sig")
print("[saved]", OUT)
