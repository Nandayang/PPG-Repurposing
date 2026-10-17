"""Fig. 5a data: PPG score (within-fold percentile of the internal models) by MELD category in patients with cirrhosis.
MELD uses the UNOS formula (components bounded below at 1, creatinine capped at 4 mg/dL).

Usage: python make_meld_panel_data.py [--root PPG_ROOT]
Outputs (figures/data/): fig_spectrum_meld_points.csv (no identifiers, shuffled), fig_spectrum_meld_summary.csv
"""
from pathlib import Path
import os
import argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
CAT = [("<10", -np.inf, 10), ("10–14", 10, 15), ("15–19", 15, 20), ("≥20", 20, np.inf)]


def meld(L):
    b = L.bili.clip(lower=1); i = L.inr.clip(lower=1); c = L.creat.clip(lower=1, upper=4)
    return (3.78 * np.log(b) + 11.2 * np.log(i) + 9.57 * np.log(c) + 6.43).round(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=Path(os.environ.get("PPG_ROOT", "."))); a = ap.parse_args()
    R = a.root
    mi = pd.read_csv(R / "results/es/patient_oof_mimic_cirrhosis_ssl.csv")
    L = pd.read_csv(R / "derived/labs_median.csv").merge(pd.read_csv(R / "derived/labs_meld_mimic.csv"), on="subject_id",
                                                          how="outer").rename(columns={"subject_id": "pid"})
    mc = pd.read_csv(R / "results/es/patient_oof_mcmed_cirrhosis_ssl.csv")
    Lm = pd.read_csv(R / "derived/mcmed_labs.csv").rename(columns={"CSN": "pid"})
    rng = np.random.default_rng(0); pts, summ = [], []
    for name, O, Lab in (("MIMIC", mi, L), ("MC-MED", mc, Lm)):
        Lab = Lab.copy(); Lab["meld"] = meld(Lab)
        d = O[O.y == 1].merge(Lab[["pid", "meld"]], on="pid", how="left").dropna(subset=["meld"])
        r, p = spearmanr(d.score, d.meld); c50 = float(O[O.y == 0].score.median())
        for k, (lab, lo, hi) in enumerate(CAT):
            s = d.score[(d.meld >= lo) & (d.meld < hi)].to_numpy()
            q = np.percentile(s, [25, 50, 75])
            summ.append(dict(cohort=name, order=k, category=lab, n=len(s), q25=q[0], q50=q[1], q75=q[2],
                             n_total=len(d), spearman_rho=float(r), spearman_p=float(p), control_median=c50))
            pts += [dict(cohort=name, order=k, category=lab, score=float(x)) for x in rng.permutation(s)]
        print(f"{name}: {len(d)} cases with MELD, Spearman rho {r:.3f}, control median {c50:.2f}")
    DATA = HERE.parent / "data"; DATA.mkdir(exist_ok=True)
    pd.DataFrame(pts).to_csv(DATA / "fig_spectrum_meld_points.csv", index=False, encoding="utf-8", float_format="%.4f")
    pd.DataFrame(summ).to_csv(DATA / "fig_spectrum_meld_summary.csv", index=False, encoding="utf-8")
    print(f"[saved] {DATA}")


if __name__ == "__main__":
    main()
