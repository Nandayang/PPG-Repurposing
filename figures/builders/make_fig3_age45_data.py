"""Fig. 4a data: splits the <50 group of fig6b_auc_by_age.csv (make_plot_data.py) into <45 and 45-49
(2,000 patient-level bootstrap resamples); other age groups are copied.

Usage: python make_fig3_age45_data.py <PPG_ROOT> <figures dir>
"""
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT, FIG = sys.argv[1], sys.argv[2]
RES = os.path.join(ROOT, "results"); rng = np.random.default_rng(2026); B = 2000

def auc_ci(y, s):
    y, s = np.asarray(y).astype(int), np.asarray(s, float); n = len(y); bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n: bs.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), *np.percentile(bs, [2.5, 97.5])

Mi = pd.read_csv(os.path.join(RES, "es", "patient_oof_mimic_cirrhosis_ssl.csv"))
Mc = pd.read_csv(os.path.join(RES, "es", "patient_oof_mcmed_cirrhosis_ssl.csv"))
E = pd.read_csv(os.path.join(RES, "mcmed", "patient_preds.csv"))
E = E[((E.in_cirrhosis == 1) | (E.in_control == 1)) & (E.n_win.fillna(0) >= 10)].copy()
E["y"] = (E.in_cirrhosis == 1).astype(int); E["age"] = E.Age
V = pd.read_csv(os.path.join(RES, "reverse", "patient_preds_es_mcmed_cirrhosis_ssl.csv"))
src = {("MIMIC", "int"): (Mi, "score"), ("MIMIC", "ext"): (V, "score"),
       ("MC-MED", "int"): (Mc, "score"), ("MC-MED", "ext"): (E, "s_es")}

old = pd.read_csv(os.path.join(FIG, "data", "fig6b_auc_by_age.csv"), encoding="utf-8-sig")
rows = []
for (test, kind), (D, col) in src.items():
    setting = old[(old.test == test) & (old.kind == kind)].setting.iloc[0]
    for g, lo_a, hi_a in [("<45", 0, 45), ("45–49", 45, 50)]:
        d = D[(D.age >= lo_a) & (D.age < hi_a)]
        a, l, h = auc_ci(d.y, d[col])
        rows.append(dict(test=test, kind=kind, setting=setting, age_group=g, auc=a, lo=l, hi=h,
                         n_pos=int(d.y.sum()), n_neg=int((1 - d.y).sum())))
    o = old[(old.test == test) & (old.kind == kind) & (old.age_group == "<50")].iloc[0]
    assert rows[-1]["n_pos"] + rows[-2]["n_pos"] == o.n_pos and rows[-1]["n_neg"] + rows[-2]["n_neg"] == o.n_neg
new = pd.DataFrame(rows)
keep = old[old.age_group != "<50"].drop(columns="order")
out = pd.concat([new, keep], ignore_index=True)
order = {"<45": 0, "45–49": 1, "50–59": 2, "60–69": 3, "≥70": 4}
out["order"] = out.age_group.map(order)
out = out.sort_values(["test", "kind", "order"], ascending=[False, False, True])[
    ["test", "kind", "setting", "age_group", "order", "auc", "lo", "hi", "n_pos", "n_neg"]]
out.to_csv(os.path.join(FIG, "data", "fig6b_auc_by_age_45.csv"), index=False, encoding="utf-8-sig")
print(out.to_string(index=False))
