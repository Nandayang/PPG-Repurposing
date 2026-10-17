"""Operating points for compensated cirrhosis at the 95%-specificity threshold chosen for all cirrhosis
(same calibration and prevalence correction as 21_paper_analyses.py; 2,000 patient-level bootstrap resamples).

Output: results/paper/result_21b_compensated.json
"""
import os, json, importlib.util, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PPG_ROOT", ".")
RES, DERIVED = os.path.join(ROOT, "results"), os.path.join(ROOT, "derived")
EPS, BOOT, PROJ = 1e-6, 2000, [0.0027, 0.05, 0.10]

spec = importlib.util.spec_from_file_location("m10", os.path.join(HERE, "10_clinical_utility.py"))
m10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m10)

R21 = json.load(open(os.path.join(RES, "paper", "result_21.json"), encoding="utf-8"))
PREV = {"MIMIC": R21["utility_MIMIC"]["prev"], "MC-MED": R21["utility_MC-MED"]["prev"]}


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); return np.log(p / (1 - p))


def tb(s):
    return s.astype(str).isin(["True", "1", "1.0"])


def shift_to(p, y, prev):
    pc = y.mean(); s = np.log(prev / (1 - prev)) - np.log(pc / (1 - pc))
    return 1 / (1 + np.exp(-(logit(p) + s)))


def internal(data):
    D = pd.read_csv(os.path.join(RES, "es", f"patient_oof_{data}_cirrhosis_ssl.csv"))
    if data == "mcmed":
        D["pid"] = D.CSN; D["com"] = tb(D.dx_comp).astype(int)
    else:
        C = pd.read_csv(os.path.join(DERIVED, "cohort_patients.csv"),
                        usecols=["subject_id", "lbl_cirr_comp"]).rename(columns={"subject_id": "pid"})
        D = D.merge(C, on="pid", how="left"); D["com"] = D.lbl_cirr_comp.fillna(0).astype(int)
    y, f = D.y.to_numpy().astype(int), D.fold.to_numpy()
    r = np.clip(D.score.to_numpy(), 0.5 / len(D), 1 - 0.5 / len(D))
    p = m10.cross_calibrate(r, y, f, "platt")
    return y, p, D.com.to_numpy().astype(int)


def external(direction):
    if direction == "MIMIC->MC-MED":
        D = pd.read_csv(os.path.join(RES, "mcmed", "patient_preds.csv"))
        D = D[(D.n_win.fillna(0) >= 10) & ((D.in_cirrhosis == 1) | (D.in_control == 1))].copy()
        D["y"] = D.in_cirrhosis.astype(int); D["score"] = D.s_es; D = D.dropna(subset=["score"])
        D["com"] = tb(D.dx_comp).astype(int)
    else:
        D = pd.read_csv(os.path.join(RES, "reverse", "patient_preds_es_mcmed_cirrhosis_ssl.csv"))
        C = pd.read_csv(os.path.join(DERIVED, "cohort_patients.csv"),
                        usecols=["subject_id", "lbl_cirr_comp"]).rename(columns={"subject_id": "pid"})
        D = D.drop(columns=["com"], errors="ignore").merge(C, on="pid", how="left")
        D["com"] = D.lbl_cirr_comp.fillna(0).astype(int)
    y = D.y.to_numpy().astype(int); p = np.clip(D.score.to_numpy(), EPS, 1 - EPS); z = logit(p)
    pr = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(z, y):
        m = LogisticRegression(C=1e6, max_iter=1000).fit(z[tr, None], y[tr])
        pr[te] = m.predict_proba(z[te, None])[:, 1]
    return y, pr, D.com.to_numpy().astype(int)


def ppv(se, sp, pi):
    return se * pi / max(se * pi + (1 - sp) * (1 - pi), EPS)


def run(name, y, p, com, prev):
    padj = shift_to(p, y, prev)
    ops = {o["name"]: o for o in m10.op_table(y, padj, prev, boot=0)}
    t = ops["Spec 95%"]["thr"]
    keep = (y == 0) | ((y == 1) & (com == 1))
    yy, pp = y[keep], padj[keep]
    se, sp = float((pp[yy == 1] >= t).mean()), float((pp[yy == 0] < t).mean())
    rng = np.random.default_rng(0); bs = []
    for _ in range(BOOT):
        j = rng.integers(0, len(yy), len(yy)); a, b = yy[j], pp[j]
        if a.sum() < 2 or (1 - a).sum() < 2: continue
        bs.append([(b[a == 1] >= t).mean(), (b[a == 0] < t).mean()])
    bs = np.array(bs)
    out = dict(setting=name, n_comp=int(yy.sum()), n_ctrl=int((1 - yy).sum()),
               all_cirrhosis_sens_at_spec95=round(ops["Spec 95%"]["sens"], 3),
               sens=round(se, 3), sens_ci=[round(x, 3) for x in np.percentile(bs[:, 0], [2.5, 97.5])],
               spec=round(sp, 3), spec_ci=[round(x, 3) for x in np.percentile(bs[:, 1], [2.5, 97.5])])
    for pi in PROJ:
        v = [ppv(a, b, pi) for a, b in bs]
        out[f"ppv_{pi}"] = round(100 * ppv(se, sp, pi), 1)
        out[f"ppv_{pi}_ci"] = [round(100 * x, 1) for x in np.percentile(v, [2.5, 97.5])]
    sub = {o["name"]: o for o in m10.op_table(yy, pp, prev, boot=BOOT)}["Spec 95%"]
    out["subset_threshold"] = dict(sens=round(sub["sens"], 3), sens_ci=[round(x, 3) for x in sub["sens_ci"]],
                                   spec=round(sub["spec"], 3))
    return out


if __name__ == "__main__":
    rows = []
    for nm, (y, p, c), db in [("ED internal (MC-MED)", internal("mcmed"), "MC-MED"),
                              ("ED external (MIMIC model, frozen)", external("MIMIC->MC-MED"), "MC-MED"),
                              ("ICU internal (MIMIC)", internal("mimic"), "MIMIC"),
                              ("ICU external (MC-MED model, frozen)", external("MC-MED->MIMIC"), "MIMIC")]:
        rows.append(run(nm, y, p, c, PREV[db]))
    print("\nCompensated cirrhosis vs all controls at the 95%-specificity threshold for all cirrhosis\n")
    for r in rows:
        print(f"{r['setting']}: compensated {r['n_comp']} / controls {r['n_ctrl']}   (all cirrhosis sens {r['all_cirrhosis_sens_at_spec95']})")
        print(f"   sens {r['sens']} ({r['sens_ci'][0]}-{r['sens_ci'][1]})   spec {r['spec']} ({r['spec_ci'][0]}-{r['spec_ci'][1]})")
        print("   PPV " + "   ".join(f"@{pi*100:g}%: {r[f'ppv_{pi}']}% ({r[f'ppv_{pi}_ci'][0]}-{r[f'ppv_{pi}_ci'][1]})" for pi in PROJ))
        s = r["subset_threshold"]
        print(f"   threshold chosen within the subset: sens {s['sens']} ({s['sens_ci'][0]}-{s['sens_ci'][1]})  spec {s['spec']}\n")
    fp = os.path.join(RES, "paper", "result_21b_compensated.json")
    json.dump(rows, open(fp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"[saved] {fp}")
