"""Statistics reported in the paper, from the patient-level predictions of 16_train_es.py (internal) and of
14_mcmed_external.py / 20_reverse_external.py (frozen, other health system).

Sections: summary, table1, utility (calibration, prevalence correction, operating points, decision curves),
external (calibration of frozen models and local recalibration), labs (FIB-4, APRI, platelets), age (age x score
interaction), meld, dispo (ED disposition strata), albumin.
Requires derived/labs_median.csv from 11_fib4_headtohead.py.
Usage: python 21_paper_analyses.py [--only labs,meld] [--boot 2000]
Outputs (results/paper/): report_21.txt, result_21.json, table1_mimic.csv, table1_mcmed.csv
"""
import os, re, json, glob, time, argparse, warnings, importlib.util
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, norm, chi2
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.environ.get("PPG_ROOT", ".")
CACHE   = os.path.join(ROOT, "cache")
DERIVED = os.path.join(ROOT, "derived")
RES     = os.path.join(ROOT, "results")
OUT     = os.path.join(RES, "paper")
MIMIC   = os.environ.get("MIMIC_DIR", "data/mimic-iii-clinical-database-1.4")
MC      = os.environ.get("MCMED_DIR", "data/mc-med")
EPS     = 1e-6
AGE_BINS = [(0, 50), (50, 60), (60, 70), (70, 200)]

_LOG = []
def say(s=""):
    print(s, flush=True); _LOG.append(str(s))
def head(t):
    say(); say("=" * 78); say(t); say("=" * 78)
def sub(t):
    say(); say("-" * 78); say(" " + t); say("-" * 78)


def load_module(fn, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


m10 = load_module("10_clinical_utility.py", "m10")
m11 = load_module("11_fib4_headtohead.py", "m11")
mst = load_module("stats_common.py", "mst")
auc_ci, fmt, paired_delta = mst.auc_ci, mst.fmt, mst.paired_delta
BOOT = 2000
R = {}


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); return np.log(p / (1 - p))


def tb(s):
    return s.astype(str).isin(["True", "1", "1.0"])


def P(*a):
    return os.path.join(RES, *a)


def load_internal(data, task="cirrhosis"):
    f = P("es", f"patient_oof_{data}_{task}_ssl.csv")
    if not os.path.exists(f):
        return None
    D = pd.read_csv(f)
    if data == "mcmed" and "CSN" in D.columns:
        D["pid"] = D.CSN
    return D


def load_ext_mimic2mcmed():
    f = P("mcmed", "patient_preds.csv")
    if not os.path.exists(f):
        return None
    D = pd.read_csv(f)
    D = D[(D.n_win.fillna(0) >= 10) & ((D.in_cirrhosis == 1) | (D.in_control == 1))].copy()
    D["y"] = D.in_cirrhosis.astype(int); D["score"] = D.s_es
    D["pid"] = D.CSN; D["age"] = D.Age; D["male"] = (D.Gender == "M").astype(int)
    return D.dropna(subset=["score"])


def load_ext_mcmed2mimic():
    f = P("reverse", "patient_preds_es_mcmed_cirrhosis_ssl.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


SETS = {}
def sets():
    if not SETS:
        SETS["MIMIC internal"] = load_internal("mimic")
        SETS["MC-MED internal"] = load_internal("mcmed")
        SETS["MIMIC->MC-MED frozen"] = load_ext_mimic2mcmed()
        SETS["MC-MED->MIMIC frozen"] = load_ext_mcmed2mimic()
    return {k: v for k, v in SETS.items() if v is not None}


def mimic_labs(sids):
    base = pd.read_csv(os.path.join(DERIVED, "labs_median.csv"))
    fp = os.path.join(DERIVED, "labs_meld_mimic.csv")
    if os.path.exists(fp):
        ext = pd.read_csv(fp)
    else:
        say("  [labs] MIMIC creatinine (50912) and sodium (50983, 50824) ...")
        items = {50912: "creat", 50983: "na", 50824: "na"}
        rng = {"creat": (0.1, 25), "na": (100, 180)}
        icu = pd.read_csv(m11._find("ICUSTAYS.csv"), usecols=["SUBJECT_ID", "INTIME", "OUTTIME"])
        icu.columns = [c.lower() for c in icu.columns]
        icu = icu[icu.subject_id.isin(sids)]
        icu["intime"] = pd.to_datetime(icu.intime); icu["outtime"] = pd.to_datetime(icu.outtime)
        win = icu.groupby("subject_id").agg(t0=("intime", "min"), t1=("outtime", "max"))
        keep, n, t0 = [], 0, time.time()
        for ch in pd.read_csv(m11._find("LABEVENTS.csv"),
                              usecols=["SUBJECT_ID", "ITEMID", "CHARTTIME", "VALUENUM"], chunksize=3_000_000):
            ch.columns = [c.lower() for c in ch.columns]; n += len(ch)
            ch = ch[ch.itemid.isin(items) & ch.subject_id.isin(sids) & ch.valuenum.notna()]
            if len(ch): keep.append(ch)
            print(f"    {n/1e6:.0f}M rows  {time.time()-t0:.0f}s", end="\r", flush=True)
        lab = pd.concat(keep, ignore_index=True)
        lab["name"] = lab.itemid.map(items)
        lab = lab[[rng[k][0] <= v <= rng[k][1] for k, v in zip(lab.name, lab.valuenum)]]
        lab["t"] = pd.to_datetime(lab.charttime)
        lab = lab.merge(win, left_on="subject_id", right_index=True, how="left")
        lab["in_icu"] = (lab.t >= lab.t0 - pd.Timedelta("6h")) & (lab.t <= lab.t1)
        v = lab[lab.in_icu].groupby(["subject_id", "name"]).valuenum.median()
        va = lab.groupby(["subject_id", "name"]).valuenum.median()
        ext = v.reindex(va.index).fillna(va).unstack("name").reset_index()
        ext.to_csv(fp, index=False)
        say(f"\n  [labs] derived/labs_meld_mimic.csv ({len(ext):,} patients)")
    L = base.merge(ext, on="subject_id", how="outer").rename(columns={"subject_id": "pid"})
    return L


MC_PAT = {
    "plt":   r"^platelet count",
    "ast":   r"^ast\b|sgot|^aspartate amino",
    "alt":   r"^alt\b|sgpt|^alanine amino",
    "bili":  r"^bilirubin, total$|^total bilirubin",
    "inr":   r"^inr$|^poc:\s*inr",
    "creat": r"^creatinine$|^poc:\s*creatinine",
    "na":    r"^sodium$|^sodium\.whole bld|^poc:\s*sodium",
    "alb":   r"^albumin$",
}
MC_RANGE = {"plt": (1, 2000), "ast": (1, 20000), "alt": (1, 20000), "bili": (0.05, 60),
            "inr": (0.5, 20), "creat": (0.1, 25), "na": (100, 180), "alb": (0.5, 7)}


def mcmed_labs():
    fp = os.path.join(DERIVED, "mcmed_labs.csv")
    if os.path.exists(fp):
        return pd.read_csv(fp)
    say("  [labs] MC-MED labs.csv ...")
    csns = set(pd.read_csv(os.path.join(DERIVED, "mcmed_patients.csv"), usecols=["CSN"]).CSN)
    keep, nmap, n, t0 = [], {}, 0, time.time()
    for ch in pd.read_csv(os.path.join(MC, "labs.csv"), chunksize=2_000_000, low_memory=False,
                          usecols=["CSN", "Result_time", "Component_name", "Component_value"]):
        n += len(ch); ch = ch[ch.CSN.isin(csns)]
        for nm in ch.Component_name.dropna().unique():
            if nm not in nmap:
                low = str(nm).strip().lower()
                nmap[nm] = next((k for k, pat in MC_PAT.items() if re.search(pat, low)), None)
        ch = ch.assign(key=ch.Component_name.map(nmap)).dropna(subset=["key"])
        if len(ch): keep.append(ch)
        print(f"    {n/1e6:.0f}M rows  {time.time()-t0:.0f}s", end="\r", flush=True)
    lab = pd.concat(keep, ignore_index=True)
    say("")
    say("  laboratory name mapping:")
    cnt = lab.groupby(["key", "Component_name"]).size().sort_values(ascending=False)
    for (k, nm), c in cnt.items():
        say(f"    {k:<6} <- {nm:<40} {c:>8,}")
    miss = [nm for nm, k in nmap.items() if k is None and re.search(r"sgot|sgpt|amino|platelet|bilirub|creatin|sodium|inr", str(nm).lower())]
    if miss:
        say(f"  similar names not mapped: {miss[:15]}")
    v = lab.Component_value.astype(str).str.replace(r"[<>=,\s]", "", regex=True)
    lab["val"] = pd.to_numeric(v, errors="coerce")
    lab = lab.dropna(subset=["val"])
    lab = lab[[MC_RANGE[k][0] <= x <= MC_RANGE[k][1] for k, x in zip(lab.key, lab.val)]]
    lab = lab.sort_values("Result_time")
    W = lab.groupby(["CSN", "key"]).val.first().unstack("key").reset_index()
    W.to_csv(fp, index=False)
    say(f"  [labs] derived/mcmed_labs.csv ({len(W):,} visits)")
    return W


LABS = {}
def labs_for(db):
    if db not in LABS:
        if db == "mimic":
            sids = set(pd.read_csv(P("es", "patient_oof_mimic_cirrhosis_ssl.csv"), usecols=["pid"]).pid)
            LABS[db] = mimic_labs(sids)
        else:
            LABS[db] = mcmed_labs().rename(columns={"CSN": "pid"})
    return LABS[db]


def db_of(name):
    return {"MIMIC internal": "mimic", "MC-MED->MIMIC frozen": "mimic"}.get(name, "mcmed")


def meld_scores(L):
    L = L.copy()
    if not {"bili", "inr", "creat"} <= set(L.columns):
        return L
    b = L.bili.clip(lower=1); i = L.inr.clip(lower=1); c = L.creat.clip(lower=1, upper=4)
    m = 3.78 * np.log(b) + 11.2 * np.log(i) + 9.57 * np.log(c) + 6.43
    L["meld"] = m.round(0)
    if "na" in L.columns:
        na = L.na.clip(125, 137)
        L["meld_na"] = np.where(m > 11, m + 1.32 * (137 - na) - 0.033 * m * (137 - na), m).round(0)
        L.loc[L.na.isna(), "meld_na"] = np.nan
    return L


def s_summary():
    head("0. Summary (7:1:2 protocol; frozen = 5-model ensemble)")
    rows = []
    for f in sorted(glob.glob(P("es", "result_*.json"))):
        j = json.load(open(f, encoding="utf-8"))
        rows.append((j["data"], j["task"], j["arm"], j["auc"], *j["ci"], j["n"], j["n_pos"],
                     " / ".join(f"{k} {v:.3f}" for k, v in list(j.get("baselines", {}).items())[:3])))
    say(f"  {'data':<7}{'task':<14}{'arm':<5}{'AUC [95% CI]':<24}{'n (+)':<14}baselines")
    for d, t, a, au, lo, hi, n, k, b in rows:
        say(f"  {d:<7}{t:<14}{a:<5}{au:.3f} [{lo:.3f}, {hi:.3f}]   {n:>5} ({k:>4})   {b}")
    fw = P("mcmed", "result.json"); rv = P("reverse", "result_es_mcmed_cirrhosis_ssl.json")
    if os.path.exists(fw):
        j = json.load(open(fw)); a = j.get("auc_es")
        if a: say(f"  MIMIC->MC-MED frozen   {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]")
    if os.path.exists(rv):
        j = json.load(open(rv, encoding="utf-8")); a = j["auc"]
        say(f"  MC-MED->MIMIC frozen   {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]")
    R["summary"] = rows


def smd(x1, x0, binary=False):
    x1, x0 = pd.Series(x1).dropna().astype(float), pd.Series(x0).dropna().astype(float)
    if len(x1) < 2 or len(x0) < 2: return np.nan
    if binary:
        p1, p0 = x1.mean(), x0.mean(); s = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
        return (p1 - p0) / s if s > 0 else 0.0
    s = np.sqrt((x1.var() + x0.var()) / 2)
    return (x1.mean() - x0.mean()) / s if s > 0 else 0.0


def table1(D, spec, name):
    a, b = D[D.y == 1], D[D.y == 0]
    say(f"  {'variable':<26}{'cirrhosis n=' + str(len(a)):>26}{'controls n=' + str(len(b)):>26}{'missing':>8}{'SMD':>8}")
    rows = []
    for lab, col, kind in spec:
        if col not in D.columns or D[col].notna().sum() == 0: continue
        miss = D[col].isna().mean()
        if kind == "bin":
            v1, v0 = a[col].astype(float), b[col].astype(float)
            c1, c0 = [f"{int(v.sum())} ({v.mean():.1%})" if v.notna().any() else "-" for v in (v1, v0)]
            s = smd(v1, v0, True)
        else:
            q1, q0 = a[col].quantile([.5, .25, .75]), b[col].quantile([.5, .25, .75])
            c1 = f"{q1.iloc[0]:.1f} [{q1.iloc[1]:.1f}, {q1.iloc[2]:.1f}]"
            c0 = f"{q0.iloc[0]:.1f} [{q0.iloc[1]:.1f}, {q0.iloc[2]:.1f}]"
            s = smd(a[col], b[col])
        say(f"  {lab:<26}{c1:>26}{c0:>26}{miss:>8.0%}{s:>8.2f}")
        rows.append(dict(variable=lab, cirrhosis=c1, control=c0, missing=miss, smd=s))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"table1_{name}.csv"), index=False, encoding="utf-8-sig")


def s_table1():
    head("1. Cohort flow and Table 1")
    sub("MIMIC-III (ICU)")
    C = pd.read_csv(os.path.join(DERIVED, "cohort_patients.csv"))
    D = load_internal("mimic")
    say(f"  ICU patients with PPG {len(C):,}")
    say(f"    cirrhosis {int((C.lbl_cirrhosis==1).sum())} (>= 10 high-quality segments: {int(((C.lbl_cirrhosis==1)&(C.n_ppg_good>=10)).sum())})"
        f"   control pool {int((C.lbl_control==1).sum()):,}")
    say(f"    analyzed after 1:4 matching: cirrhosis {int(D.y.sum())} / controls {int((1-D.y).sum()):,}")
    T = D.merge(C[["subject_id", "cis", "lbl_cirr_decomp", "lbl_hf", "lbl_sepsis", "lbl_esrd", "n_ppg_good"]]
                .rename(columns={"subject_id": "pid"}), on="pid", how="left")
    T = T.merge(meld_scores(labs_for("mimic")), on="pid", how="left")
    T["age_c"] = T.age.where(T.age <= 89, 90)
    T["metavision"] = (T.cis == "Metavision").astype(float)
    T["decomp_"] = np.where(T.y == 1, T.lbl_cirr_decomp, np.nan)
    table1(T, [("Age (>89 set to 90)", "age_c", "cont"), ("Male", "male", "bin"),
               ("Heart failure", "lbl_hf", "bin"), ("Sepsis", "lbl_sepsis", "bin"), ("ESRD", "lbl_esrd", "bin"),
               ("Decompensated (cirrhosis)", "decomp_", "bin"), ("Metavision era", "metavision", "bin"),
               ("Monitor heart rate", "hr", "cont"), ("PPG pulse rate", "ppg_hr", "cont"),
               ("Systolic BP (arterial)", "sbp", "cont"), ("Diastolic BP (arterial)", "dbp", "cont"), ("Respiratory rate", "rr", "cont"),
               ("High-quality PPG segments", "n_ppg_good", "cont"),
               ("Platelets", "plt", "cont"), ("AST", "ast", "cont"), ("ALT", "alt", "cont"),
               ("Total bilirubin", "bili", "cont"), ("INR", "inr", "cont"), ("Creatinine", "creat", "cont"),
               ("Albumin", "alb", "cont"), ("MELD", "meld", "cont")], "mimic")
    sub("MC-MED (ED)")
    try:
        V = pd.read_csv(os.path.join(MC, "visits.csv"), usecols=["MRN", "CSN"])
        Wf = pd.read_csv(os.path.join(MC, "waveform_summary.csv"), usecols=["CSN", "Type"])
        pl = set(Wf[Wf.Type == "Pleth"].CSN)
        Co = pd.read_csv(os.path.join(DERIVED, "mcmed_cohort_visits.csv"), usecols=["MRN", "CSN", "grp"])
        Pm = pd.read_csv(os.path.join(DERIVED, "mcmed_patients.csv"))
        say(f"  ED visits {len(V):,} / patients {V.MRN.nunique():,}; patients with Pleth {V[V.CSN.isin(pl)].MRN.nunique():,}")
        say(f"    cirrhosis {(Co.grp=='cirrhosis').sum()}   matched controls {(Co.grp=='control').sum():,}")
        say(f"    cached: cirrhosis {int(Pm.in_cirrhosis.sum())} / controls {int(Pm.in_control.sum()):,}")
        ok = Pm.n_win.fillna(0) >= 10
        say(f"    analyzed (>= 10 windows): cirrhosis {int((ok & (Pm.in_cirrhosis==1)).sum())} / "
            f"controls {int((ok & (Pm.in_control==1)).sum()):,}")
    except Exception as e:
        say(f"  [!] cohort flow not available: {e}")
    D = load_internal("mcmed").copy()
    D = D.merge(meld_scores(labs_for("mcmed")), on="pid", how="left")
    D["acuity12"] = D.Triage_acuity.astype(str).str[:1].isin(["1", "2"]).astype(float)
    D["admit"] = D.ED_dispo.isin(["Inpatient", "ICU", "Observation"]).astype(float)
    D["decomp_"] = np.where(D.y == 1, tb(D.dx_decomp).astype(float), np.nan)
    for c in ["rad_cirr", "bb_ever", "nsbb_ever"]:
        if c in D.columns: D[c] = pd.to_numeric(D[c], errors="coerce")
    table1(D, [("Age", "age", "cont"), ("Male", "male", "bin"),
               ("Triage acuity 1-2", "acuity12", "bin"), ("Admitted/observation/ICU", "admit", "bin"),
               ("Decompensated (cirrhosis)", "decomp_", "bin"), ("Cirrhosis on imaging", "rad_cirr", "bin"),
               ("Beta-blocker", "bb_ever", "bin"), ("Non-selective beta-blocker", "nsbb_ever", "bin"),
               ("Triage heart rate", "Triage_HR", "cont"), ("Triage systolic BP", "Triage_SBP", "cont"),
               ("Triage diastolic BP", "Triage_DBP", "cont"), ("Triage respiratory rate", "Triage_RR", "cont"), ("Triage SpO2", "Triage_SpO2", "cont"),
               ("PPG pulse rate", "ppg_hr", "cont"), ("Perfusion index", "pi", "cont"), ("Good-window fraction", "frac_good", "cont"),
               ("Platelets", "plt", "cont"), ("AST", "ast", "cont"), ("ALT", "alt", "cont"),
               ("Total bilirubin", "bili", "cont"), ("INR", "inr", "cont"), ("Creatinine", "creat", "cont"),
               ("Sodium", "na", "cont"), ("Albumin", "alb", "cont"), ("MELD-Na", "meld_na", "cont")], "mcmed")


def mcmed_source_prev():
    try:
        V = pd.read_csv(os.path.join(MC, "visits.csv"), usecols=["MRN", "CSN"])
        Wf = pd.read_csv(os.path.join(MC, "waveform_summary.csv"), usecols=["CSN", "Type"])
        n = V[V.CSN.isin(set(Wf[Wf.Type == "Pleth"].CSN))].MRN.nunique()
        Co = pd.read_csv(os.path.join(DERIVED, "mcmed_cohort_visits.csv"), usecols=["MRN", "grp"])
        k = Co[Co.grp == "cirrhosis"].MRN.nunique()
        return k / n, k, n
    except Exception as e:
        say(f"  [!] MC-MED source prevalence not available ({e}); 1.5% used"); return 0.015, None, None


def utility_block(name, y, p_cal, prev, age=None, coh=None, key=""):
    pc = y.mean()
    shift = np.log(prev / (1 - prev)) - np.log(pc / (1 - pc))
    padj = 1 / (1 + np.exp(-(logit(p_cal) + shift)))
    say(f"  prevalence: cohort {pc:.1%} -> target {prev:.2%} (logit shift {shift:+.3f})")
    ops = m10.op_table(y, padj, prev, boot=BOOT)
    say(f"  {'point':<10}{'sensitivity [95% CI]':>24}{'specificity [95% CI]':>24}{'PPV':>8}{'NPV':>8}{'NNS':>7}")
    for r in ops:
        say(f"  {r['name']:<10}{r['sens']:>8.3f} [{r['sens_ci'][0]:.3f},{r['sens_ci'][1]:.3f}]"
            f"{r['spec']:>8.3f} [{r['spec_ci'][0]:.3f},{r['spec_ci'][1]:.3f}]"
            f"{r['ppv']:>8.1%}{r['npv']:>8.1%}{r['nns']:>7.1f}")
    out = dict(prev=prev, shift=float(shift), ops=ops)
    if age is not None and coh is not None:
        yo = [r for r in ops if r["name"] == "Youden"][0]
        st = m10.age_standardised(y, padj, age, yo["thr"], coh)
        ppv = st["sens_std"] * prev / max(st["sens_std"] * prev + (1 - st["spec_std"]) * (1 - prev), EPS)
        say(f"  age-standardized (Youden): sens {st['sens_std']:.3f}  spec {st['spec_std']:.3f}  "
            f"PPV {ppv:.1%}  NNS {1/max(ppv, EPS):.1f}   (crude NNS {yo['nns']:.1f})")
        out["age_std"] = st
    dc = m10.dca(y, padj, prev, [0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30])
    say(f"  decision curve  {'threshold':>9}{'model':>9}{'all':>9}   per 1000")
    for r in dc:
        say(f"                  {r['pt']:>9.3f}{r['nb_model']:>9.4f}{r['nb_all']:>9.4f}   {r['per_1000']:>6.1f}")
    out["dca"] = dc
    pv = [0.01, 0.02, 0.05, 0.075, 0.10, 0.15]
    say("  NNS by prevalence (Youden / Spec 90%): " + "  ".join(
        f"{x:.0%}: " + "/".join(f"{1/max(r['sens']*x/(r['sens']*x+(1-r['spec'])*(1-x)), EPS):.1f}"
                               for r in ops if r["name"] in ("Youden", "Spec 90%")) for x in pv))
    return out


def s_utility():
    head("2. Clinical utility (internal; Platt scaling fitted on the other folds, prevalence correction)")
    coh = pd.read_csv(os.path.join(DERIVED, "cohort_patients.csv"))
    prev_mc, k, n = mcmed_source_prev()
    for nm, D, prev, cc in [("MIMIC", load_internal("mimic"), float((coh.lbl_cirrhosis == 1).mean()), coh),
                            ("MC-MED", load_internal("mcmed"), prev_mc, None)]:
        sub(f"{nm} internal   n={len(D):,} ({int(D.y.sum())}+)")
        if nm == "MC-MED" and n:
            say(f"  source: {n:,} ED patients with Pleth, {k} with cirrhosis ({prev:.2%})")
        y, f = D.y.to_numpy().astype(int), D.fold.to_numpy()
        r = np.clip(D.score.to_numpy(), 0.5 / len(D), 1 - 0.5 / len(D))
        pcal = m10.cross_calibrate(r, y, f, "platt")
        cm = m10.cal_metrics(pcal, y)
        say(f"  calibration: Brier {cm['brier']:.4f}  slope {cm['slope']:.3f}  intercept {cm['intercept']:+.3f}  ECE {cm['ece']:.4f}")
        R[f"utility_{nm}"] = dict(calibration=cm, **utility_block(
            nm, y, pcal, prev, D.age.to_numpy() if cc is not None else None, cc))


def citl(y, p):
    z = logit(p)
    return brentq(lambda a: np.sum(y - 1 / (1 + np.exp(-(a + z)))), -20, 20)


def s_external():
    head("3. Frozen models: calibration and local recalibration")
    coh = pd.read_csv(os.path.join(DERIVED, "cohort_patients.csv"))
    prev_mi = float((coh.lbl_cirrhosis == 1).mean()); prev_mc, _, _ = mcmed_source_prev()
    S = sets()
    src_prev = {k: float(load_internal(db).y.mean()) for k, db in [("MIMIC", "mimic"), ("MC-MED", "mcmed")]}
    for nm, prev, src in [("MIMIC->MC-MED frozen", prev_mc, "MIMIC"), ("MC-MED->MIMIC frozen", prev_mi, "MC-MED")]:
        if nm not in S: continue
        D = S[nm]; y = D.y.to_numpy().astype(int); p = np.clip(D.score.to_numpy(), EPS, 1 - EPS)
        sub(f"{nm}   n={len(D):,} ({int(y.sum())}+)")
        cm = m10.cal_metrics(p, y); a = citl(y, p)
        exp_shift = np.log(y.mean() / (1 - y.mean())) - np.log(src_prev[src] / (1 - src_prev[src]))
        say(f"  frozen probabilities: Brier {cm['brier']:.4f}  slope {cm['slope']:.3f}  CITL {a:+.3f}  ECE {cm['ece']:.4f}")
        say(f"    intercept expected from the sampled prevalences ({src_prev[src]:.1%} -> {y.mean():.1%}): {exp_shift:+.3f}")
        pr = np.zeros(len(y)); pi_ = np.zeros(len(y)); z = logit(p)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(z, y):
            m = LogisticRegression(C=1e6, max_iter=1000).fit(z[tr, None], y[tr])
            pr[te] = m.predict_proba(z[te, None])[:, 1]
            a_tr = citl(y[tr], p[tr]); pi_[te] = 1 / (1 + np.exp(-(z[te] + a_tr)))
        for lab, q in [("intercept", pi_), ("intercept + slope", pr)]:
            c2 = m10.cal_metrics(q, y)
            say(f"  local recalibration ({lab}, 5-fold): Brier {c2['brier']:.4f}  slope {c2['slope']:.3f}  "
                f"intercept {c2['intercept']:+.3f}  ECE {c2['ece']:.4f}")
        R[f"external_{nm}"] = dict(raw=cm, citl=a, expected_shift=float(exp_shift),
                                   recal=m10.cal_metrics(pr, y),
                                   **utility_block(nm, y, pr, prev))


def labs_h2h(name, D, L, folds=5):
    d = D[["pid", "y", "score", "fold", "age", "male"]].merge(L, on="pid", how="left")
    d["age"] = d.age.clip(upper=90)
    for c in ["ast", "alt", "plt", "bili", "inr"]:
        if c not in d.columns: d[c] = np.nan
    d = m11.add_scores(d).reset_index(drop=True)
    r = np.clip(d.score.to_numpy(), 0.5 / len(d), 1 - 0.5 / len(d))
    d["ppg_l"] = logit(r)
    say(f"  {'test':<8}{'avail.':>8}{'positive':>14}{'negative':>12}")
    for c in ["plt", "ast", "alt", "fib4"]:
        if c in d.columns:
            s = d[c].notna(); say(f"  {c.upper():<8}{s.mean():>8.1%}{s[d.y==1].mean():>14.1%}{s[d.y==0].mean():>12.1%}")
    cc = d.fib4.notna()
    if cc.sum() < 50 or d.y[cc].nunique() < 2:
        say("  too few patients with FIB-4"); return None
    s = d[cc].reset_index(drop=True); ys, fs = s.y.to_numpy().astype(int), s.fold.to_numpy()
    cands = {"PPG": s.ppg_l.to_numpy(),
             "FIB-4": s.fib4.to_numpy(), "APRI": s.apri.to_numpy(), "platelets": -s.plt.to_numpy(),
             "platelets+AST+ALT+age": m11.cv_logit(s, ["log_plt", "log_ast", "log_alt", "age"], ys, fs, folds),
             "PPG+platelets": m11.cv_logit(s, ["ppg_l", "log_plt"], ys, fs, folds),
             "PPG+FIB-4": m11.cv_logit(s, ["ppg_l", "log_fib4"], ys, fs, folds)}
    res = {}
    say(f"\n  patients with FIB-4: {len(s):,} ({int(ys.sum())}+); combined models cross-validated on the same folds")
    for k, v in cands.items():
        v = np.nan_to_num(v, nan=np.nanmedian(v)); cands[k] = v
        rr = auc_ci(ys, v, BOOT); res[k] = rr[:3]; say(f"    {k:<20}{fmt(rr)}")
    say("  paired bootstrap:")
    for x, w in [("PPG", "FIB-4"), ("PPG", "platelets"),
                 ("PPG", "APRI"), ("PPG+platelets", "platelets"),
                 ("PPG+FIB-4", "FIB-4"), ("PPG+platelets", "PPG"), ("PPG+FIB-4", "PPG")]:
        o, lo, hi, p = paired_delta(ys, cands[x], cands[w], BOOT)
        say(f"    {x:<14} vs {w:<14} {o:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={min(p,1):.3g}")
        res[f"{x} vs {w}"] = [float(o), float(lo), float(hi), float(min(p, 1))]
    yall = d.y.to_numpy().astype(int)
    its = {"PPG": d.ppg_l.to_numpy(), "FIB-4": d.fib4.fillna(d.fib4.median()).to_numpy(),
           "platelets": (-d.plt.fillna(d.plt.median())).to_numpy()}
    say(f"  all {len(d):,} patients (missing values imputed): " +
        "  ".join(f"{k} {roc_auc_score(yall, v):.3f}" for k, v in its.items()))
    ctl = d.y == 0
    cor = {c: spearmanr(d.ppg_l[ctl & d[c].notna()], d[c][ctl & d[c].notna()]).correlation
           for c in ["plt", "fib4", "ast", "alt", "bili", "inr", "alb", "creat"]
           if c in d.columns and (ctl & d[c].notna()).sum() > 100}
    say("  negatives: Spearman rho of the score with " + "  ".join(f"{k} {v:+.2f}" for k, v in cor.items()))
    res["control_corr"] = cor; res["n_fib4"] = int(len(s)); res["frac_fib4"] = float(cc.mean())
    return res


def s_labs():
    head("4. PPG score and serum fibrosis markers")
    for db, dbn in [("mimic", "MIMIC (ICU medians)"), ("mcmed", "MC-MED (first ED values)")]:
        for task, tn in [("cirrhosis", "cirrhosis vs controls"), ("decomp", "decompensated vs compensated")]:
            D = load_internal(db, task)
            if D is None:
                sub(f"{dbn} / {tn}: results/es/patient_oof_{db}_{task}_ssl.csv not found")
                continue
            sub(f"{dbn} / {tn}   n={len(D):,} ({int(D.y.sum())}+)  PPG AUC {roc_auc_score(D.y, D.score):.3f}")
            R[f"labs_{db}_{task}"] = labs_h2h(tn, D, labs_for(db))


def logistic_mle(X, y, it=100):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        H = (X * W[:, None]).T @ X + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p)); b += step
        if np.abs(step).max() < 1e-8: break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return b, np.sqrt(np.diag(np.linalg.inv(H))), ll


def s_age():
    head("5. Age x score interaction: logit(y) ~ score + age + sex + score x age")
    say("  score: standardized logit of the probability or rank; age centred at 60 years, per 10 years")
    for nm, D in sets().items():
        D = D.dropna(subset=["score", "age"]).copy()
        y = D.y.to_numpy().astype(float)
        s = logit(np.clip(D.score.to_numpy(), 0.5 / len(D), 1 - 0.5 / len(D))); s = (s - s.mean()) / s.std()
        a = (D.age.clip(upper=90).to_numpy() - 60) / 10; m = D.male.to_numpy().astype(float)
        X1 = np.column_stack([np.ones_like(s), s, a, m]); X2 = np.column_stack([X1, s * a])
        _, _, ll1 = logistic_mle(X1, y); b, se, ll2 = logistic_mle(X2, y)
        z = b[4] / se[4]; pw = 2 * norm.sf(abs(z)); plr = chi2.sf(2 * (ll2 - ll1), 1)
        say(f"\n  [{nm}]  n={len(D):,} ({int(y.sum())}+)")
        say(f"    interaction (log-OR per SD per 10 years) {b[4]:+.3f} [{b[4]-1.96*se[4]:+.3f}, {b[4]+1.96*se[4]:+.3f}]"
            f"   Wald p={pw:.2g}   LR p={plr:.2g}")
        ors = {ag: float(np.exp(b[1] + b[4] * (ag - 60) / 10)) for ag in (40, 55, 70, 80)}
        say("    OR per SD: " + "   ".join(f"age {k} {v:.2f}" for k, v in ors.items()))
        aucs = []
        for lo, hi in AGE_BINS:
            q = D[(D.age >= lo) & (D.age < hi)]
            aucs.append(roc_auc_score(q.y, q.score) if q.y.nunique() > 1 else np.nan)
        say("    AUC by age: " + "  ".join(f"{lo}-{hi if hi < 200 else ''} {v:.3f}" for (lo, hi), v in zip(AGE_BINS, aucs)))
        rc = spearmanr(D[D.y == 0].score, D[D.y == 0].age).correlation
        rp = spearmanr(D[D.y == 1].score, D[D.y == 1].age).correlation
        say(f"    score vs age, Spearman: controls {rc:+.3f}   cirrhosis {rp:+.3f}")
        R[f"age_{nm}"] = dict(inter=[float(b[4]), float(se[4]), float(pw), float(plr)], or_by_age=ors,
                              auc_strata=aucs, rho_ctrl=rc, rho_case=rp)


def partial_spearman(x, y, z):
    rx, ry, rz = [pd.Series(v).rank().to_numpy() for v in (x, y, z)]
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return pearsonr(ex, ey)


def s_meld():
    head("6. Within cirrhosis: score vs MELD / MELD-Na")
    for nm, D in sets().items():
        L = meld_scores(labs_for(db_of(nm)))
        d = D[D.y == 1][["pid", "score", "age"]].merge(L, on="pid", how="left")
        d["age"] = d.age.clip(upper=90)
        sub(f"{nm}   cirrhosis {len(d)}, with MELD {int(d.meld.notna().sum()) if 'meld' in d else 0}")
        if "meld" not in d or d.meld.notna().sum() < 30:
            say("  too few patients"); continue
        out = {}
        for col, lab in [("meld", "MELD"), ("meld_na", "MELD-Na"), ("bili", "bilirubin"), ("inr", "INR"),
                         ("creat", "creatinine"), ("na", "sodium"), ("alb", "albumin"), ("plt", "platelets")]:
            if col not in d or d[col].notna().sum() < 30: continue
            q = d.dropna(subset=[col]); r = spearmanr(q.score, q[col])
            pr = partial_spearman(q.score, q[col], q.age)
            say(f"  score vs {lab:<10} rho={r.correlation:+.3f} (p={r.pvalue:.2g}, n={len(q)})   "
                f"partial (age) {pr[0]:+.3f} (p={pr[1]:.2g})")
            out[col] = [float(r.correlation), float(r.pvalue), float(pr[0]), float(pr[1]), int(len(q))]
        q = d.dropna(subset=["meld"])
        if q.meld.ge(15).sum() >= 10 and q.meld.lt(15).sum() >= 10:
            rr = auc_ci((q.meld >= 15).astype(int), q.score, BOOT)
            say(f"  MELD >= 15 vs < 15: {fmt(rr)}")
            out["auc_meld15"] = list(map(float, rr[:3]))
        t = pd.qcut(q.meld.rank(method="first"), 3, labels=["low", "mid", "high"])
        say("  median score by MELD tertile: " + "  ".join(f"{k} {q.score[t == k].median():.3f}" for k in ["low", "mid", "high"]))
        R[f"meld_{nm}"] = out


DISPO = [("discharged", ["Discharge"]), ("observation", ["Observation"]), ("admitted (ward or ICU)", ["Inpatient", "ICU"]),
         ("any admission", ["Inpatient", "ICU", "Observation"])]


def cv_prob(D, cols, seed=0):
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    cols = [c for c in cols if c in D.columns and D[c].notna().any()]
    X = D[cols].apply(pd.to_numeric, errors="coerce")
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    return cross_val_predict(m, X, D.y.astype(int), method="predict_proba",
                             cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]


def within_strata_auc(y, s, g, n=BOOT, seed=0):
    y, s, g = map(np.asarray, (y, s, g))
    def one(idx):
        num = den = 0.0
        for k in np.unique(g[idx]):
            j = idx[g[idx] == k]; yy = y[j]
            if yy.min() == yy.max(): continue
            w = yy.sum() * (len(yy) - yy.sum()); num += roc_auc_score(yy, s[j]) * w; den += w
        return num / den if den else np.nan
    est = one(np.arange(len(y))); rng = np.random.default_rng(seed)
    bs = [one(rng.integers(0, len(y), len(y))) for _ in range(max(200, n // 4))]
    return est, *np.nanpercentile(bs, [2.5, 97.5])


def pop_by_dispo():
    V = pd.read_csv(os.path.join(MC, "visits.csv"), usecols=["MRN", "CSN", "ED_dispo"])
    Wf = pd.read_csv(os.path.join(MC, "waveform_summary.csv"), usecols=["CSN", "Type", "Duration"])
    pl = Wf[Wf.Type == "Pleth"].groupby("CSN").Duration.sum().rename("pleth_s")
    V = V.merge(pl, left_on="CSN", right_index=True).sort_values("pleth_s", ascending=False).drop_duplicates("MRN")
    Co = pd.read_csv(os.path.join(DERIVED, "mcmed_cohort_visits.csv"), usecols=["MRN", "grp", "ED_dispo"])
    Co = Co[Co.grp == "cirrhosis"].drop_duplicates("MRN")
    return V.ED_dispo.value_counts(), Co.ED_dispo.value_counts()


def s_dispo():
    head("7. ED disposition strata (MC-MED)")
    S = sets()
    try:
        pop, cir = pop_by_dispo()
    except Exception as e:
        pop = cir = None; say(f"  [!] source prevalence not available: {e}")
    VIT = ["age", "male", "Triage_HR", "Triage_RR", "Triage_SpO2", "Triage_SBP", "Triage_DBP"]
    for nm in ["MC-MED internal", "MIMIC->MC-MED frozen"]:
        if nm not in S: continue
        D = S[nm].copy()
        if "ED_dispo" not in D.columns:
            D = D.merge(pd.read_csv(os.path.join(DERIVED, "mcmed_patients.csv"), usecols=["CSN", "ED_dispo"]),
                        left_on="pid", right_on="CSN", how="left", suffixes=("", "_"))
        sub(f"{nm}   n={len(D):,} ({int(D.y.sum())}+)  AUC {roc_auc_score(D.y, D.score):.3f}")
        D["dgrp"] = D.ED_dispo.map({"Discharge": "discharge", "Observation": "observation", "Inpatient": "inpatient", "ICU": "ICU"}).fillna("other")
        w = within_strata_auc(D.y, D.score, D.dgrp)
        say(f"  within-stratum AUC (4 strata): {w[0]:.3f} [{w[1]:.3f}, {w[2]:.3f}]")
        say(f"  disposition alone: AUC {roc_auc_score(D.y, D.dgrp.map({'discharge':0,'observation':1,'inpatient':2,'ICU':3,'other':0})):.3f}")
        say(f"  {'disposition':<22}{'n (+)':>14}{'cases':>9}{'model AUC [95% CI]':>28}{'triage+age+sex':>18}{'combined':>10}{'gain':>9}")
        out = {}
        for lab, codes in DISPO:
            q = D[D.ED_dispo.isin(codes)].reset_index(drop=True)
            if q.y.nunique() < 2 or q.y.sum() < 20:
                say(f"  {lab:<22}too few"); continue
            r = auc_ci(q.y, q.score, BOOT)
            pv = cv_prob(q, VIT); q["_s"] = logit(np.clip(q.score, EPS, 1 - EPS)); pm = cv_prob(q, VIT + ["_s"])
            say(f"  {lab:<22}{len(q):>8} ({int(q.y.sum()):>4}){q.y.mean():>9.1%}   {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]"
                f"{roc_auc_score(q.y, pv):>18.3f}{roc_auc_score(q.y, pm):>10.3f}{roc_auc_score(q.y, pm) - roc_auc_score(q.y, pv):>+9.3f}")
            out[lab] = dict(n=len(q), n_pos=int(q.y.sum()), auc=list(map(float, r[:3])),
                            vitals=float(roc_auc_score(q.y, pv)), combined=float(roc_auc_score(q.y, pm)))
        R[f"dispo_{nm}"] = dict(within=list(map(float, w)), strata=out)
        if nm != "MC-MED internal" or pop is None:
            continue
        for lab, codes in DISPO[2:] + DISPO[:1]:
            npop = int(pop.reindex(codes).fillna(0).sum()); ncir = int(cir.reindex(codes).fillna(0).sum())
            q = D[D.ED_dispo.isin(codes)].reset_index(drop=True)
            if npop == 0 or q.y.nunique() < 2: continue
            prev = ncir / npop
            sub(f"MC-MED internal / {lab}: {ncir} of {npop:,} with cirrhosis ({prev:.2%})")
            r = np.clip(q.score.to_numpy(), 0.5 / len(q), 1 - 0.5 / len(q))
            pc = m10.cross_calibrate(r, q.y.to_numpy().astype(int), q.fold.to_numpy(), "platt")
            R[f"dispo_utility_{lab}"] = utility_block(lab, q.y.to_numpy().astype(int), pc, prev)
        q = D[D.ED_dispo.isin(DISPO[3][1])].reset_index(drop=True)
        sub(f"MC-MED internal / admitted: laboratory markers (n={len(q):,}, {int(q.y.sum())}+)")
        R["dispo_labs_admitted"] = labs_h2h("admitted", q, labs_for("mcmed"))


def ensure_fold(D, seed=0):
    D = D.reset_index(drop=True).copy()
    if "fold" not in D.columns:
        D["fold"] = -1
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(D, D.y)):
            D.loc[te, "fold"] = k
    return D


def alb_adjust(nm, D, L):
    D = ensure_fold(D)
    d = D[["pid", "y", "score", "fold", "age", "male"]].merge(L, on="pid", how="left")
    for c in ["ast", "alt", "plt", "bili", "inr", "alb"]:
        if c not in d.columns: d[c] = np.nan
    d["age"] = d.age.clip(upper=90)
    d = m11.add_scores(d).reset_index(drop=True)
    d["ppg_l"] = logit(np.clip(d.score.to_numpy(), 0.5 / len(d), 1 - 0.5 / len(d)))
    d["log_alb"] = np.log(d.alb.clip(lower=0.5))
    a = d.alb.notna()
    say(f"  albumin available: {a.mean():.1%} (cirrhosis {a[d.y==1].mean():.1%} / controls {a[d.y==0].mean():.1%})   "
        f"median {d.alb[d.y==1].median():.1f} vs {d.alb[d.y==0].median():.1f} g/dL")
    out = {}
    s = d[a & d.plt.notna()].reset_index(drop=True)
    if len(s) < 100 or s.y.nunique() < 2:
        say("  too few patients"); return None
    ys, fs = s.y.to_numpy().astype(int), s.fold.to_numpy()
    say(f"\n  A. albumin and platelets available: {len(s):,} ({int(ys.sum())}+)")
    M = {"PPG": s.ppg_l.to_numpy(), "albumin": -s.alb.to_numpy(),
         "PPG+albumin": m11.cv_logit(s, ["ppg_l", "log_alb"], ys, fs, 5),
         "albumin+platelets": m11.cv_logit(s, ["log_alb", "log_plt"], ys, fs, 5),
         "PPG+albumin+platelets": m11.cv_logit(s, ["ppg_l", "log_alb", "log_plt"], ys, fs, 5)}
    for k, v in M.items():
        v = np.nan_to_num(v, nan=np.nanmedian(v)); M[k] = v
        say(f"    {k:<20}{fmt(auc_ci(ys, v, BOOT))}")
    for x, w in [("PPG+albumin", "albumin"), ("PPG+albumin+platelets", "albumin+platelets"), ("PPG+albumin", "PPG")]:
        o, lo, hi, p = paired_delta(ys, M[x], M[w], BOOT)
        say(f"    {x:<22} vs {w:<18} {o:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={min(p, 1):.3g}")
        out[f"{x} vs {w}"] = [float(o), float(lo), float(hi), float(min(p, 1))]
    f = s[s.fib4.notna()].reset_index(drop=True)
    if len(f) >= 100 and f.y.nunique() == 2:
        yf, ff = f.y.to_numpy().astype(int), f.fold.to_numpy()
        b0 = m11.cv_logit(f, ["log_fib4", "log_alb"], yf, ff, 5); b1 = m11.cv_logit(f, ["ppg_l", "log_fib4", "log_alb"], yf, ff, 5)
        b0, b1 = [np.nan_to_num(v, nan=np.nanmedian(v)) for v in (b0, b1)]
        o, lo, hi, p = paired_delta(yf, b1, b0, BOOT)
        say(f"\n  B. FIB-4 also available: {len(f):,}   FIB-4+albumin {roc_auc_score(yf, b0):.3f} -> +PPG {roc_auc_score(yf, b1):.3f}"
            f"   {o:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={min(p, 1):.3g}")
        out["PPG+FIB-4+albumin vs FIB-4+albumin"] = [float(o), float(lo), float(hi), float(min(p, 1))]
    z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
    X0 = np.column_stack([np.ones(len(s)), z(s.ppg_l), z(s.age), s.male])
    X1 = np.column_stack([X0, z(s.log_alb)]); X2 = np.column_stack([X1, z(s.log_plt)])
    say("\n  C. OR per SD of the PPG score:")
    for lab, X in [("age+sex", X0), ("+albumin", X1), ("+albumin+platelets", X2)]:
        b, se, _ = logistic_mle(X, ys.astype(float))
        say(f"    adjusted for {lab:<20} OR {np.exp(b[1]):.2f} [{np.exp(b[1]-1.96*se[1]):.2f}, {np.exp(b[1]+1.96*se[1]):.2f}]"
            f"   p={2*norm.sf(abs(b[1]/se[1])):.2g}" + (f"   (albumin OR {np.exp(b[4]):.2f})" if X.shape[1] > 4 else ""))
        out[f"OR_{lab}"] = [float(np.exp(b[1])), float(se[1])]
    say("\n  D. AUC by albumin:")
    for lab, m in [("albumin >= 3.5", s.alb >= 3.5), ("albumin < 3.5", s.alb < 3.5)]:
        q = s[m]
        rr = auc_ci(q.y, q.ppg_l, BOOT) if q.y.nunique() == 2 and q.y.sum() >= 10 else None
        say(f"    {lab:<18}" + (fmt(rr) if rr else f"too few ({int(q.y.sum())}+ / {len(q)})"))
        if rr: out[f"auc_{lab}"] = list(map(float, rr[:3]))
    return out


def s_albumin():
    head("8. Adjustment for albumin")
    for nm, D in sets().items():
        sub(f"{nm}   n={len(D):,} ({int(D.y.sum())}+)")
        R[f"albumin_{nm}"] = alb_adjust(nm, D, labs_for(db_of(nm)))


def main():
    global BOOT
    ap = argparse.ArgumentParser()
    ALL = "summary,table1,utility,external,labs,age,meld,dispo,albumin"
    ap.add_argument("--only", default=ALL)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(); BOOT = a.boot
    os.makedirs(OUT, exist_ok=True)
    todo = a.only.split(",")
    say(f"sections: {todo}   bootstrap {BOOT}")
    for k, f in [("summary", s_summary), ("table1", s_table1), ("utility", s_utility),
                 ("external", s_external), ("labs", s_labs), ("age", s_age), ("meld", s_meld),
                 ("dispo", s_dispo), ("albumin", s_albumin)]:
        if k in todo:
            try:
                f()
            except Exception as e:
                import traceback; say(f"\n[!] section {k} failed: {e!r}"); say(traceback.format_exc())
    suf = "" if todo == ALL.split(",") else "_" + "_".join(todo)
    json.dump(R, open(os.path.join(OUT, f"result_21{suf}.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    open(os.path.join(OUT, f"report_21{suf}.txt"), "w", encoding="utf-8").write("\n".join(_LOG))
    print(f"\n[saved] {os.path.join(OUT, f'report_21{suf}.txt')}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore"); main()
