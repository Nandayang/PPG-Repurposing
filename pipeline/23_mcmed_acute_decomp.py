"""MC-MED, exploratory: decompensation defined from index-visit evidence instead of past medical history, evaluated with
the existing patient-level predictions (no retraining).

A: decompensation code as primary diagnosis, or ascites on imaging at this visit (negated mentions excluded).
B: A or related orders (paracentesis, octreotide, lactulose, rifaximin, terlipressin). C: B or GI bleeding / jaundice.
Verified compensated: abdominal imaging at this visit without ascites and none of the above.
Usage: python 23_mcmed_acute_decomp.py [--infer]
Outputs: results/acute/report_23.txt, result_23.json; derived/mcmed_acute_decomp.csv
"""
import os, re, json, argparse, warnings, importlib.util
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.environ.get("PPG_ROOT", ".")
CACHE   = os.path.join(ROOT, "cache")
DERIVED = os.path.join(ROOT, "derived")
RES     = os.path.join(ROOT, "results")
OUT     = os.path.join(RES, "acute")
MC      = os.environ.get("MCMED_DIR", "data/mc-med")
MIN_WIN = 10

_LOG = []
def say(s=""):
    print(s, flush=True); _LOG.append(str(s))
def head(t):
    say(); say("=" * 78); say(t); say("=" * 78)


def load_module(fn, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


mst = load_module("stats_common.py", "mst")
auc_ci, fmt, paired_delta = mst.auc_ci, mst.fmt, mst.paired_delta


def norm(s):
    return "" if pd.isna(s) else re.sub(r"[^A-Z0-9]", "", str(s).upper())


DX10_STRICT = ("R188", "K7031", "K7011", "I8501", "I8511", "K72", "K767", "K652", "K7682")
DX9_STRICT  = ("78959", "4560", "45620", "5722", "5724", "56723")
DX10_BROAD  = ("K920", "K921", "K922", "R17")
DX9_BROAD   = ("578", "7824")
ABD_STUDY   = r"abd|abdom|pelvi|liver|hepat|ruq|right upper|paracent|a/p|chest/abd"
NEG = re.compile(r"\b(no|without|negative for|free of|absence of|resolution of|resolved|no evidence of|"
                 r"no significant|no new|no large|no definite|not)\b[^.;\n]{0,40}\bascit", re.I)
NEG_AFTER = re.compile(r"ascit\w*[^.;\n]{0,20}\b(resolved|has resolved|no longer)", re.I)
ORD = r"paracent|octreotide|sandostatin|lactulose|rifaximin|xifaxan|terlipressin"


def ascites_positive(txt):
    if not isinstance(txt, str) or "ascit" not in txt.lower():
        return False
    for sent in re.split(r"[.;\n]", txt):
        if "ascit" in sent.lower() and not NEG.search(sent) and not NEG_AFTER.search(sent):
            return True
    return False


def build_labels(P):
    cs = set(P.CSN)
    V = pd.read_csv(os.path.join(MC, "visits.csv"), usecols=["CSN", "Dx_ICD9", "Dx_ICD10"], low_memory=False)
    V = V[V.CSN.isin(cs)].copy()
    V["c10"] = V.Dx_ICD10.map(norm); V["c9"] = V.Dx_ICD9.map(norm)
    V["dx_strict"] = V.c10.str.startswith(DX10_STRICT) | V.c9.str.startswith(DX9_STRICT)
    V["dx_broad"] = V.c10.str.startswith(DX10_BROAD) | V.c9.str.startswith(DX9_BROAD)
    say(f"  primary diagnosis: decompensation {int(V.dx_strict.sum())}; GI bleeding/jaundice {int(V.dx_broad.sum())}")
    top = V[V.dx_strict | V.dx_broad].c10.value_counts().head(12)
    say("    codes: " + "  ".join(f"{k} x{v}" for k, v in top.items()))
    R = pd.read_csv(os.path.join(MC, "rads.csv"), usecols=["CSN", "Study", "Impression"], dtype={"Impression": str, "Study": str})
    R = R[R.CSN.isin(cs)].copy()
    R["abd"] = R.Study.fillna("").str.lower().str.contains(ABD_STUDY, regex=True)
    R["asc"] = R.Impression.map(ascites_positive) | R.Study.fillna("").str.lower().str.contains("paracent")
    R["mention"] = R.Impression.fillna("").str.lower().str.contains("ascit")
    g = R.groupby("CSN").agg(img_abd=("abd", "max"), img_ascites=("asc", "max"))
    say(f"  imaging: {R.CSN.nunique()} visits, abdominal {int(g.img_abd.sum())}; "
        f"ascites mentioned {int(R[R.mention].CSN.nunique())}, positive {int(g.img_ascites.sum())}")
    sn = R[R.mention].copy()
    sn["sentence"] = sn.Impression.map(lambda t: " | ".join(s.strip() for s in re.split(r"[.;\n]", t) if "ascit" in s.lower())[:300])
    sn[["CSN", "Study", "asc", "sentence"]].to_csv(os.path.join(DERIVED, "mcmed_ascites_snippets_LOCAL.csv"),
                                                   index=False, encoding="utf-8-sig")
    say("    sentences written to derived/mcmed_ascites_snippets_LOCAL.csv for review")
    ordf = pd.Series(False, index=pd.Index(sorted(cs), name="CSN"))
    try:
        cols = pd.read_csv(os.path.join(MC, "orders.csv"), nrows=5).columns.tolist()
        say(f"  orders.csv columns: {cols}")
        txt = [c for c in cols if re.search(r"name|desc|proc|display|med", c, re.I) and c.upper() != "CSN"]
        if "CSN" not in cols or not txt:
            raise ValueError("no CSN or order name column")
        hit, names = set(), {}
        for ch in pd.read_csv(os.path.join(MC, "orders.csv"), usecols=["CSN"] + txt, chunksize=2_000_000,
                              low_memory=False, dtype=str):
            ch["CSN"] = pd.to_numeric(ch.CSN, errors="coerce"); ch = ch[ch.CSN.isin(cs)]
            t = ch[txt].fillna("").agg(" ".join, axis=1).str.lower()
            m = t.str.contains(ORD, regex=True); hit |= set(ch.loc[m, "CSN"])
            for k in t[m].str.extract(f"({ORD})", expand=False).dropna(): names[k] = names.get(k, 0) + 1
        ordf = ordf | ordf.index.isin(hit)
        say(f"  orders ({'/'.join(txt)}): {len(hit)} visits  " + "  ".join(f"{k} x{v}" for k, v in sorted(names.items(), key=lambda x: -x[1])))
    except Exception as e:
        say(f"  [!] orders not used ({e})")
    L = P[["CSN"]].merge(V[["CSN", "dx_strict", "dx_broad"]], on="CSN", how="left") \
                  .merge(g, left_on="CSN", right_index=True, how="left")
    L["ord"] = L.CSN.map(ordf).fillna(False)
    for c in ["dx_strict", "dx_broad", "img_abd", "img_ascites", "ord"]:
        L[c] = L[c].fillna(False).astype(bool)
    L["acute_A"] = L.dx_strict | L.img_ascites
    L["acute_B"] = L.acute_A | L.ord
    L["acute_C"] = L.acute_B | L.dx_broad
    L["verified_comp"] = L.img_abd & ~L.acute_C
    return L


def scores(P, infer=False):
    S = P[["CSN"]].copy()
    f = os.path.join(RES, "es", "patient_oof_mcmed_cirrhosis_ssl.csv")
    if os.path.exists(f):
        S = S.merge(pd.read_csv(f)[["CSN", "score"]].rename(columns={"score": "cirr_int"}), on="CSN", how="left")
    f = os.path.join(RES, "es", "patient_oof_mcmed_decomp_ssl.csv")
    if os.path.exists(f):
        S = S.merge(pd.read_csv(f)[["CSN", "score"]].rename(columns={"score": "dec_int"}), on="CSN", how="left")
    f = os.path.join(RES, "mcmed", "patient_preds.csv")
    if os.path.exists(f):
        S = S.merge(pd.read_csv(f)[["CSN", "s_es"]].rename(columns={"s_es": "cirr_ext"}), on="CSN", how="left")
    fp = os.path.join(OUT, "patient_dec_ext.csv")
    if infer and not os.path.exists(fp):
        import torch
        m14 = load_module("14_mcmed_external.py", "m14")
        m14.MODELS["dec_ext"] = ("es_mimic_decomp_ssl_fold{f}.pt", "2d", "", None, 5)
        I = pd.read_csv(os.path.join(CACHE, "mcmed_index.csv"))
        X = np.load(os.path.join(CACHE, "X_beat_mcmed_aligned.npy"), mmap_mode="r")
        M = pd.read_csv(os.path.join(CACHE, "beat_mcmed_aligned_meta.csv"))
        ok = (M.ok_beat == 1).to_numpy() & I.CSN.isin(set(P.CSN)).to_numpy()
        rows = np.flatnonzero(ok); nb = M.n_beat.fillna(0).to_numpy().astype(np.int64)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        say(f"  [infer] MIMIC decompensation models, {len(rows):,} windows ({dev})")
        W = m14.infer("dec_ext", X, rows, nb, dev)
        d = pd.DataFrame({"CSN": I.CSN.to_numpy()[rows], "p": np.nanmean(W, 1)})
        d.groupby("CSN").p.mean().rename("dec_ext").reset_index().to_csv(fp, index=False)
    if os.path.exists(fp):
        S = S.merge(pd.read_csv(fp), on="CSN", how="left")
    return S


def cv_prob(D, cols, y, seed=0):
    cols = [c for c in cols if c in D.columns and D[c].notna().any()]
    X = D[cols].apply(pd.to_numeric, errors="coerce")
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    return cross_val_predict(m, X, y, method="predict_proba", cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    head("MC-MED: decompensation at the index visit (exploratory)")
    P = pd.read_csv(os.path.join(DERIVED, "mcmed_patients.csv"))
    P = P[(P.in_cirrhosis == 1) & (P.n_win.fillna(0) >= MIN_WIN)].copy()
    P["age"] = P.Age; P["male"] = (P.Gender == "M").astype(int)
    P["hist_decomp"] = P.dx_decomp.astype(str).isin(["True", "1", "1.0"])
    say(f"  cirrhosis (>= {MIN_WIN} windows): {len(P)}; history of decompensation {int(P.hist_decomp.sum())}")

    L = build_labels(P)
    D = P.merge(L, on="CSN").merge(scores(P, a.infer), on="CSN", how="left")
    D.drop(columns=[c for c in ["Impression"] if c in D.columns]).to_csv(
        os.path.join(DERIVED, "mcmed_acute_decomp.csv"), index=False)

    head("1. Labels")
    for k in ["acute_A", "acute_B", "acute_C", "verified_comp"]:
        say(f"  {k:<14} {int(D[k].sum()):>4} / {len(D)}")
    ct = pd.crosstab(D.hist_decomp.map({True: "history: decompensated", False: "history: compensated"}),
                     D.acute_A.map({True: "this visit: A", False: "this visit: none"}))
    say("  vs history-based label:\n" + "\n".join("    " + l for l in ct.to_string().splitlines()))

    SC = [c for c in ["cirr_int", "dec_int", "cirr_ext", "dec_ext"] if c in D.columns and D[c].notna().any()]
    NAMES = {"cirr_int": "MC-MED cirrhosis model (OOF)", "dec_int": "MC-MED decompensation model (OOF)",
             "cirr_ext": "MIMIC cirrhosis model (frozen)", "dec_ext": "MIMIC decompensation model (frozen)"}
    VIT = ["age", "male", "Triage_HR", "Triage_RR", "Triage_SpO2", "Triage_SBP", "Triage_DBP"]
    R = {"n": len(D)}
    designs = [("history", "history of decompensation", D, "hist_decomp"),
               ("A", "A: primary diagnosis or ascites on imaging", D, "acute_A"),
               ("B", "B: A or related orders", D, "acute_B"),
               ("C", "C: B or GI bleeding / jaundice", D, "acute_C"),
               ("A_vs_verified", "A vs verified compensated", D[D.acute_A | D.verified_comp], "acute_A")]
    for key, title, d, lab in designs:
        d = d.dropna(subset=SC[:1]).reset_index(drop=True); y = d[lab].astype(int).to_numpy()
        head(f"2. {title}   n={len(d)} ({int(y.sum())}+ / {int((1-y).sum())}-)")
        if y.sum() < 20 or (1 - y).sum() < 20:
            say("  too few"); continue
        out = {}
        for c in SC:
            q = d[c].notna().to_numpy()
            r = auc_ci(y[q], d.loc[q, c], a.boot); out[c] = list(map(float, r[:3]))
            say(f"  {NAMES[c]:<34}{fmt(r)}")
        pv = cv_prob(d, VIT, y)
        for nm, cols in [("age+sex", ["age", "male"]), ("pulse rate", ["ppg_hr"]), ("triage vitals+age+sex", VIT)]:
            p = cv_prob(d, cols, y); say(f"  {nm:<34}{fmt(auc_ci(y, p, a.boot))}")
        for c in SC:
            dd = d.assign(_s=np.log(np.clip(d[c], 1e-6, 1 - 1e-6) / (1 - np.clip(d[c], 1e-6, 1 - 1e-6))))
            pm = cv_prob(dd, VIT + ["_s"], y)
            o, lo, hi, p = paired_delta(y, pm, pv, a.boot)
            say(f"  vitals + {c:<10} {roc_auc_score(y, pm):.3f}  vs vitals {roc_auc_score(y, pv):.3f}   "
                f"increment {o:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={min(p, 1):.3g}")
            out[f"inc_{c}"] = [float(o), float(lo), float(hi), float(min(p, 1))]
        R[key] = out

    json.dump(R, open(os.path.join(OUT, "result_23.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(os.path.join(OUT, "report_23.txt"), "w", encoding="utf-8").write("\n".join(_LOG))
    print(f"\n[saved] {os.path.join(OUT, 'report_23.txt')}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore"); main()
