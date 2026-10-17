"""Cohorts for the multilabel model (cirrhosis, heart failure, sepsis, end-stage renal disease).

Chronic conditions are patient-level; sepsis is taken from the index visit or admission only. Controls have none of
the four conditions and no chronic liver disease. One visit or admission per patient (sepsis visit first, then the one
with the most signal).

MIMIC:  python 17_multilabel_prep.py --data mimic   -> cache/X_ppg_ml.npy, cache/index_ml.csv
MC-MED: python 17_multilabel_prep.py --data mcmed   -> derived/ml_mcmed_cohort.csv, derived/ml_mcmed_download_csns.txt
        then 05_mcmed_download.py --csns ..., 13_mcmed_cache.py --cohort ... --suffix _ml
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import re, argparse, time, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

DERIVED = os.path.join(os.environ.get("PPG_ROOT", "."), "derived")
CACHE   = os.path.join(os.environ.get("PPG_ROOT", "."), "cache")
MC      = os.environ.get("MCMED_DIR", "data/mc-med")
PPG_DIR = os.environ.get("MIMIC_PPG_DIR", "data/mimic-iii-ext-ppg")
MIMIC   = os.environ.get("MIMIC_DIR", "data/mimic-iii-clinical-database-1.4")
L = 3750
DIS = ["cirrhosis", "hf", "sepsis", "esrd"]


def norm(s):
    return "" if pd.isna(s) else re.sub(r"[^A-Z0-9]", "", str(s).upper())


I9 = {"cirrhosis": ("5712", "5715", "5716"),
      "hf": ("428",), "esrd": ("5856",), "sepsis": ("99591", "99592", "78552"),
      "decomp": ("7895", "78959", "4560", "45620", "5722", "5724", "56723"),
      "liver_any": ("570", "571", "572", "573", "070", "4560", "4561", "7895", "V427", "1550")}
I10 = {"cirrhosis": ("K703", "K717", "K743", "K744", "K745", "K746"),
       "hf": ("I50",), "esrd": ("N186",), "sepsis": ("A40", "A41", "R652"),
       "decomp": ("R18", "I850", "I851", "I864", "K72", "K767", "K652"),
       "liver_any": ("K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77", "B18",
                     "I850", "I851", "I864", "R18", "Z944", "C22")}


def hit(codes, pref):
    return codes.str.startswith(tuple(pref), na=False)


def prep_mcmed(a):
    V = pd.read_csv(os.path.join(MC, "visits.csv"), low_memory=False)
    V["c10"] = V.Dx_ICD10.map(norm); V["c9"] = V.Dx_ICD9.map(norm)
    V["v_sepsis"] = hit(V.c10, I10["sepsis"]) | hit(V.c9, I9["sepsis"])
    chronic = {k: set() for k in ["cirrhosis", "hf", "esrd", "decomp", "liver_any", "sepsis_any"]}
    for k in ["cirrhosis", "hf", "esrd", "decomp", "liver_any"]:
        chronic[k] |= set(V.loc[hit(V.c10, I10[k]) | hit(V.c9, I9[k]), "MRN"])
    chronic["sepsis_any"] |= set(V.loc[V.v_sepsis, "MRN"])
    print("[pmh] reading ...")
    for ch in pd.read_csv(os.path.join(MC, "pmh.csv"), usecols=["MRN", "CodeType", "Code"],
                          chunksize=1_000_000, low_memory=False):
        c = ch.Code.map(norm); is10 = ch.CodeType.astype(str).str.contains("10")
        for k in ["cirrhosis", "hf", "esrd", "decomp", "liver_any"]:
            chronic[k] |= set(ch.loc[(is10 & hit(c, I10[k])) | (~is10 & hit(c, I9[k])), "MRN"])
        chronic["sepsis_any"] |= set(ch.loc[(is10 & hit(c, I10["sepsis"])) | (~is10 & hit(c, I9["sepsis"])), "MRN"])

    W = pd.read_csv(os.path.join(MC, "waveform_summary.csv"))
    P = W[W.Type == "Pleth"].groupby("CSN").Duration.sum().rename("pleth_s")
    V = V.merge(P, left_on="CSN", right_index=True, how="inner")
    V["Age"] = pd.to_numeric(V.Age, errors="coerce")
    V = V.sort_values(["MRN", "v_sepsis", "pleth_s"], ascending=[True, False, False]).drop_duplicates("MRN")
    for k in ["cirrhosis", "hf", "esrd"]:
        V[f"y_{k}"] = V.MRN.isin(chronic[k]).astype(int)
    V["y_sepsis"] = V.v_sepsis.astype(int)
    V["y_decomp"] = (V.y_cirrhosis == 1) & V.MRN.isin(chronic["decomp"])
    V["control"] = (~V.MRN.isin(chronic["liver_any"] | chronic["hf"] | chronic["esrd"] | chronic["sepsis_any"])
                    & ~V.v_sepsis).astype(int)
    V["any_pos"] = V[[f"y_{k}" for k in DIS]].max(1)
    pos = V[V.any_pos == 1]
    print(f"[mcmed] patients with Pleth {len(V):,}   any condition {len(pos):,}   eligible controls {int(V.control.sum()):,}")
    for k in DIS:
        print(f"   {k:<10} {int(V[f'y_{k}'].sum()):>6,}   median age {V[V[f'y_{k}']==1].Age.median():.0f}")

    ctl = V[(V.control == 1) & V.Age.notna()].copy()
    bins = [0, 30, 40, 50, 60, 70, 80, 200]
    pos_d = pd.crosstab(pd.cut(pos.Age, bins, right=False), pos.Gender, normalize=True)
    ctl["cell"] = pd.cut(ctl.Age, bins, right=False).astype(str) + "|" + ctl.Gender.astype(str)
    take = []
    rng = np.random.default_rng(0)
    for (ab, g), frac in pos_d.stack().items():
        n = int(round(frac * a.n_ctrl)); cand = ctl[ctl.cell == f"{ab}|{g}"]
        if n and len(cand):
            take.append(cand.sample(min(n, len(cand)), random_state=int(rng.integers(1e9))))
    ctl = pd.concat(take)
    print(f"[mcmed] controls {len(ctl):,} (median age {ctl.Age.median():.0f}; cases {pos.Age.median():.0f})")

    coh = pd.concat([pos, ctl.drop(columns="cell")]).drop_duplicates("CSN")
    rows = []
    for g in DIS:
        rows.append(coh[coh[f"y_{g}"] == 1].assign(grp=g))
    rows.append(coh[coh.control == 1].assign(grp="control"))
    out = pd.concat(rows)
    out["cirrhosis"] = out.y_cirrhosis.astype(bool); out["hf"] = out.y_hf.astype(bool)
    out["sepsis"] = out.y_sepsis.astype(bool); out["esrd"] = out.y_esrd.astype(bool)
    out["decomp"] = out.y_decomp.astype(bool); out["comp"] = out.cirrhosis & ~out.decomp
    out["control"] = out.control.astype(bool)
    keep = ["MRN", "CSN", "Age", "Gender", "Triage_HR", "Triage_RR", "Triage_SpO2", "Triage_SBP",
            "Triage_DBP", "Triage_acuity", "ED_dispo", "pleth_s", "grp",
            "cirrhosis", "decomp", "comp", "hf", "sepsis", "esrd", "control"]
    out[keep].to_csv(os.path.join(DERIVED, "ml_mcmed_cohort.csv"), index=False)

    have = []
    for c in coh.CSN.astype(int).astype(str):
        have.append(os.path.isdir(os.path.join(MC, "data", c[-3:], c, "Pleth")))
    coh["on_disk"] = have
    miss = coh[~coh.on_disk]
    with open(os.path.join(DERIVED, "ml_mcmed_download_csns.txt"), "w") as f:
        f.write("\n".join(miss.CSN.astype(int).astype(str)) + "\n")
    print(f"\n[out] derived/ml_mcmed_cohort.csv   {coh.CSN.nunique():,} visits")
    print(f"      waveforms present {int(coh.on_disk.sum()):,}, to download {len(miss):,} "
          f"(derived/ml_mcmed_download_csns.txt)")
    print("\nlabel overlap:")
    Y = coh[[f"y_{k}" for k in DIS]]
    print((Y.T @ Y).rename(index=lambda s: s[2:], columns=lambda s: s[2:]).to_string())


def _read(args):
    i, rel = args
    import wfdb
    try:
        r = wfdb.rdrecord(os.path.join(PPG_DIR, os.path.normpath(rel)))
        names = [n.upper() for n in r.sig_name]
        if "PLETH" not in names:
            return i, None
        x = r.p_signal[:, names.index("PLETH")]
        return (i, x[:L].astype(np.float16)) if len(x) >= L else (i, None)
    except Exception:
        return i, None


def spectral_hr(X, fs=125.0):
    f = np.fft.rfftfreq(L, 1 / fs); b = (f >= 0.6) & (f <= 3.0); w = np.hanning(L).astype(np.float32)
    x = np.nan_to_num(X.astype(np.float32)); x = (x - x.mean(1, keepdims=True)) * w
    P = np.abs(np.fft.rfft(x, axis=1)) ** 2; Pb = P[:, b]
    h = f[b][Pb.argmax(1)] * 60; h[Pb.max(1) / (Pb.mean(1) + 1e-12) < 3] = np.nan
    return h


def prep_mimic(a):
    seg = pd.read_parquet(os.path.join(DERIVED, "segments_index.parquet"),
                          columns=["subject_id", "hadm_id", "folder_path", "start_segment", "ppg_all_good",
                                   "age", "gender", "median_30s_hr", "median_30s_sbp", "median_30s_dbp",
                                   "median_30s_rr", "event_rhythm", "clinical_information_system"])
    seg = seg[seg.ppg_all_good == 1].copy()
    dx = pd.read_csv(os.path.join(MIMIC, "DIAGNOSES_ICD.csv"), usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
                     dtype={"ICD9_CODE": "string"})
    dx["c"] = dx.ICD9_CODE.fillna("").str.strip()
    pat = {k: set(dx.loc[hit(dx.c, I9[k]), "SUBJECT_ID"]) for k in ["cirrhosis", "hf", "esrd", "decomp", "liver_any", "sepsis"]}
    sep_hadm = set(dx.loc[hit(dx.c, I9["sepsis"]), "HADM_ID"])

    seg["s_hadm"] = seg.hadm_id.isin(sep_hadm)
    n = seg.groupby(["subject_id", "hadm_id"]).agg(n=("folder_path", "size"), s=("s_hadm", "first")).reset_index()
    n = n[n.n >= 10].sort_values(["subject_id", "s", "n"], ascending=[True, False, False]).drop_duplicates("subject_id")
    seg = seg.merge(n[["subject_id", "hadm_id"]], on=["subject_id", "hadm_id"])
    P = n.rename(columns={"s": "y_sepsis"})[["subject_id", "hadm_id", "y_sepsis"]]
    for k in ["cirrhosis", "hf", "esrd"]:
        P[f"y_{k}"] = P.subject_id.isin(pat[k]).astype(int)
    P["y_sepsis"] = P.y_sepsis.astype(int)
    P["y_decomp"] = ((P.y_cirrhosis == 1) & P.subject_id.isin(pat["decomp"])).astype(int)
    P["control"] = (~P.subject_id.isin(pat["liver_any"] | pat["hf"] | pat["esrd"] | pat["sepsis"])).astype(int)
    print(f"[mimic] patients {len(P):,} (>= 10 good segments in the admission), controls {int(P.control.sum()):,}")
    for k in DIS:
        print(f"   {k:<10} {int(P[f'y_{k}'].sum()):>6,}")
    Y = P[[f"y_{k}" for k in DIS]]
    print("label overlap:\n" + (Y.T @ Y).rename(index=lambda s: s[2:], columns=lambda s: s[2:]).to_string())

    seg = seg.sort_values(["subject_id", "start_segment"]).reset_index(drop=True)
    g = seg.groupby("subject_id", sort=False)
    r = g.cumcount().to_numpy(); nn = g.subject_id.transform("size").to_numpy()
    m = np.minimum(nn, a.per_pat)
    seg = seg.loc[np.floor(r * m / nn) > np.floor((r - 1) * m / nn)].reset_index(drop=True)
    if a.test:
        seg = seg[seg.subject_id.isin(seg.subject_id.drop_duplicates().head(40))].reset_index(drop=True)
    N = len(seg); print(f"[seg] {N:,} segments / {seg.subject_id.nunique():,} patients")

    suf = "_test" if a.test else ""
    X = np.lib.format.open_memmap(os.path.join(CACHE, f"X_ppg_ml{suf}.npy"), mode="w+", dtype=np.float16, shape=(N, L))
    ok = np.zeros(N, bool); t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, x in ex.map(_read, list(enumerate(seg.folder_path)), chunksize=64):
            if x is not None:
                X[i] = x; ok[i] = True
            done += 1
            if done % 20000 == 0:
                el = time.time() - t0; print(f"   {done:,}/{N:,}  {el:.0f}s  eta {el/done*(N-done):.0f}s", flush=True)
    X.flush()
    hr = np.full(N, np.nan, np.float32)
    for s in range(0, N, 20000):
        hr[s:s + 20000] = spectral_hr(np.asarray(X[s:s + 20000]))
    seg["ok_ppg"] = ok; seg["ppg_hr"] = hr
    seg = seg.merge(P, on=["subject_id", "hadm_id"], how="left")
    seg.drop(columns=["ppg_all_good", "s_hadm"]).to_csv(os.path.join(CACHE, f"index_ml{suf}.csv"), index=False)
    P.to_csv(os.path.join(DERIVED, f"ml_mimic_patients{suf}.csv"), index=False)
    print(f"\n[done] {time.time()-t0:.0f}s   read {ok.mean():.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=["mcmed", "mimic"])
    ap.add_argument("--n-ctrl", type=int, default=4000, help="MC-MED controls")
    ap.add_argument("--per-pat", type=int, default=100, help="MIMIC segments per patient")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 3)))
    ap.add_argument("--test", action="store_true", help="MIMIC: 40 patients only")
    a = ap.parse_args()
    (prep_mcmed if a.data == "mcmed" else prep_mimic)(a)


if __name__ == "__main__":
    warnings.filterwarnings("ignore"); main()
