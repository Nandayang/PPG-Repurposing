"""MC-MED waveform cache: non-overlapping 30-s Pleth windows at 125 Hz that pass the signal-quality index, up to 100 per
visit spread evenly over time, standardized and stored as float16; plus visit-level covariates.

Outputs: cache/X_mcmed<suffix>.npy, cache/mcmed_index<suffix>.csv, derived/mcmed_patients<suffix>.csv
Usage: python 13_mcmed_cache.py [--cohort derived/ml_mcmed_cohort.csv --suffix _ml] [--test]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob, argparse, time, warnings
from datetime import datetime, timedelta
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

MC      = os.environ.get("MCMED_DIR", "data/mc-med")
WDIR    = os.path.join(MC, "data")
DERIVED = os.path.join(os.environ.get("PPG_ROOT", "."), "derived")
CACHE   = os.path.join(os.environ.get("PPG_ROOT", "."), "cache")
COH     = os.path.join(DERIVED, "mcmed_cohort_visits.csv")
FS, L   = 125.0, 3750
GROUPS  = ["cirrhosis", "control", "hf", "sepsis", "esrd"]


def csn_dir(csn):
    s = str(int(csn))
    return os.path.join(WDIR, s[-3:], s, "Pleth")


def parse_hea(path):
    try:
        Ls = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
        a = Ls[0]
        fs = float(a[2].split("/")[0]); n = int(a[3])
        t0 = None
        if len(a) > 5:
            for f in ("%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S"):
                try:
                    t0 = datetime.strptime(f"{a[5]} {a[4][:15]}", f); break
                except ValueError:
                    pass
        return dict(seg=os.path.basename(path)[:-4], fs=fs, n=n, t0=t0)
    except Exception:
        return None


def sqi_window(x, fs=FS):
    from scipy.signal import find_peaks, butter, filtfilt
    r = dict(nan=float(np.isnan(x).mean()))
    if r["nan"] > 0.05:
        return {**r, "ok": False, "why": "nan"}
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
    r["sat"] = float(((x <= x.min()) | (x >= x.max())).mean())
    if x.std() < 1e-6:
        return {**r, "ok": False, "why": "flat"}
    b, a = butter(2, [0.5 / (fs / 2), 4 / (fs / 2)], "band")
    y = filtfilt(b, a, x)
    pk, _ = find_peaks(y, distance=int(0.33 * fs), prominence=0.5 * y.std())
    if len(pk) < 5:
        return {**r, "ok": False, "why": "few_peaks"}
    ibi = np.diff(pk) / fs; med = np.median(ibi)
    r["hr"] = float(60 / med)
    r["ibi_reg"] = float((np.abs(ibi - med) < 0.2 * med).mean())
    Lb = int(med * fs)
    beats = [y[p - Lb // 3: p + 2 * Lb // 3] for p in pk
             if p - Lb // 3 >= 0 and p + 2 * Lb // 3 <= len(y)]
    if len(beats) >= 3:
        B = np.array(beats); T = B.mean(0)
        r["tmpl_corr"] = float(np.median([np.corrcoef(T, b_)[0, 1] for b_ in B]))
    else:
        r["tmpl_corr"] = 0.0
    r["ok"] = bool((30 <= r["hr"] <= 180) and r["ibi_reg"] >= 0.8 and
                   r["tmpl_corr"] >= 0.9 and r["sat"] < 0.10)
    r["why"] = "" if r["ok"] else "quality"
    return r


def spectral_hr(x, fs=FS, lo=0.6, hi=3.0):
    x = x - x.mean()
    w = np.hanning(len(x))
    f = np.fft.rfftfreq(len(x), 1 / fs); P = np.abs(np.fft.rfft(x * w)) ** 2
    b = (f >= lo) & (f <= hi); Pb = P[b]
    if Pb.max() / (Pb.mean() + 1e-12) < 3.0:
        return np.nan
    return float(f[b][Pb.argmax()] * 60)


def process_visit(args):
    csn, n_cand, per_pat, seed = args
    import wfdb
    d = csn_dir(csn)
    H = [h for h in (parse_hea(p) for p in sorted(glob.glob(os.path.join(d, "*.hea")))) if h]
    H = [h for h in H if h["fs"] == FS and h["n"] >= L]
    cand = [(h["seg"], j, h["t0"]) for h in H for j in range(h["n"] // L)]
    n_all = len(cand)
    if n_all == 0:
        return csn, None, dict(n_win_all=0, n_eval=0, n_good=0, frac_good=np.nan)
    if n_all > n_cand:
        cand = [cand[i] for i in np.linspace(0, n_all - 1, n_cand).astype(int)]

    rec_cache, good = {}, []
    n_eval = 0
    for seg, j, t0 in cand:
        if seg not in rec_cache:
            try:
                rec_cache = {seg: wfdb.rdrecord(os.path.join(d, seg), channels=[0]).p_signal[:, 0]}
            except Exception:
                rec_cache = {seg: None}
        sig = rec_cache[seg]
        if sig is None:
            continue
        x = sig[j * L:(j + 1) * L].astype(np.float64)
        q = sqi_window(x); n_eval += 1
        if q["ok"]:
            x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
            z = ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float16)
            t = (t0 + timedelta(seconds=j * L / FS)).isoformat(sep=' ') if t0 else ''
            good.append((z, dict(seg=seg, win=j, win_time=t, sqi_hr=q["hr"],
                                 ibi_reg=q["ibi_reg"], tmpl_corr=q["tmpl_corr"], sat=q["sat"],
                                 ppg_hr=spectral_hr(z.astype(np.float64)))))
    stat = dict(n_win_all=n_all, n_eval=n_eval, n_good=len(good),
                frac_good=len(good) / n_eval if n_eval else np.nan)
    if len(good) > per_pat:
        good = [good[i] for i in np.linspace(0, len(good) - 1, per_pat).astype(int)]
    return csn, good, stat


BB   = r"propranolol|carvedilol|nadolol|metoprolol|atenolol|bisoprolol|labetalol|nebivolol|inderal|coreg|toprol|lopressor"
NSBB = r"propranolol|carvedilol|nadolol|inderal|coreg"
RAD_CIRR = r"cirrho|nodular (?:liver|contour|hepatic)|shrunken liver"
RAD_PH   = r"splenomegaly|ascites|varices|recanaliz|portosystemic|collateral"


def covariates(P, skip_numerics=False):
    R = pd.read_csv(os.path.join(MC, "rads.csv"), usecols=["CSN", "Impression"], dtype={"Impression": str})
    R = R[R.CSN.isin(P.CSN)]
    t = R.Impression.fillna("").str.lower()
    R["c"] = t.str.contains(RAD_CIRR, regex=True); R["p"] = t.str.contains(RAD_PH, regex=True)
    g = R.groupby("CSN").agg(has_rad=("c", "size"), rad_cirr=("c", "max"), rad_ph=("p", "max"))
    P = P.merge(g, left_on="CSN", right_index=True, how="left")
    P["has_rad"] = P.has_rad.fillna(0).gt(0).astype(int)
    for c in ["rad_cirr", "rad_ph"]:
        P[c] = P[c].fillna(False).astype(int)
    print(f"[rads] with imaging {P.has_rad.mean():.1%}")

    M = pd.read_csv(os.path.join(MC, "meds.csv"), dtype=str)
    M["MRN"] = pd.to_numeric(M.MRN, errors="coerce")
    tx = M[[c for c in ["Name", "Generic_name"] if c in M.columns]].fillna("").agg(" ".join, axis=1).str.lower()
    bb = set(M.loc[tx.str.contains(BB, regex=True), "MRN"])
    ns = set(M.loc[tx.str.contains(NSBB, regex=True), "MRN"])
    P["bb_ever"] = P.MRN.isin(bb).astype(int); P["nsbb_ever"] = P.MRN.isin(ns).astype(int)
    print(f"[meds] beta-blocker {P.bb_ever.mean():.1%}")

    if skip_numerics:
        return P
    t0 = time.time(); acc = {}
    want = {"Perf": "pi", "HR": "num_hr", "SpO2": "num_spo2", "RR": "num_rr"}
    ids = set(P.CSN)
    for ch in pd.read_csv(os.path.join(MC, "numerics.csv"), usecols=["CSN", "Measure", "Value"],
                          chunksize=5_000_000, low_memory=False):
        ch = ch[ch.CSN.isin(ids) & ch.Measure.isin(want)]
        ch["Value"] = pd.to_numeric(ch.Value, errors="coerce")
        for (c, m), v in ch.groupby(["CSN", "Measure"]).Value:
            acc.setdefault((c, m), []).append(v.to_numpy())
    rows = [dict(CSN=c, var=want[m], v=float(np.nanmedian(np.concatenate(vs))))
            for (c, m), vs in acc.items()]
    if rows:
        N_ = pd.DataFrame(rows).pivot(index="CSN", columns="var", values="v")
        P = P.merge(N_, left_on="CSN", right_index=True, how="left")
    print(f"[numerics] {time.time()-t0:.0f}s   PI available {P.get('pi', pd.Series(dtype=float)).notna().mean():.1%}")
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pat", type=int, default=100)
    ap.add_argument("--n-cand", type=int, default=300, help="candidate windows evaluated per visit")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 3)))
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--skip-numerics", action="store_true")
    ap.add_argument("--cohort", default=COH, help="cohort file")
    ap.add_argument("--suffix", default="", help="suffix of the output files")
    a = ap.parse_args()

    C = pd.read_csv(a.cohort)
    grp = C.groupby("CSN").grp.agg(lambda s: set(s))
    keep = [c for c in ["MRN", "CSN", "Age", "Gender", "Triage_HR", "Triage_RR", "Triage_SpO2",
                        "Triage_SBP", "Triage_DBP", "Triage_acuity", "ED_dispo",
                        "cirrhosis", "decomp", "comp", "cld", "phtn", "control",
                        "hf", "sepsis", "esrd", "pe"] if c in C.columns]
    P = C[keep].drop_duplicates("CSN").reset_index(drop=True)
    for c in ["cirrhosis", "decomp", "comp", "cld", "phtn", "control", "hf", "sepsis", "esrd", "pe"]:
        if c in P.columns:
            P = P.rename(columns={c: f"dx_{c}"})
    for g in GROUPS:
        P[f"in_{g}"] = P.CSN.map(lambda c: int(g in grp[c]))
    if a.test:
        P = pd.concat([P[P.in_cirrhosis == 1].head(15), P[P.in_cirrhosis == 0].head(15)],
                      ignore_index=True)
    print(f"[cohort] {len(P):,} visits  " +
          "  ".join(f"{g}={int(P[f'in_{g}'].sum())}" for g in GROUPS))

    t0 = time.time(); res = {}
    tasks = [(int(c), a.n_cand, a.per_pat, 0) for c in P.CSN]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, (csn, good, stat) in enumerate(ex.map(process_visit, tasks, chunksize=4)):
            res[csn] = (good, stat)
            if (k + 1) % 200 == 0 or k + 1 == len(tasks):
                el = time.time() - t0
                print(f"   {k+1:,}/{len(tasks):,}  {el:.0f}s  eta {el/(k+1)*(len(tasks)-k-1):.0f}s", flush=True)

    S = pd.DataFrame([dict(CSN=c, **s) for c, (_, s) in res.items()])
    P = P.merge(S, on="CSN", how="left")
    N = sum(len(g) for g, _ in res.values() if g)
    print(f"[win] {N:,} windows")
    suf = a.suffix + ("_test" if a.test else "")
    X = np.lib.format.open_memmap(os.path.join(CACHE, f"X_mcmed{suf}.npy"), mode="w+",
                                  dtype=np.float16, shape=(N, L))
    rows, i = [], 0
    info = P.set_index("CSN")
    for c in P.CSN:
        good, _ = res[int(c)]
        if not good:
            continue
        r0 = info.loc[c]
        for z, m in good:
            X[i] = z; i += 1
            rows.append(dict(subject_id=int(r0.MRN), CSN=int(c), **m,
                             age=r0.Age, gender=r0.Gender, ok_ppg=True,
                             **{f"in_{g}": int(r0[f"in_{g}"]) for g in GROUPS}))
    X.flush()
    I = pd.DataFrame(rows)
    I["lbl_cirrhosis"] = I.in_cirrhosis; I["lbl_control"] = I.in_control
    I.to_csv(os.path.join(CACHE, f"mcmed_index{suf}.csv"), index=False)

    ppg = I.groupby("CSN").agg(n_win=("win", "size"), ppg_hr=("ppg_hr", "median"),
                               sqi_hr=("sqi_hr", "median"), tmpl_corr=("tmpl_corr", "median"))
    P = P.merge(ppg, left_on="CSN", right_index=True, how="left")
    P = covariates(P, a.skip_numerics)
    P.to_csv(os.path.join(DERIVED, f"mcmed_patients{suf}.csv"), index=False)

    print(f"\n[done] {time.time()-t0:.0f}s   X_mcmed{suf}.npy {os.path.getsize(os.path.join(CACHE, f'X_mcmed{suf}.npy'))/1e9:.2f} GB")
    P["ge10"] = P.n_win.fillna(0) >= 10
    for g in GROUPS:
        s = P[P[f"in_{g}"] == 1]
        if len(s):
            print(f"  {g:<10} visits {len(s):>5,}   >=10 windows {s.ge10.mean():6.1%}   "
                  f"median good fraction {s.frac_good.median():.2f}   median windows {s.n_win.median():.0f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore"); main()
