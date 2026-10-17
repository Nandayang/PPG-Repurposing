"""Age- and sex-matched ICU cohort and waveform cache.

Outputs (cache/): X_ppg.npy, X_ecg.npy (N x 3750, float16), index.csv
Usage: python 02_cache_waveforms.py [--per-pat 100] [--k 4] [--test]
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, warnings, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

PPG_DIR = os.environ.get("MIMIC_PPG_DIR", "data/mimic-iii-ext-ppg")
DERIVED = os.path.join(os.environ.get("PPG_ROOT", "."), "derived")
CACHE   = os.path.join(os.environ.get("PPG_ROOT", "."), "cache")
L = 3750
os.makedirs(CACHE, exist_ok=True)

LABELS = ["lbl_cirrhosis","lbl_cld","lbl_phtn","lbl_decomp","lbl_ascites","lbl_he",
          "lbl_var_bleed","lbl_hrs","lbl_hf","lbl_sepsis","lbl_esrd","lbl_pe",
          "lbl_cirr_decomp","lbl_cirr_comp","lbl_control"]

def match_controls(cases, pool, k=4, age_tol=3, seed=0):
    used, out = set(), []
    for _, c in cases.sample(frac=1, random_state=seed).iterrows():
        cand = pool[(pool.male == c.male) &
                    (pool.age.sub(c.age).abs() <= age_tol) &
                    (~pool.subject_id.isin(used))]
        if len(cand) == 0: continue
        t = cand.sample(min(k, len(cand)), random_state=int(c.subject_id) % 10000)
        used |= set(t.subject_id); out.append(t)
    return pd.concat(out) if out else pool.head(0)

def _read(args):
    i, rel = args
    import wfdb
    p = os.path.join(PPG_DIR, os.path.normpath(rel))
    try:
        r = wfdb.rdrecord(p)
        names = [n.upper() for n in r.sig_name]
        sig = r.p_signal
        ppg = sig[:, names.index("PLETH")] if "PLETH" in names else None
        ecg = sig[:, names.index("II")] if "II" in names else None
        if ppg is None or len(ppg) < L: return i, None, None
        ppg = ppg[:L]
        ecg = ecg[:L] if (ecg is not None and len(ecg) >= L) else None
        return i, ppg.astype(np.float16), (None if ecg is None else ecg.astype(np.float16))
    except Exception:
        return i, None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pat", type=int, default=100, help="maximum segments per patient")
    ap.add_argument("--k", type=int, default=4, help="controls per case")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4)//3)))
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    print("[load] segments_index.parquet ...")
    seg = pd.read_parquet(os.path.join(DERIVED, "segments_index.parquet"))
    pat = pd.read_csv(os.path.join(DERIVED, "cohort_patients.csv"))
    pat = pat[pat.n_ppg_good >= 10].copy()
    pat["male"] = (pat.gender == "M").astype(int)

    cases = pat[pat.lbl_cld == 1]
    pool  = pat[pat.lbl_control == 1]
    ctrl  = match_controls(cases, pool, k=a.k)
    keep  = set(cases.subject_id) | set(ctrl.subject_id)
    print(f"[cohort] CLD cases {len(cases)}  matched controls {len(ctrl)}  total {len(keep)}")
    print(f"         median age: cases {cases.age.median():.0f} / controls {ctrl.age.median():.0f}")

    seg = seg[seg.subject_id.isin(keep) & (seg.ppg_all_good == 1)].copy()
    print(f"[seg] high-quality segments {len(seg):,}")

    seg = seg.sort_values(["subject_id", "start_segment"]).reset_index(drop=True)
    g = seg.groupby("subject_id", sort=False)
    r = g.cumcount().to_numpy()
    n = g["subject_id"].transform("size").to_numpy()
    m = np.minimum(n, a.per_pat)
    keep = np.floor(r * m / n) > np.floor((r - 1) * m / n)
    seg = seg.loc[keep].reset_index(drop=True)
    if a.test: seg = seg.head(200)
    N = len(seg)
    print(f"[seg] sampled {N:,} segments / {seg.subject_id.nunique():,} patients"
          f" (median {seg.groupby('subject_id').size().median():.0f} per patient)")

    Xp = np.lib.format.open_memmap(os.path.join(CACHE,"X_ppg.npy"), mode="w+",
                                   dtype=np.float16, shape=(N, L))
    Xe = np.lib.format.open_memmap(os.path.join(CACHE,"X_ecg.npy"), mode="w+",
                                   dtype=np.float16, shape=(N, L))
    ok_p = np.zeros(N, bool); ok_e = np.zeros(N, bool)

    t0 = time.time(); done = 0
    tasks = list(enumerate(seg.folder_path.tolist()))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, p, e in ex.map(_read, tasks, chunksize=64):
            if p is not None: Xp[i] = p; ok_p[i] = True
            if e is not None: Xe[i] = e; ok_e[i] = True
            done += 1
            if done % 20000 == 0:
                el = time.time() - t0
                print(f"   {done:,}/{N:,}  {el:.0f}s  eta {el/done*(N-done):.0f}s", flush=True)
    Xp.flush(); Xe.flush()

    seg["ok_ppg"] = ok_p; seg["ok_ecg"] = ok_e
    cols = ["subject_id","folder_path","strat_fold","ok_ppg","ok_ecg",
            "median_30s_hr","median_30s_sbp","median_30s_dbp","median_30s_rr",
            "age","gender","event_rhythm","clinical_information_system"] + \
           [c for c in seg.columns if c.startswith("lbl_")]
    seg[[c for c in cols if c in seg.columns]].to_csv(os.path.join(CACHE,"index.csv"), index=False)

    print(f"\n[done] {time.time()-t0:.0f}s")
    print(f"  PPG read {ok_p.mean():.1%}   ECG available {ok_e.mean():.1%}")
    print(f"  X_ppg.npy {os.path.getsize(os.path.join(CACHE,'X_ppg.npy'))/1e9:.2f} GB")
    v = np.asarray(Xp[ok_p][:2000], dtype=np.float32)
    print(f"  PPG range min={np.nanmin(v):.3f} max={np.nanmax(v):.3f} "
          f"nan={np.isnan(v).mean():.2%}")
    print(f"  segments: cirrhosis {int(seg.lbl_cirrhosis.sum()):,} / CLD {int(seg.lbl_cld.sum()):,} "
          f"/ controls {int(seg.lbl_control.sum()):,}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore"); main()
