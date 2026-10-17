"""Handcrafted features for every 30-s window used in the main analysis: pulse-contour descriptors of the
phase-normalized median beat (07_morphology.py), pulse rate, beat consistency, amplitude variability and
short-term pulse-rate variability (IBI 0.33-1.5 s within 20% of the window median). Patient values are medians.

Usage: python 27_morph_hrv.py --db {mimic,mcmed} [--test]
Outputs (derived/): morph_hrv_patient_<db>.csv, morph_hrv_window_<db>.csv.gz, phase_curves_<db>.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, time, warnings, importlib.util
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

FS, NBIN = 125.0, 100
SRC = {
    "mimic": ("X_ppg.npy",   "index.csv",       "beat_coh_aligned_meta.csv",   "subject_id", "split712_mimic_cirrhosis.csv"),
    "mcmed": ("X_mcmed.npy", "mcmed_index.csv", "beat_mcmed_aligned_meta.csv", "CSN",        "split712_mcmed_cirrhosis.csv"),
}
MORPH = ["crest", "notch_pos", "notch_found", "notch_depth", "RI", "IPA", "AI", "dT", "width50",
         "b_a", "d_a", "sys_area"]
OTHER = ["hr", "hrv", "beat_consistency", "amp_cv"]
HRV = ["ibi_mean", "sdnn", "rmssd", "pnn50", "sd1", "sd2", "ibi_cv"]
FEATS = MORPH + OTHER + HRV

_X = None; _M07 = None; _SOS = None


def _load07(root):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("m07", os.path.join(here, "07_morphology.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _init(xpath, root):
    global _X, _M07
    _X = np.load(xpath, mmap_mode="r"); _M07 = _load07(root)


def _beats(x):
    from scipy.signal import butter, sosfiltfilt, find_peaks
    global _SOS
    if _SOS is None:
        _SOS = butter(3, [0.5 / (FS / 2), 8.0 / (FS / 2)], btype="band", output="sos")
    xf = sosfiltfilt(_SOS, x).astype(np.float32)
    sd = xf.std()
    if sd < 1e-6: return None
    pk, _ = find_peaks(xf, distance=int(FS * 60 / 180), prominence=sd * 0.3)
    if len(pk) < 4: return None
    rr = float(np.median(np.diff(pk)))
    d1 = np.gradient(xf)
    feet = []
    for p in pk:
        w0 = max(0, p - int(0.60 * rr))
        if p - w0 < 4: continue
        j = w0 + int(np.argmax(d1[w0:p]))
        k0 = max(0, j - int(0.28 * rr))
        if j - k0 < 2: continue
        feet.append(k0 + int(np.argmin(xf[k0:j + 1])))
    feet = np.unique(feet)
    if len(feet) < 4: return None
    M, dur, amp = [], [], []
    g = np.linspace(0, 1, NBIN)
    for i in range(len(feet) - 1):
        s, e = feet[i], feet[i + 1]; n = e - s
        if not (int(FS * 0.33) <= n <= int(FS * 1.5)): continue
        b = xf[s:e].astype(np.float64); rng = b.max() - b.min()
        if rng < 1e-6: continue
        bn = np.interp(g, np.linspace(0, 1, n), (b - b.min()) / rng)
        if int(np.argmax(bn)) > int(0.55 * NBIN): continue
        M.append(bn); dur.append(n / FS); amp.append(rng)
    if len(M) < 3: return None
    return np.asarray(M), np.asarray(dur), np.asarray(amp), feet


def _hrv(feet):
    ibi = np.diff(feet) / FS
    ok = (ibi >= 0.33) & (ibi <= 1.5)
    if ok.sum() < 10: return {}
    med = np.median(ibi[ok]); ok &= np.abs(ibi - med) <= 0.2 * med
    if ok.sum() < 10: return {}
    v = ibi[ok]; sdnn = float(v.std(ddof=1))
    adj = ok[:-1] & ok[1:]; dd = np.diff(ibi)[adj]
    out = dict(ibi_mean=float(v.mean() * 1000), sdnn=sdnn * 1000, ibi_cv=sdnn / float(v.mean()),
               n_ibi=int(ok.sum()), frac_ibi_ok=float(ok.mean()))
    if len(dd) >= 5:
        sd1 = float(np.sqrt(0.5) * dd.std(ddof=1))
        out.update(rmssd=float(np.sqrt(np.mean(dd ** 2)) * 1000), pnn50=float(np.mean(np.abs(dd) > 0.05)),
                   sd1=sd1 * 1000, sd2=float(np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0.0)) * 1000))
    return out


def _job(i):
    x = np.nan_to_num(np.asarray(_X[i], dtype=np.float32), nan=0.0)
    r = _beats(x)
    if r is None: return i, None, None
    M, dur, amp, feet = r
    med = np.median(M, axis=0)
    f = _M07._morph(med)
    if f is None: return i, None, None
    f["hr"] = float(60.0 / np.median(dur))
    f["hrv"] = float(np.std(dur) / max(np.mean(dur), 1e-6))
    f["n_beat"] = int(len(M))
    f["beat_consistency"] = float(np.mean([np.corrcoef(bb, med)[0, 1] for bb in M[:40]]))
    f["amp_cv"] = float(np.std(amp) / max(np.mean(amp), 1e-9))
    f.update(_hrv(feet))
    return i, f, med.astype(np.float32)


def cliff(x1, x0):
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(x1, x0)
    return 2 * u / (len(x1) * len(x0)) - 1, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, choices=list(SRC))
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 3)))
    ap.add_argument("--root", default=os.environ.get("PPG_ROOT", "."))
    a = ap.parse_args()
    CACHE, DERIVED = os.path.join(a.root, "cache"), os.path.join(a.root, "derived")
    fx, fi, fm, key, fs_ = SRC[a.db]
    xpath = os.path.join(CACHE, fx)
    X = np.load(xpath, mmap_mode="r")
    I = pd.read_csv(os.path.join(CACHE, fi), usecols=[key])
    Mt = pd.read_csv(os.path.join(CACHE, fm), usecols=["ok_beat"])
    assert len(X) == len(I) == len(Mt), f"{fx} {len(X)} / {fi} {len(I)} / {fm} {len(Mt)} do not match"
    S = pd.read_csv(os.path.join(DERIVED, fs_), usecols=["pid", "y"])
    rows = np.flatnonzero((Mt.ok_beat == 1).to_numpy() & I[key].isin(S.pid).to_numpy())
    if a.test: rows = rows[::max(1, len(rows) // 2000)][:2000]
    sfx = "_test" if a.test else ""
    print(f"[run] {a.db}: {len(rows):,} windows / {I.loc[rows, key].nunique():,} patients, {a.workers} workers", flush=True)

    res, curv = {}, {}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(xpath, a.root)) as ex:
        for i, f, med in ex.map(_job, rows.tolist(), chunksize=256):
            if f is not None: res[i] = f; curv[i] = med
            done += 1
            if done % 25000 == 0:
                el = time.time() - t0
                print(f"   {done:,}/{len(rows):,}  {el:.0f}s  eta {el / done * (len(rows) - done):.0f}s", flush=True)
    print(f"[run] {time.time() - t0:.0f}s   success {len(res) / max(len(rows), 1):.1%}", flush=True)

    W = pd.DataFrame.from_dict(res, orient="index"); W.index.name = "row"; W = W.reset_index()
    W["pid"] = I.loc[W.row.to_numpy(), key].to_numpy()
    for c in FEATS:
        if c not in W.columns: W[c] = np.nan
    W.to_csv(os.path.join(DERIVED, f"morph_hrv_window_{a.db}{sfx}.csv.gz"), index=False)

    agg = {c: (c, "median") for c in FEATS}
    agg["n_win"] = ("row", "size")
    agg["n_win_hrv"] = ("rmssd", "count")
    P = W.groupby("pid").agg(**agg).reset_index().merge(S, on="pid", how="left")
    P.to_csv(os.path.join(DERIVED, f"morph_hrv_patient_{a.db}{sfx}.csv"), index=False)

    C = pd.DataFrame(np.stack([curv[i] for i in W.row]), index=W.pid.to_numpy())
    PC = C.groupby(level=0).median()
    np.savez(os.path.join(DERIVED, f"phase_curves_{a.db}{sfx}.npz"),
             pid=PC.index.to_numpy(), curves=PC.to_numpy().astype(np.float32))

    out = [f"db={a.db}  windows {len(rows):,} (with features {len(W):,})  patients {len(P):,}",
           f"  windows with PRV {W.rmssd.notna().mean():.1%}; median per patient {P.n_win_hrv.median():.0f}",
           "", f"{'feature':<18}{'cases':>10}{'controls':>10}{'delta':>11}{'p':>12}"]
    c1, c0 = P[P.y == 1], P[P.y == 0]
    for c in FEATS:
        x1, x0 = c1[c].dropna(), c0[c].dropna()
        if len(x1) < 10 or len(x0) < 10: continue
        d, p = cliff(x1, x0)
        out.append(f"{c:<18}{x1.median():>10.3f}{x0.median():>10.3f}{d:>+11.3f}{p:>12.2e}")
    if a.db == "mimic":
        fo = os.path.join(DERIVED, "morphology_patient.csv")
        if os.path.exists(fo):
            O = pd.read_csv(fo).rename(columns={"subject_id": "pid"})
            Q = P.merge(O, on="pid", suffixes=("", "_07"))
            out += ["", f"vs morphology_patient.csv (07), {len(Q):,} patients: Pearson r"]
            out += ["  " + "  ".join(f"{c} {Q[c].corr(Q[c + '_07']):.3f}" for c in MORPH + ["hr", "hrv"]
                                     if c + "_07" in Q.columns)]
    txt = "\n".join(out); print("\n" + txt)
    open(os.path.join(DERIVED, f"morph_hrv_report_{a.db}{sfx}.txt"), "w", encoding="utf-8").write(txt)
    print(f"\n[out] derived/morph_hrv_patient_{a.db}{sfx}.csv")


if __name__ == "__main__":
    warnings.filterwarnings("ignore"); main()
