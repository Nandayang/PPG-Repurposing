"""Beat detection and pulse-contour descriptors of the median beat (also imported by 27_morph_hrv.py).

Outputs (derived/): morphology_seg.csv, morphology_patient.csv, phase_curves.npz, morphology_report.txt
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, warnings, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

CACHE   = os.path.join(os.environ.get("PPG_ROOT", "."), "cache")
DERIVED = os.path.join(os.environ.get("PPG_ROOT", "."), "derived")
FS, L, NBIN = 125.0, 3750, 100
os.makedirs(DERIVED, exist_ok=True)

_X = None
def _init():
    global _X
    _X = np.load(os.path.join(CACHE, "X_ppg.npy"), mmap_mode="r")

def _beats(x):
    from scipy.signal import butter, sosfiltfilt, find_peaks
    global _SOS
    try: sos = _SOS
    except NameError:
        sos = _SOS = butter(3, [0.5/(FS/2), 8.0/(FS/2)], btype="band", output="sos")
    xf = sosfiltfilt(sos, x).astype(np.float32)
    sd = xf.std()
    if sd < 1e-6: return None, None
    pk, _ = find_peaks(xf, distance=int(FS*60/180), prominence=sd*0.3)
    if len(pk) < 4: return None, None
    rr = float(np.median(np.diff(pk)))
    d1 = np.gradient(xf)
    feet = []
    for p in pk:
        w0 = max(0, p - int(0.60*rr))
        if p - w0 < 4: continue
        j = w0 + int(np.argmax(d1[w0:p]))
        k0 = max(0, j - int(0.28*rr))
        if j - k0 < 2: continue
        feet.append(k0 + int(np.argmin(xf[k0:j+1])))
    feet = np.unique(feet)
    if len(feet) < 4: return None, None
    M, dur = [], []
    g = np.linspace(0, 1, NBIN)
    for i in range(len(feet)-1):
        s, e = feet[i], feet[i+1]; n = e - s
        if not (int(FS*0.33) <= n <= int(FS*1.5)): continue
        b = xf[s:e].astype(np.float64); rng = b.max() - b.min()
        if rng < 1e-6: continue
        bn = np.interp(g, np.linspace(0,1,n), (b-b.min())/rng)
        if int(np.argmax(bn)) > int(0.55*NBIN): continue
        M.append(bn); dur.append(n/FS)
    if len(M) < 3: return None, None
    return np.asarray(M), np.asarray(dur)

def _morph(m):
    t = np.linspace(0, 1, NBIN)
    sp = int(np.argmax(m))
    if sp < 3 or sp > int(0.55*NBIN): return None
    d1 = np.gradient(m); d2 = np.gradient(d1)

    lo = max(sp + 4, int(0.25*NBIN)); hi = int(0.70*NBIN)
    if hi - lo < 4: return None
    w = m[lo:hi]
    loc = [j for j in range(2, len(w)-2)
           if w[j] <= w[j-1] and w[j] <= w[j-2] and w[j] <= w[j+1] and w[j] <= w[j+2]]
    if loc:
        notch = lo + int(loc[int(np.argmin([w[j] for j in loc]))]); found = 1
    else:
        notch = lo + int(np.argmax(d2[lo:hi])); found = 0

    dia = m[notch:]
    dp = notch + int(np.argmax(dia))
    a_i = int(np.argmax(d2[:max(int(0.30*NBIN), 4)]))
    b_i = a_i + int(np.argmin(d2[a_i:max(a_i+2, int(0.45*NBIN))]))
    dseg = d2[notch:min(notch+30, NBIN)]
    d_i = notch + int(np.argmin(dseg)) if len(dseg) > 2 else notch
    a, b, dd = d2[a_i], d2[b_i], d2[d_i]
    half = m >= 0.5
    ipa = float(np.trapezoid(m[:notch]) / max(np.trapezoid(m[notch:]), 1e-6))
    return dict(
        crest        = float(t[sp]),
        notch_pos    = float(t[notch]),
        notch_found  = int(found),
        notch_depth  = float(m[sp] - m[notch]),
        RI           = float(m[dp] / max(m[sp], 1e-6)),
        IPA          = float(np.clip(ipa, 0, 20)),
        AI           = float((m[dp] - m[sp]) / max(m[sp], 1e-6)),
        dT           = float(t[dp] - t[sp]),
        width50      = float(half.sum() / NBIN),
        b_a          = float(b / a) if abs(a) > 1e-9 else np.nan,
        d_a          = float(dd / a) if abs(a) > 1e-9 else np.nan,
        sys_area     = float(np.trapezoid(m[:notch]) / NBIN),
    )

def _job(i):
    x = np.asarray(_X[i], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    M, dur = _beats(x)
    if M is None: return i, None, None
    med = np.median(M, axis=0)
    f = _morph(med)
    if f is None: return i, None, None
    f["hr"] = float(60.0/np.median(dur))
    f["hrv"] = float(np.std(dur)/max(np.mean(dur),1e-6))
    f["n_beat"] = int(len(M))
    f["beat_consistency"] = float(np.mean([np.corrcoef(bb, med)[0,1] for bb in M[:40]]))
    return i, f, med.astype(np.float32)

FEATS = ["crest","notch_pos","notch_found","notch_depth","RI","IPA","AI","dT","width50",
         "b_a","d_a","sys_area","hr","hrv","beat_consistency"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4)//3)))
    a = ap.parse_args()

    idx = pd.read_csv(os.path.join(CACHE, "index.csv"))
    n = len(idx)
    rows = np.arange(n)[idx.ok_ppg.to_numpy() == True]
    if a.test: rows = rows[::max(1, len(rows)//2000)][:2000]
    print(f"[run] {len(rows):,} segments, {a.workers} workers")

    res = {}; curves = np.zeros((n, NBIN), np.float32); okc = np.zeros(n, bool)
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        for i, f, med in ex.map(_job, rows.tolist(), chunksize=256):
            if f is not None: res[i] = f; curves[i] = med; okc[i] = True
            done += 1
            if done % 25000 == 0:
                el = time.time()-t0
                print(f"   {done:,}/{len(rows):,}  {el:.0f}s  eta {el/done*(len(rows)-done):.0f}s", flush=True)
    print(f"[run] {time.time()-t0:.0f}s   success {len(res)/len(rows):.1%}")

    S = pd.DataFrame.from_dict(res, orient="index")
    S.index.name = "row"; S = S.reset_index()
    keep = ["row","subject_id","strat_fold","age","gender"] + [c for c in idx.columns if c.startswith("lbl_")]
    S = S.merge(idx.reset_index().rename(columns={"index":"row"})[keep], on="row")
    S.to_csv(os.path.join(DERIVED,"morphology_seg.csv"), index=False)

    agg = {c:(c,"median") for c in FEATS}
    agg.update({"n_seg":("row","size"), "fold":("strat_fold","first"),
                "age":("age","median"), "gender":("gender","first")})
    agg.update({c:(c,"max") for c in S.columns if c.startswith("lbl_")})
    P = S.groupby("subject_id").agg(**agg).reset_index()
    P["male"] = (P.gender == "M").astype(int)
    P.to_csv(os.path.join(DERIVED,"morphology_patient.csv"), index=False)
    print(f"[out] {len(P):,} patients")

    tmp = pd.DataFrame(curves[S.row.to_numpy()], index=S.subject_id.to_numpy())
    PC = tmp.groupby(level=0).median()
    np.savez(os.path.join(DERIVED,"phase_curves.npz"),
             sid=PC.index.to_numpy(), curves=PC.to_numpy().astype(np.float32))

    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    out = []
    def P_(s=""): print(s); out.append(s)

    cir = P[P.lbl_cirrhosis == 1]; ctl = P[(P.lbl_control == 1) & (P.lbl_cirrhosis == 0)]
    P_(f"\n=== Cirrhosis {len(cir)} vs matched controls {len(ctl)} ===")
    P_(f"{'feature':<18}{'cases':>10}{'controls':>10}{'delta':>11}{'p':>12}")
    for c in FEATS:
        x1 = cir[c].dropna(); x0 = ctl[c].dropna()
        if len(x1) < 10 or len(x0) < 10: continue
        u, pv = stats.mannwhitneyu(x1, x0); d = 2*u/(len(x1)*len(x0)) - 1
        star = " ***" if pv < 1e-4 else (" **" if pv < 1e-2 else (" *" if pv < .05 else ""))
        P_(f"{c:<18}{x1.median():>10.3f}{x0.median():>10.3f}{d:>+11.3f}{pv:>12.2e}{star}")

    D = pd.concat([cir, ctl]); D["_y"] = (D.lbl_cirrhosis == 1).astype(int)
    P_("\n=== Handcrafted features (patient-level OOF AUC, 10 folds) ===")
    SETS = [("pulse rate",["hr"]),
            ("age+sex",["age","male"]),
            ("morphology",[c for c in FEATS if c != "hr"]),
            ("morphology+rate",FEATS),
            ("morphology+rate+age+sex",FEATS+["age","male"])]
    for nm, fs in SETS:
        fs = [c for c in fs if c in D.columns]
        oy, op = [], []
        for f in range(10):
            tr, te = D[D.fold != f], D[D.fold == f]
            if te._y.nunique() < 2: continue
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=3000, C=0.5))
            m.fit(tr[fs], tr._y); oy += list(te._y); op += list(m.predict_proba(te[fs])[:,1])
        P_(f"  {nm:<24} AUC = {roc_auc_score(oy,op):.3f}   ({len(fs)} features)")

    sid = PC.index.to_numpy(); cur = PC.to_numpy()
    m1 = np.isin(sid, cir.subject_id); m0 = np.isin(sid, ctl.subject_id)
    diff = np.median(cur[m1],0) - np.median(cur[m0],0)
    t = np.linspace(0,1,NBIN); i = int(np.argmax(np.abs(diff)))
    P_("\n=== Median beat difference (cases - controls) ===")
    P_(f"  maximum at phase {t[i]:.3f}: {diff[i]:+.4f}")
    thr = 0.5*np.abs(diff).max(); band = t[np.abs(diff) >= thr]
    P_(f"  phases with |difference| >= half maximum: {band.min():.2f}-{band.max():.2f}")
    with open(os.path.join(DERIVED,"morphology_report.txt"),"w",encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print("\n[out] derived/morphology_report.txt")

if __name__ == "__main__":
    warnings.filterwarnings("ignore"); main()
