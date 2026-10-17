"""Spectral pulse rate of each cached PPG segment (peak of the 0.6-3 Hz power spectrum).

Output: cache/ppg_hr.npy (bpm; NaN when no clear peak)
"""
import os, time, numpy as np, pandas as pd

CACHE = os.path.join(os.environ.get("PPG_ROOT", "."), "cache")
FS, N = 125.0, 3750
LO, HI = 0.6, 3.0
CHUNK = 20000

def main():
    X = np.load(os.path.join(CACHE, "X_ppg.npy"), mmap_mode="r")
    idx = pd.read_csv(os.path.join(CACHE, "index.csv"))
    n = len(X)
    assert n == len(idx), f"{n} waveforms vs {len(idx)} index rows"

    freqs = np.fft.rfftfreq(N, 1.0 / FS)
    band = (freqs >= LO) & (freqs <= HI)
    fb = freqs[band]
    win = np.hanning(N).astype(np.float32)

    hr = np.full(n, np.nan, dtype=np.float32)
    t0 = time.time()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        x = np.asarray(X[s:e], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0)
        sd = x.std(axis=1)
        x = (x - x.mean(axis=1, keepdims=True)) * win
        P = np.abs(np.fft.rfft(x, axis=1)) ** 2
        Pb = P[:, band]
        h = fb[Pb.argmax(axis=1)] * 60.0
        prom = Pb.max(axis=1) / (Pb.mean(axis=1) + 1e-12)
        h[(sd < 1e-6) | (prom < 3.0)] = np.nan
        hr[s:e] = h
        print(f"   {e:,}/{n:,}  {time.time()-t0:.0f}s", flush=True)

    np.save(os.path.join(CACHE, "ppg_hr.npy"), hr)
    ok = np.isfinite(hr)
    print(f"\n[out] cache/ppg_hr.npy   valid {ok.mean():.1%}")
    print(f"  median {np.nanmedian(hr):.1f}  "
          f"IQR {np.nanpercentile(hr,25):.1f}-{np.nanpercentile(hr,75):.1f} bpm")

    if "median_30s_hr" in idx.columns:
        ref = pd.to_numeric(idx["median_30s_hr"], errors="coerce").to_numpy()
        m = ok & np.isfinite(ref)
        r = np.corrcoef(hr[m], ref[m])[0, 1]
        d = hr[m] - ref[m]
        print(f"\n[check] vs metadata HR (n={m.sum():,}): r = {r:.4f}, "
              f"median difference {np.median(d):+.2f} bpm, |d| < 5 bpm {(np.abs(d)<5).mean():.1%}")

if __name__ == "__main__":
    main()
