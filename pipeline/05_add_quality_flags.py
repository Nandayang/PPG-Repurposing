"""Adds PPG and ECG quality flags to cache/index.csv; both_good marks segments in which both signals are of high quality
(used for the PPG vs ECG comparison).
"""
import os, pandas as pd

DERIVED = os.path.join(os.environ.get("PPG_ROOT", "."), "derived")
CACHE   = os.path.join(os.environ.get("PPG_ROOT", "."), "cache")

def main():
    idx = pd.read_csv(os.path.join(CACHE, "index.csv"))
    n0  = len(idx)
    if "both_good" in idx.columns:
        print("[skip] index.csv already has quality flags")
        return

    seg = pd.read_parquet(os.path.join(DERIVED, "segments_index.parquet"),
                          columns=["folder_path","ppg_all_good","ecg_all_good",
                                   "ppg_n_good","ecg_n_good","has_ecg"])
    m = idx.merge(seg, on="folder_path", how="left", validate="one_to_one")
    assert len(m) == n0, f"row count changed {n0} -> {len(m)}"
    miss = m.ecg_all_good.isna().mean()
    assert miss == 0, f"{miss:.2%} of segments not found in segments_index"

    for c in ["ppg_all_good","ecg_all_good","has_ecg"]:
        m[c] = m[c].astype("int8")
    m["both_good"] = ((m.ppg_all_good == 1) & (m.ecg_all_good == 1) &
                      (m.ok_ppg == True) & (m.ok_ecg == True)).astype("int8")

    m.to_csv(os.path.join(CACHE, "index.csv"), index=False)
    print(f"[out] cache/index.csv updated ({n0:,} rows)")

    print(f"\nsegments: ppg_all_good {m.ppg_all_good.mean():.1%} | "
          f"ecg_all_good {m.ecg_all_good.mean():.1%} | both_good {m.both_good.mean():.1%}")

    pt = m.groupby("subject_id").agg(
        n=("subject_id","size"), n_both=("both_good","sum"),
        cir=("lbl_cirrhosis","max"), cld=("lbl_cld","max"),
        phtn=("lbl_phtn","max"), dec=("lbl_cirr_decomp","max"),
        comp=("lbl_cirr_comp","max"), ctl=("lbl_control","max")).reset_index()
    print("\nPatients with >= 10 segments of high-quality PPG and ECG:")
    print(f"  {'task':<12}{'pos':>8}{'neg':>8}")
    for nm, p, q in [("cirrhosis","cir","ctl"), ("cld","cld","ctl"),
                     ("phtn","phtn","ctl"), ("decomp","dec","comp")]:
        a = ((pt[p]==1) & (pt.n_both>=10)).sum()
        b = ((pt[q]==1) & (pt[p]==0) & (pt.n_both>=10)).sum()
        print(f"  {nm:<12}{a:>8}{b:>8}")
    print(f"\n  median per patient {pt.n_both.median():.0f}; "
          f"patients with none {int((pt.n_both==0).sum())}")

if __name__ == "__main__":
    main()
