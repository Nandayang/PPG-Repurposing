"""MC-MED cohort: diagnosis labels, age- and sex-matched controls and the list of visits whose Pleth records are needed.

Inputs (MCMED_DIR): visits.csv, pmh.csv, waveform_summary.csv
Outputs (derived/): mcmed_cohort_visits.csv, mcmed_download_csns.txt
"""
import os, re
import numpy as np, pandas as pd

MC  = os.environ.get("MCMED_DIR", "data/mc-med")
OUT = os.path.join(os.environ.get("PPG_ROOT", "."), "derived")
os.makedirs(OUT, exist_ok=True)


def norm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


I10 = {
    "cirrhosis": ("K703", "K717", "K743", "K744", "K745", "K746"),
    "phtn":      ("K766",),
    "varices":   ("I850", "I851", "I864"),
    "ascites":   ("R18",),
    "he":        ("K72",),
    "hrs":       ("K767",),
    "sbp_perit": ("K652",),
    "cld_extra": ("K70", "K71", "K73", "K74", "K75", "K760", "K768", "K769", "B18"),
    "hf":        ("I50",),
    "sepsis":    ("A41", "R652"),
    "esrd":      ("N186",),
    "pe":        ("I26",),
    "liver_any": ("K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77",
                  "B18", "I850", "I851", "I864", "R18", "Z944", "C22"),
}
I9 = {
    "cirrhosis": ("5712", "5715", "5716", "57149"),
    "phtn":      ("5723",),
    "varices":   ("4560", "4561"),
    "ascites":   ("7895",),
    "he":        ("5722",),
    "hrs":       ("5724",),
    "sbp_perit": ("5675", "56723"),
    "cld_extra": ("571", "5728", "070"),
    "hf":        ("428",),
    "sepsis":    ("99591", "99592", "78552"),
    "esrd":      ("5856",),
    "pe":        ("4151",),
    "liver_any": ("570", "571", "572", "573", "070", "4560", "4561",
                  "7895", "V427", "1550"),
}
KEYS = list(I10)


def _mask(series_norm, prefixes):
    return series_norm.str.startswith(tuple(prefixes), na=False)


def build_patient_labels():
    lab = {k: set() for k in KEYS}

    print("[pmh] reading ...")
    n = 0
    for ch in pd.read_csv(os.path.join(MC, "pmh.csv"),
                          usecols=["MRN", "CodeType", "Code"],
                          chunksize=1_000_000, low_memory=False):
        n += len(ch)
        ch["c"] = ch["Code"].map(norm)
        is10 = ch["CodeType"].astype(str).str.contains("10", na=False)
        for k in KEYS:
            m = (is10 & _mask(ch["c"], I10[k])) | (~is10 & _mask(ch["c"], I9[k]))
            lab[k] |= set(ch.loc[m, "MRN"].unique())
    print(f"[pmh] {n:,} rows")

    print("[visits] reading ...")
    v = pd.read_csv(os.path.join(MC, "visits.csv"),
                    usecols=["MRN", "Dx_ICD9", "Dx_ICD10"], low_memory=False)
    v["c10"] = v["Dx_ICD10"].map(norm)
    v["c9"] = v["Dx_ICD9"].map(norm)
    for k in KEYS:
        m = _mask(v["c10"], I10[k]) | _mask(v["c9"], I9[k])
        lab[k] |= set(v.loc[m, "MRN"].unique())

    print("[labels] " + "  ".join(f"{k}={len(s):,}" for k, s in lab.items()))
    return lab


def main():
    lab = build_patient_labels()

    V = pd.read_csv(os.path.join(MC, "visits.csv"), low_memory=False,
                    usecols=["MRN", "CSN", "Age", "Gender", "Triage_HR", "Triage_RR",
                             "Triage_SpO2", "Triage_SBP", "Triage_DBP",
                             "Triage_acuity", "ED_dispo", "ED_LOS"])
    V["Age"] = pd.to_numeric(V["Age"], errors="coerce")
    print(f"\n[visits] {len(V):,} visits / {V.MRN.nunique():,} patients")

    W = pd.read_csv(os.path.join(MC, "waveform_summary.csv"))
    print("\n[waveform] by signal type:")
    print(W.groupby("Type").agg(visits=("CSN", "nunique"),
                                median_min=("Duration", lambda s: round(s.median()/60, 1)),
                                total_h=("Duration", lambda s: round(s.sum()/3600))))
    P = (W[W.Type == "Pleth"].groupby("CSN")
         .agg(pleth_s=("Duration", "sum"), pleth_seg=("Segments", "sum")).reset_index())
    E = (W[W.Type == "II"].groupby("CSN").agg(ecg_s=("Duration", "sum")).reset_index())
    V = V.merge(P, on="CSN", how="left").merge(E, on="CSN", how="left")
    V["has_pleth"] = V.pleth_s.notna()

    cirr   = lab["cirrhosis"]
    cld    = cirr | lab["cld_extra"] | lab["phtn"] | lab["varices"] | lab["he"] | lab["hrs"]
    decomp = lab["ascites"] | lab["varices"] | lab["he"] | lab["hrs"] | lab["sbp_perit"]
    V["cirrhosis"] = V.MRN.isin(cirr)
    V["cld"]       = V.MRN.isin(cld)
    V["phtn"]      = V.MRN.isin(lab["phtn"])
    V["decomp"]    = V.cirrhosis & V.MRN.isin(decomp)
    V["comp"]      = V.cirrhosis & ~V.MRN.isin(decomp)
    V["control"]   = ~V.MRN.isin(lab["liver_any"])
    for k in ["hf", "sepsis", "esrd", "pe"]:
        V[k] = V.MRN.isin(lab[k])

    def row(name, m):
        s = V[m]; sp = s[s.has_pleth]
        return dict(group=name, visits=len(s), patients=s.MRN.nunique(),
                    pleth_visits=len(sp), pleth_patients=sp.MRN.nunique(),
                    pleth_median_min=round(sp.pleth_s.median()/60, 1) if len(sp) else 0,
                    median_age=s.Age.median())

    ALL = pd.Series(True, index=V.index)
    tab = pd.DataFrame([
        row("all", ALL), row("cirrhosis", V.cirrhosis),
        row("  decompensated", V.decomp), row("  compensated", V.comp),
        row("chronic liver disease", V.cld), row("portal hypertension", V.phtn),
        row("controls", V.control),
        row("heart failure", V.hf), row("sepsis", V.sepsis),
        row("ESRD", V.esrd), row("pulmonary embolism", V.pe),
    ])
    print("\n=== Cohort ===")
    print(tab.to_string(index=False))

    P_ = V[V.has_pleth].sort_values("pleth_s", ascending=False).drop_duplicates("MRN")
    pos = P_[P_.cirrhosis].dropna(subset=["Age", "Gender"])
    neg = P_[P_.control].dropna(subset=["Age", "Gender"])
    print(f"\n[before matching] cirrhosis {len(pos):,} (median age {pos.Age.median():.0f}) | "
          f"controls {len(neg):,} (median age {neg.Age.median():.0f})")

    sel, used = [], set()
    pool = {g: d.sample(frac=1, random_state=0).to_dict("records")
            for g, d in neg.groupby("Gender")}
    for _, r in pos.sample(frac=1, random_state=1).iterrows():
        c = 0
        for cand in pool.get(r.Gender, []):
            if cand["MRN"] in used:
                continue
            if abs(cand["Age"] - r.Age) <= 3:
                used.add(cand["MRN"]); sel.append(cand); c += 1
                if c == 4:
                    break
    M = pd.DataFrame(sel)
    print(f"[after matching] controls {len(M):,} (median age {M.Age.median():.0f})")

    spec = []
    for k in ["hf", "sepsis", "esrd"]:
        cand = P_[P_[k] & ~P_.cirrhosis & ~P_.MRN.isin(used)].dropna(subset=["Age", "Gender"])
        take = cand.sample(min(len(cand), len(pos)), random_state=2)
        spec.append(take.assign(grp=k))
        print(f"[comparator] {k}: {len(take):,} of {len(cand):,}")

    coh = pd.concat([pos.assign(grp="cirrhosis"), M.assign(grp="control")] + spec,
                    ignore_index=True)

    tot_h = coh.pleth_s.sum() / 3600
    print(f"\n[download] {len(coh):,} visits, {tot_h:,.0f} h of Pleth")

    coh.to_csv(os.path.join(OUT, "mcmed_cohort_visits.csv"), index=False)
    with open(os.path.join(OUT, "mcmed_download_csns.txt"), "w") as f:
        f.write("\n".join(coh.CSN.astype(str).tolist()))
    print(f"\n[saved] {os.path.join(OUT, 'mcmed_cohort_visits.csv')}")
    print(f"[saved] {os.path.join(OUT, 'mcmed_download_csns.txt')}  ({len(coh):,} CSNs)")


if __name__ == "__main__":
    main()
