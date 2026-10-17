"""Segment index and diagnosis labels for MIMIC-III-Ext-PPG.

Outputs (derived/): segments_index.parquet, cohort_patients.csv
"""
import os, warnings
import pandas as pd

PPG_DIR   = os.environ.get("MIMIC_PPG_DIR", "data/mimic-iii-ext-ppg")
MIMIC_DIR = os.environ.get("MIMIC_DIR", "data/mimic-iii-clinical-database-1.4")
OUT_DIR   = os.path.join(os.environ.get("PPG_ROOT", "."), "derived")
META      = os.path.join(PPG_DIR, "metadata.csv")
CHUNK     = 1_000_000
os.makedirs(OUT_DIR, exist_ok=True)

COLS = ["signal_file_name","folder_path","start_segment","event_rhythm",
        "record_id","event_id","segment_id","clinical_information_system",
        "median_30s_hr","iqr_30s_hr","median_30s_rr","iqr_30s_rr",
        "median_30s_sbp","iqr_30s_sbp","median_30s_dbp","iqr_30s_dbp",
        "vector_10s_pleth_sqi","vector_10s_ecg_sqi","vector_10s_abp_sqi","resp_sqi",
        "subject_id","hadm_id","icustay_id",
        "age","weight","height","gender","ethnicity","strat_fold"]
NUM = ["median_30s_hr","iqr_30s_hr","median_30s_rr","iqr_30s_rr",
       "median_30s_sbp","iqr_30s_sbp","median_30s_dbp","iqr_30s_dbp",
       "resp_sqi","age","weight","height"]

CIRRHOSIS = {"5712","5715","5716"}
CLD = CIRRHOSIS | {"5710","5711","5713","57140","57141","57142","57149","5718","5719",
                   "5722","5723","5724","5728","45620","45621","4560","4561",
                   "07032","07033","07044","07054","07070","07071"}
DECOMP = {"7895","78959","4560","45620","5722","5724","56723"}
GROUPS = {"cirrhosis":CIRRHOSIS, "cld":CLD, "phtn":{"5723"},
          "ascites":{"7895","78959"}, "he":{"5722"},
          "var_bleed":{"4560","45620"}, "hrs":{"5724"},
          "sbp_peritonitis":{"56723"}, "decomp":DECOMP,
          "hf":{"4280","4281","42820","42821","42822","42823","42830","42831",
                "42832","42833","42840","42841","42842","42843","4289"},
          "sepsis":{"99591","99592","78552"}, "esrd":{"5856"},
          "pe":{"41511","41512","41513","41519"}}

def build_labels():
    dx = pd.read_csv(os.path.join(MIMIC_DIR,"DIAGNOSES_ICD.csv"),
                     usecols=["SUBJECT_ID","ICD9_CODE"],
                     dtype={"SUBJECT_ID":"int32","ICD9_CODE":"string"})
    dx["ICD9_CODE"] = dx["ICD9_CODE"].fillna("").str.strip()
    L = {k:set(dx.loc[dx.ICD9_CODE.isin(v),"SUBJECT_ID"].unique()) for k,v in GROUPS.items()}
    print("[labels] " + "  ".join(f"{k}={len(v)}" for k,v in L.items()))
    return L

def parse_sqi(series, tag):
    s = series.astype("string")
    uniq = s.dropna().unique()
    ng, ag = {}, {}
    for u in uniq:
        try:
            v = [x.strip() for x in str(u).strip("[]").split(",")]
            n = sum(1 for x in v if x == "1")
            ng[u] = n; ag[u] = int(n == len(v) and n > 0)
        except Exception:
            ng[u] = 0; ag[u] = 0
    return (s.map(ng).astype("float32").fillna(0).astype("int8"),
            s.map(ag).astype("float32").fillna(0).astype("int8"))

def main():
    L = build_labels()
    print(f"\n[scan] {META}")
    parts, n = [], 0
    for i, ch in enumerate(pd.read_csv(META, usecols=COLS, chunksize=CHUNK, low_memory=False)):
        parts.append(ch); n += len(ch)
        print(f"   chunk {i:>3}  total {n:,}", flush=True)
    df = pd.concat(parts, ignore_index=True); del parts
    print(f"[scan] {len(df):,} segments / {df.subject_id.nunique():,} patients")

    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["subject_id"]  = df.subject_id.astype("int32")
    df["strat_fold"]  = pd.to_numeric(df.strat_fold, errors="coerce").astype("int8")
    df["start_segment"] = pd.to_datetime(df.start_segment, errors="coerce")

    for tag, col in [("ppg","vector_10s_pleth_sqi"),("ecg","vector_10s_ecg_sqi"),
                     ("abp","vector_10s_abp_sqi")]:
        ng, ag = parse_sqi(df[col], tag)
        df[f"{tag}_n_good"], df[f"{tag}_all_good"] = ng, ag
        df[f"has_{tag}"] = df[col].astype("string").fillna("nan").ne("[nan, nan, nan]").astype("int8")
        df.drop(columns=[col], inplace=True)
        print(f"   {tag}: present {df[f'has_{tag}'].mean():.1%}  all windows good {df[f'{tag}_all_good'].mean():.1%}")

    sid = df.subject_id
    for k, s in L.items():
        df[f"lbl_{k}"] = sid.isin(s).astype("int8")
    df["lbl_cirr_decomp"] = ((df.lbl_cirrhosis==1)&(df.lbl_decomp==1)).astype("int8")
    df["lbl_cirr_comp"]   = ((df.lbl_cirrhosis==1)&(df.lbl_decomp==0)).astype("int8")
    df["lbl_control"]     = (df.lbl_cld==0).astype("int8")

    p = os.path.join(OUT_DIR,"segments_index.parquet")
    df.to_parquet(p, index=False)
    print(f"\n[out] {p}  ({os.path.getsize(p)/1e6:.0f} MB)")

    agg = {"n_seg":("subject_id","size"),
           "n_ppg_good":("ppg_all_good","sum"), "n_ecg_good":("ecg_all_good","sum"),
           "frac_has_abp":("has_abp","mean"), "frac_has_ecg":("has_ecg","mean"),
           "hr":("median_30s_hr","median"), "sbp":("median_30s_sbp","median"),
           "dbp":("median_30s_dbp","median"), "rr":("median_30s_rr","median"),
           "age":("age","median"), "gender":("gender","first"),
           "fold":("strat_fold","first"), "cis":("clinical_information_system","first")}
    agg.update({c:(c,"max") for c in df.columns if c.startswith("lbl_")})
    pat = df.groupby("subject_id").agg(**agg).reset_index()
    pp = os.path.join(OUT_DIR,"cohort_patients.csv")
    pat.to_csv(pp, index=False)
    print(f"[out] {pp}")

    print("\n=== Patients ===")
    for c in sorted([c for c in pat.columns if c.startswith("lbl_")]):
        print(f"  {c:<20} {int(pat[c].sum()):>6}")
    print(f"\nmedian segments per patient {pat.n_seg.median():.0f}")
    print(f"median high-quality PPG segments per patient {pat.n_ppg_good.median():.0f}  "
          f"(>=10: {int((pat.n_ppg_good>=10).sum())})")
    print(f"CareVue/Metavision:\n{pat.cis.value_counts().to_string()}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore"); main()
