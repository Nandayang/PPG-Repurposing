# ppg-cirrhosis

Analysis code for **"Repurposing routine pulse oximetry waveforms for cirrhosis detection across health systems"**.

The study tests whether the photoplethysmography (PPG) waveform recorded by routine pulse oximeters carries information on cirrhosis, using intensive care (MIMIC-III-Ext-PPG) and emergency department (MC-MED) recordings.

This first release contains cohort construction, data preparation, handcrafted waveform features, and the statistical analyses and figures computed from patient-level predictions. Code for the beat-matrix representation, self-supervised pretraining, model training and inference will be added after peer review.

The repository contains code only: no data, no patient-level outputs and no model weights.

## Data access

All data are available from PhysioNet to credentialed users under the respective data use agreements and cannot be redistributed:

| Dataset | Version | Used for |
|---|---|---|
| [MIMIC-III-Ext-PPG](https://physionet.org/content/mimic-iii-ext-ppg/) | 1.1.0 | ICU PPG / ECG segments |
| [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/) | 1.4 | ICU diagnoses, laboratory values |
| [MC-MED](https://physionet.org/content/mc-med/) | 1.0.1 | ED pulse oximetry waveforms and clinical tables |

## Setup

Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
```

Paths are set with environment variables (defaults in brackets):

| Variable | Meaning |
|---|---|
| `PPG_ROOT` [`.`] | Working folder; scripts create `cache/`, `derived/` and `results/` in it |
| `MIMIC_PPG_DIR` [`data/mimic-iii-ext-ppg`] | MIMIC-III-Ext-PPG |
| `MIMIC_DIR` [`data/mimic-iii-clinical-database-1.4`] | MIMIC-III Clinical Database (CSV files) |
| `MCMED_DIR` [`data/mc-med`] | MC-MED clinical tables; waveforms in `<MCMED_DIR>/data` |
| `PHYSIONET_USERNAME` | Optional, for `05_mcmed_download.py` (the password is prompted and not stored) |

Scripts in `pipeline/` load one another by file name, so keep them in one folder and run them as `python pipeline/<script>.py`. Each script starts with a short description of what it does and writes.

## Included

**Cohorts and data preparation**
```bash
python pipeline/01_build_cohort.py
python pipeline/02_cache_waveforms.py
python pipeline/04_ppg_hr.py
python pipeline/05_add_quality_flags.py
python pipeline/12_specificity.py --apply
python pipeline/11_fib4_headtohead.py
python pipeline/04_mcmed_census.py
python pipeline/05_mcmed_download.py --download
python pipeline/13_mcmed_cache.py
python pipeline/17_multilabel_prep.py --data mimic
```

**Handcrafted waveform features**
```bash
python pipeline/27_morph_hrv.py --db mimic
python pipeline/27_morph_hrv.py --db mcmed
python pipeline/29_morph_effects.py
python pipeline/30_input_ablation.py
python pipeline/31_deep_vs_handcrafted.py
```

**Statistics**
```bash
python pipeline/21_paper_analyses.py
python pipeline/21b_compensated_ops.py
python pipeline/22c_sampling_gains.py
python pipeline/23_mcmed_acute_decomp.py
python pipeline/28_fib4_reference.py
python pipeline/34_supplementary_tables.py
```
`07_morphology.py`, `10_clinical_utility.py`, `24_figure_inputs.py` and `stats_common.py` are imported by other scripts.

The statistics, `30_input_ablation.py` and `31_deep_vs_handcrafted.py` read the patient-level predictions of the deep models in `results/` (`es/`, `mcmed/`, `reverse/`, `spot/`), which are produced by the model code of the later release.

**Figures**

Each folder in `figures/` holds one drawing script that reads summary tables from `figures/data/` and writes to its own `out/` folder. The tables are written by `figures/builders/` and `31_deep_vs_handcrafted.py`; those for the liver-disease spectrum (Fig. 5), emulated degradation (Fig. 7) and the multilabel model (Supplementary Fig. 2) come from scripts of the later release named in each drawing script.
```bash
python figures/builders/make_plot_data.py
python figures/builders/make_fig3_age45_data.py "$PPG_ROOT" figures
python figures/builders/make_fig3_mechanism_data.py "$PPG_ROOT"
python figures/builders/make_meld_panel_data.py --root "$PPG_ROOT"
python figures/fig3_features/draw_fig3.py
```

## Reproducibility

- Patient-level splits are stored once (`derived/split712_<data>_<task>.csv`), so all models and baselines share them.
- Internal AUCs of the deep models use within-fold percentile ranks pooled across test folds; logistic baselines use pooled cross-validated probabilities.
- Confidence intervals are patient-level bootstrap percentiles with fixed seeds.

## License

MIT (see `LICENSE`).
