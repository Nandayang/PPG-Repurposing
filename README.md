# ppg-cirrhosis

Analysis code for **"Repurposing routine pulse oximetry waveforms for cirrhosis detection across health systems"**.

The study tests whether the photoplethysmography (PPG) waveform recorded by routine pulse oximeters carries information on cirrhosis, using intensive care (MIMIC-III-Ext-PPG) and emergency department (MC-MED) recordings. Each 30-s window is folded into a fixed-time beat matrix, an encoder is pretrained by masked reconstruction with masks defined on the original time axis, and the model is fine-tuned to separate patients with cirrhosis from age- and sex-matched controls, with internal cross-validation and frozen cross-system evaluation.

Partial code has been released, and the full code will be made publicly available upon publication.

## Data access

All data are available from PhysioNet to credentialed users under the respective data use agreements, and cannot be redistributed here:

| Dataset | Version | Used for |
|---|---|---|
| [MIMIC-III-Ext-PPG](https://physionet.org/content/mimic-iii-ext-ppg/) | 1.1.0 | ICU PPG / ECG segments, pretraining corpus |
| [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/) | 1.4 | ICU diagnoses, laboratory values |
| [MC-MED](https://physionet.org/content/mc-med/) | 1.0.1 | ED pulse oximetry waveforms and clinical tables |
