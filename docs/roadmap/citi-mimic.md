# CITI Training → MIMIC-III Access Path

## Step 1: CITI Training (IN PROGRESS)

User enrolled 2026-05-30. Three required courses:

| Course | Status |
|--------|--------|
| Biomedical Research | In progress |
| Biomedical Responsible Conduct of Research (RCR) | Not started |
| HIPS for Researchers (Health Information Privacy & Security) | Not started |

Estimated completion: ~3–4 hours total.

## Step 2: PhysioNet Account + MIMIC-III Request

After all 3 CITI certificates:
1. Go to physionet.org → create account
2. Upload all 3 CITI completion certificates
3. Submit MIMIC-III credentialed access request
4. Approval time: typically 1–7 days

## Step 3: Download MIMIC-III

After approval:
```bash
# Install PhysioNet client
pip install wfdb

# Download via wget with credentialed token
wget -r -N -c -np --user <username> --ask-password \
  https://physionet.org/files/mimiciii/1.4/
```

Key tables needed:
- `PATIENTS` — demographics, DOB, DOD
- `ADMISSIONS` — hadm_id, admit time, discharge time, hospital expire flag
- `ICUSTAYS` — ICU admission windows
- `CHARTEVENTS` — vitals (130M+ rows — use chunked loading)
- `LABEVENTS` — lab values
- `DIAGNOSES_ICD` — ICD-9 diagnosis codes

## Step 4: Build 3 MIMIC Projects

See `clinical-projects.md` for full specs.
