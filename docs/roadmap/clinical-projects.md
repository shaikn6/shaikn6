# Clinical Projects — Post MIMIC-III Access

Three projects planned. All require MIMIC-III credentialed access.

---

## 1. ICU Mortality Predictor

**Repo:** `icu-mortality-predictor`

**Goal:** Predict 30-day ICU mortality using first 24h of chart data.

**Stack:** MIMIC-III · XGBoost · SHAP · FastAPI · Streamlit

**Features:**
- Vitals: HR, MAP, SpO2, temp, RR, GCS
- Labs: lactate, creatinine, BUN, WBC, hemoglobin
- SAPS-II / SOFA score replication as baseline
- XGBoost vs Logistic Regression vs Random Forest comparison
- SHAP waterfall plots per patient
- FastAPI `/predict` endpoint
- Streamlit ICU dashboard

**Target metrics:** AUC > 0.85 (matches published MIMIC baselines)

---

## 2. Sepsis Early Warning System

**Repo:** `sepsis-early-warning`

**Goal:** Time-series early warning for sepsis onset (Sepsis-3 definition).

**Stack:** MIMIC-III · LSTM · PyTorch · Kafka (simulated streaming) · FastAPI

**Features:**
- Sepsis-3 labeling from MIMIC-III (qSOFA + organ dysfunction)
- LSTM sequence classifier on rolling 6h windows
- Simulated real-time streaming via Kafka
- Alert threshold tuning (precision/recall tradeoff)
- Calibration curve + reliability diagram

**Target metrics:** Sensitivity > 0.80 at 6h before onset

---

## 3. 30-Day Readmission Predictor

**Repo:** `hospital-readmission-predictor`

**Goal:** Predict 30-day hospital readmission risk at discharge.

**Stack:** MIMIC-III · BioBERT · XGBoost · fairlearn · FastAPI

**Features:**
- Discharge summary NLP via BioBERT (clinical text features)
- Structured features: diagnosis codes, LOS, comorbidities
- Hybrid model: BioBERT embeddings + XGBoost tabular
- ECOA fairness audit (race, gender, insurance subgroups)
- SHAP explanations + PDF report generation

**Target metrics:** AUC > 0.78 (matches CMS published readmission model baselines)

---

## Implementation Status

| Repo | Code | Tests | MIMIC Integration |
|------|------|-------|-------------------|
| icu-mortality-predictor | ✅ Complete | ✅ 48/48 pass | Awaiting data |
| sepsis-early-warning | ✅ Complete | ✅ 75/75 pass | Awaiting data |
| hospital-readmission-predictor | ✅ Complete | ✅ 75/75 pass | Awaiting data |

## Timeline

| Milestone | Status |
|-----------|--------|
| Code scaffolded, tests passing (synthetic) | ✅ 2026-05-31 |
| CITI training complete | In progress |
| PhysioNet access approved | ~1-7 days after CITI |
| Connect MIMIC data → validate real AUC | Next after access |
| All 3 public on GitHub | ~2026-06-30 |
