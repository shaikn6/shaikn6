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

## Timeline

| Milestone | Estimated Date |
|-----------|---------------|
| CITI training complete | 2026-05-30 to 2026-06-02 |
| PhysioNet access approved | 2026-06-03 to 2026-06-10 |
| ICU Mortality Predictor | Week of 2026-06-10 |
| Sepsis Early Warning | Week of 2026-06-17 |
| 30-Day Readmission | Week of 2026-06-24 |
| All 3 public on GitHub | 2026-06-30 |
