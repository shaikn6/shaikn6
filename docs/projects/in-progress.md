# Projects In Progress

## Status: BOTH COMPLETE — 2026-05-30

---

## 1. chexpert-pathology-classifier

**Goal:** CheXpert benchmark chest X-ray classifier — real published AUCs to compare against.

**Stack:** DenseNet121 · PyTorch · Grad-CAM · CheXpert dataset (224K images, 14 labels)

**Result: 131/131 tests passing.**

| File | Purpose |
|------|---------|
| `src/model.py` | DenseNet121, selective layer freezing, 14-label sigmoid head |
| `src/data.py` | CheXpert dataset + mock generator (no download needed) |
| `src/train.py` | AdamW + cosine LR + early stopping + MLflow |
| `src/evaluate.py` | Per-label AUC vs Irvin 2019, calibration plots |
| `src/gradcam.py` | Hook-based Grad-CAM on denseblock4, top-k saliency overlays |
| `src/api.py` | FastAPI /predict + /predict/batch + /health + /labels |

**Why it matters for FAANG:** Real benchmark dataset, real published AUC comparison = reproducible research standard.

---

## 2. clinical-survival-analysis

**Goal:** Production-grade survival analysis on real public clinical datasets.

**Stack:** lifelines · scikit-survival · PyTorch · Kaplan-Meier · Cox PH · RSF · XGBoost survival

**Datasets:**
- WHAS500 (Worcester Heart Attack Study — real, public, no approval needed)
- GBSG (German Breast Cancer Study Group — real, public)
- Synthetic ICU cohort calibrated to MIMIC-III population statistics

**Result: 120/120 tests passing.**

- 4 models: KM · Cox PH · RSF (100 trees) · XGBoost survival
- Real datasets: NCCTG Lung Cancer + GBSG2 + synthetic ICU (MIMIC-calibrated, ~22% event rate)
- Metrics: C-index, Integrated Brier Score, time-dependent AUC at 30/90/180/365 days
- Streamlit dashboard (5 tabs) + FastAPI `/predict` endpoint
- Agent fixed 5 version-mismatch bugs (lifelines 0.30, sksurv 0.26)

**Why it matters for FAANG:** Survival analysis = gold standard in clinical outcome modeling.

---

## Next Projects (post MIMIC-III access)

See `roadmap/clinical-projects.md` — 3 ICU projects require MIMIC-III credentialed access.
