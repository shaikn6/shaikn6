# Projects In Progress

## Status: Both agents hit session limit (3:40pm ET) — need relaunch

---

## 1. chexpert-pathology-classifier

**Goal:** CheXpert benchmark chest X-ray classifier — real published AUCs to compare against.

**Stack:** DenseNet121 · PyTorch · Grad-CAM · CheXpert dataset (224K images, 14 labels)

**Planned deliverables:**
- DenseNet121 fine-tuned on CheXpert (or reproduced from scratch on subset)
- 14-label multi-label classification (Atelectasis, Cardiomegaly, Effusion, etc.)
- Grad-CAM saliency maps per label
- AUC table comparing against Irvin et al. 2019 published benchmarks
- 50+ unit + integration tests
- CHANGELOG.md, MIT license, Quick Start

**Why it matters for FAANG:** Real benchmark dataset, real published comparison = reproducible research standard.

---

## 2. clinical-survival-analysis

**Goal:** Production-grade survival analysis on real public clinical datasets.

**Stack:** lifelines · scikit-survival · PyTorch · Kaplan-Meier · Cox PH · RSF · XGBoost survival

**Datasets:**
- WHAS500 (Worcester Heart Attack Study — real, public, no approval needed)
- GBSG (German Breast Cancer Study Group — real, public)
- Synthetic ICU cohort calibrated to MIMIC-III population statistics

**Planned deliverables:**
- 4-model comparison: KM · Cox PH · Random Survival Forest · XGBoost survival
- C-index, IBS, time-dependent AUC metrics
- Calibration plots + feature importance (RSF permutation)
- Interactive Streamlit dashboard
- 60+ tests
- CHANGELOG.md, MIT license, Quick Start

**Why it matters for FAANG:** Survival analysis = gold standard in clinical outcome modeling (Google Health, Apple Health all use it).

---

## Next Step

Relaunch both agents now that session limit has reset.
