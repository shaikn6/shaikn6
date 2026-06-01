<div align="center">

# Nagizaaz Shaik
### MLOps & AI Systems Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-nagizaazshaik-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/nagizaazshaik)
[![Email](https://img.shields.io/badge/Email-shaikn6%40udayton.edu-EA4335?style=flat&logo=gmail)](mailto:shaikn6@udayton.edu)
[![GitHub](https://img.shields.io/badge/GitHub-shaikn6-181717?style=flat&logo=github)](https://github.com/shaikn6)

</div>

---

## About

MLOps Engineer with 4+ years building production ML systems in regulated financial environments. I specialize in the full lifecycle — data pipelines to deployed models — with a strong focus on **LLMOps**, **ML observability**, and **human-AI interaction**.

Currently pursuing M.S. at University of Dayton (2024–2026), researching cognitive-adaptive AI systems.

---

## Core Skills

| Layer | Technologies |
|-------|-------------|
| **ML / DL** | PyTorch, scikit-learn, XGBoost, HuggingFace, Federated Learning, Differential Privacy |
| **LLMOps** | LangChain, LangGraph, Evidently AI, RAG pipelines, Prompt versioning, Red-teaming |
| **Data Engineering** | Apache Airflow, Kafka, Redis, Apache Spark, dbt, SQL lineage |
| **Cloud / Infra** | AWS (SageMaker, Redshift, S3, CloudWatch), Docker, FastAPI, MLflow |
| **Observability** | Drift detection, Grad-CAM explainability, SHAP, fairness auditing |

---

## Project Portfolio

> 🔒 All repositories are **private** — source code available on request for verified employers.
> 📧 Contact: shaikn6@udayton.edu

### LLMOps & Generative AI
| Project | Description | Stack |
|---------|-------------|-------|
| `legal-case-rag` | Hybrid RAG over 10 landmark federal court cases — FAISS semantic + TF-IDF keyword + α-weighted fusion; NDCG@5 eval dashboard proving hybrid beats pure semantic | FAISS · scikit-learn · LangChain · Streamlit |
| `on-device-llm-optimizer` | KD pipeline: Phi-3 Mini 3.8B → 1B student on Apple Silicon (MLX); INT4 quant + CoreML export targeting iPhone Neural Engine; 7.6× compression | MLX · coremltools · Streamlit |
| `adaptive-cognitive-rag` | RAG that adapts explanation depth to real-time cognitive load; persistent knowledge graph | FAISS · networkx · Streamlit |
| `clinical-note-llmops` | HIPAA: Presidio PII scrubbing → spaCy NER → ICD-10 extraction → FHIR R4 | spaCy · FastAPI · presidio |
| `finance-llmops-platform` | RAG over SEC 10-K + earnings calls with hallucination grounding & MLflow versioning | LangChain · Evidently · MLflow |
| `prompt-ops` | Git-style LLM prompt versioning, A/B testing with chi-squared significance | FastAPI · sentence-transformers |
| `llm-safety-auditor` | 50-attack adversarial red-teaming, OWASP LLM Top 10 compliance grid | sentence-transformers · ReportLab |
| `rag-memory-bank` | Episodic long-term memory layer for RAG — FAISS + recency decay re-ranking | FAISS · spaCy · SQLite |
| `differential-privacy-llm` | DP-SGD from first principles, Rényi DP budget accounting, privacy-utility tradeoffs | PyTorch · Opacus |
| `llm-code-archaeologist` | LLM excavates git history — intent classification, debt tracking, drift detection | GitPython · KMeans · Plotly |

### ML Systems & Data Engineering
| Project | Description | Stack |
|---------|-------------|-------|
| `sql-to-dag-compiler` | Oracle SQL/PLSQL → Airflow 2.x DAG auto-generation | sqlparse · Jinja2 · Airflow |
| `kafka-stream-feature-store` | Real-time feature store: Kafka → Redis, sub-60s freshness, FastAPI serving | Kafka · Redis · FastAPI |
| `agentic-pipeline-healer` | LangGraph agent monitors Airflow failures, diagnoses & auto-applies fixes | LangGraph · Airflow · AST |
| `ml-fairness-audit` | ECOA/Fair Lending bias detection, SHAP explanations, PDF audit report | fairlearn · SHAP · ReportLab |
| `federated-credit-risk` | 3-institution federated training, zero raw data sharing, Flower FedAvg | Flower · PyTorch · FastAPI |
| `evidently-llm-sentinel` | Extends Evidently AI to LLM output: semantic drift, hallucination risk | Evidently · Grafana |
| `redshift-wlm-optimizer` | XGBoost predicts query cost, auto-routes to WLM queue, SHAP explanations | XGBoost · SHAP · FastAPI |
| `etl-lineage-graph` | Column-level SQL lineage via CTE-aware parser + networkx graph | sqlparse · networkx · Plotly |

### Healthcare & Domain AI
| Project | Description | Stack |
|---------|-------------|-------|
| `medical-imaging-ai` | Chest X-ray CNN + Grad-CAM explainability: Pneumonia/Cardiomegaly/Effusion | PyTorch · Grad-CAM · Streamlit |
| `healthcare-claims-anomaly` | Insurance fraud: Isolation Forest + Autoencoder ensemble, SHAP explainability | PyTorch · SHAP · FastAPI |
| `regulatory-compliance-copilot` | Policy gap RAG across 5 compliance domains, risk scoring, remediation roadmap | FAISS · ReportLab · Streamlit |
| `cognitive-load-adaptive-ai` | Keystroke dynamics → cognitive load → adaptive LLM response verbosity | RandomForest · FastAPI |

### Systems & Domain Showcases
| Project | Description | Stack |
|---------|-------------|-------|
| `high-traffic-ticket-engine` | Redis Lua atomic inventory, 10K concurrent, 0 oversells (Taylor Swift-scale) | Redis · FastAPI · fakeredis |
| `flight-ops-intelligence` | ML delay predictor, weather analysis, interactive Folium map (500 flights) | RF · Folium · geopy |
| `stellar-birth-chart` | Keplerian orbital mechanics: real planetary positions for any birth date | Skyfield · Plotly · Streamlit |
| `ocean-sail-navigator` | Dijkstra over lat/lon grid with J/24 polar diagram + storm avoidance | NetworkX · Folium · SciPy |
| `sign-language-translator` | MediaPipe + PyTorch ASL recognition, 96.7% accuracy — B.Tech CV project | MediaPipe · PyTorch · OpenCV |
| `music-mood-engine` | librosa audio features → KMeans mood clusters → playlist recommender | librosa · UMAP · scikit-learn |

---

## Experience Highlights

**MLOps Engineer** — Financial Services (India, 2019–2023)
- Migrated Oracle → Redshift: built SQL-to-DAG compiler processing 200+ stored procedures
- Deployed real-time feature store (Kafka → Redis) serving 50K predictions/day
- Built XGBoost WLM query optimizer reducing financial report runtime by 85%

**Graduate Researcher** — University of Dayton (2024–2026)
- Researching cognitive-adaptive AI: how LLMs should modulate complexity based on user state
- Work on federated learning for privacy-preserving credit risk assessment

---

## Currently

- F-1 STEM OPT eligible (authorized to work in the US)
- Open to: MLOps Engineer, AI/ML Engineer, Data Engineer, LLMOps roles
- Based in Ohio — remote preferred, open to US relocation

---

<div align="center">
<sub>📧 shaikn6@udayton.edu &nbsp;·&nbsp; All project repos available on request for verified employers</sub>
</div>
