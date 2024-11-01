<div align="center">

# Nagizaaz Shaik

**Data Engineer → MLOps Engineer → ML / AI Engineer**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-nagizaazshaik-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/nagizaazshaik)
[![GitHub](https://img.shields.io/badge/GitHub-shaikn6-181717?style=flat&logo=github)](https://github.com/shaikn6)
[![Portfolio](https://img.shields.io/badge/Portfolio-shaikn6.github.io%2Fportfolio-0078d4?style=flat&logo=github)](https://shaikn6.github.io/portfolio)

</div>

---

## About

Started as a **Data Engineer** building production ETL pipelines and cloud warehouses. Grew into **MLOps** — deploying, monitoring, and maintaining ML systems in regulated financial environments. Now focused on **ML Engineering and AI** — LLMs, RAG systems, AI agents, and clinical AI.

---

## Skills

| Domain | Technologies |
|--------|-------------|
| **Machine Learning** | scikit-learn · XGBoost · Random Forest · Gradient Boosting · SHAP · fairlearn |
| **Deep Learning** | PyTorch · DenseNet121 · EfficientNet-B4 · CNNs · Autoencoders · Grad-CAM · ScoreCAM · Federated Learning |
| **NLP** | spaCy · BERT · BioBERT · HuggingFace Transformers · sentence-transformers · NER · ICD-10 extraction |
| **LLMs & RAG** | LangChain · LangGraph · FAISS · RAG pipelines · Prompt engineering · Hallucination detection |
| **AI Agents** | LangGraph state machines · Agentic workflows · Tool use · Episodic memory · Red-teaming |
| **Clinical AI** | Survival analysis · Cox PH · DeepSurv · DeepHit · Chest X-ray classification · DICOM · FHIR R4 |
| **MLOps** | MLflow · Evidently AI · Drift detection · Model monitoring · A/B testing · SHAP explainability |
| **Data Engineering** | Apache Airflow · Kafka · Redis · Apache Spark · dbt · SQL lineage · ETL/ELT |
| **Cloud & Infra** | AWS (SageMaker · Redshift · S3 · CloudWatch) · Docker · FastAPI · PostgreSQL · SQLite |
| **Security & Governance** | HIPAA · GDPR · ECOA / Fair Lending · PII detection & redaction · Differential Privacy · Data lineage · Audit logging · OWASP LLM Top 10 |

---

## Certifications

- AWS Certified Machine Learning – Specialty
- AWS Certified Cloud Practitioner
- GCP Professional Data Engineer
- Snowflake SnowPro Advanced: Data Engineer
- dbt Analytics Engineering Certification

---

## Public Repositories

10 production-grade projects — all security audited (58 issues resolved).

| Repo | Description | Stack |
|------|-------------|-------|
| [`chexpert-pathology-classifier`](https://github.com/shaikn6/chexpert-pathology-classifier) | DenseNet121 + EfficientNet-B4 · CheXpert 14-label benchmark · MC Dropout · Grad-CAM · ScoreCAM · DICOM | PyTorch · pydicom · FastAPI |
| [`clinical-survival-analysis`](https://github.com/shaikn6/clinical-survival-analysis) | 6 models: KM · Cox PH · RSF · XGBoost · DeepSurv · DeepHit · competing risks · Aalen-Johansen CIF | lifelines · scikit-survival · PyTorch |
| [`agentic-pipeline-healer`](https://github.com/shaikn6/agentic-pipeline-healer) | LangGraph multi-DAG orchestrator · AST-level fixes · Slack Block Kit · SQLite audit log | LangGraph · Airflow · FastAPI |
| [`llm-safety-auditor`](https://github.com/shaikn6/llm-safety-auditor) | 250+ adversarial attacks · 6 mutation strategies · OWASP LLM Top 10 · PDF audit report | sentence-transformers · ReportLab |
| [`sql-to-dag-compiler`](https://github.com/shaikn6/sql-to-dag-compiler) | Oracle SQL/PLSQL + dbt → Airflow 2.x DAGs · Mermaid/DOT lineage export | sqlparse · Jinja2 · Airflow |
| [`flight-ops-intelligence`](https://github.com/shaikn6/flight-ops-intelligence) | ML delay predictor · Open-Meteo live weather · FastAPI · Folium route risk map | XGBoost · FastAPI · Folium |
| [`medical-imaging-ai`](https://github.com/shaikn6/medical-imaging-ai) | Chest X-ray CNN · Grad-CAM · ScoreCAM · EfficientNet-B4 · DICOM pipeline · PHI scrubbing | PyTorch · pydicom · Streamlit |
| [`federated-credit-risk`](https://github.com/shaikn6/federated-credit-risk) | 3-institution Flower FedAvg · zero raw data sharing · model poisoning guard · ECOA audit | Flower · PyTorch · FastAPI |
| [`kafka-stream-feature-store`](https://github.com/shaikn6/kafka-stream-feature-store) | Kafka → Redis · sub-60s freshness · JSON serialization · FastAPI serving layer | Kafka · Redis · FastAPI |
| [`clinical-note-llmops`](https://github.com/shaikn6/clinical-note-llmops) | HIPAA · Presidio PII scrubbing → BioBERT NER → ICD-10 → FHIR R4 · audit logging | spaCy · presidio · FastAPI |

---

## Full Project Portfolio

### Clinical AI

| Project | Description | Stack |
|---------|-------------|-------|
| [`chexpert-pathology-classifier`](https://github.com/shaikn6/chexpert-pathology-classifier) | CheXpert benchmark: DenseNet121 + EfficientNet-B4 · 14-label · Grad-CAM + ScoreCAM · MC Dropout · DICOM · clinical report generator | PyTorch · pydicom · FastAPI |
| [`clinical-survival-analysis`](https://github.com/shaikn6/clinical-survival-analysis) | 6-model comparison: KM · Cox PH · RSF · XGBoost survival · DeepSurv · DeepHit · competing risks · NCCTG + GBSG2 + synthetic ICU | lifelines · scikit-survival · PyTorch |
| [`medical-imaging-ai`](https://github.com/shaikn6/medical-imaging-ai) | Chest X-ray pathology detection · Grad-CAM + ScoreCAM · EfficientNet-B4 · DICOM pipeline · PHI scrubbing | PyTorch · pydicom · Streamlit |

### LLMOps & Generative AI

| Project | Description | Stack |
|---------|-------------|-------|
| `adaptive-cognitive-rag` | RAG adapts explanation depth to real-time cognitive load via keystroke dynamics | FAISS · networkx · Streamlit |
| [`clinical-note-llmops`](https://github.com/shaikn6/clinical-note-llmops) | HIPAA: Presidio PII scrubbing → BioBERT NER → ICD-10 extraction → FHIR R4 output | spaCy · presidio · FastAPI |
| `finance-llmops-platform` | RAG over SEC 10-K + earnings calls with hallucination grounding & MLflow versioning | LangChain · Evidently · MLflow |
| `prompt-ops` | Git-style LLM prompt versioning with A/B testing and chi-squared significance | FastAPI · sentence-transformers |
| [`llm-safety-auditor`](https://github.com/shaikn6/llm-safety-auditor) | 250+ auto-generated adversarial attacks · 6 mutation strategies · OWASP LLM Top 10 scoring · PDF audit report | sentence-transformers · ReportLab |
| `rag-memory-bank` | Episodic long-term memory for RAG agents — FAISS + recency-decay re-ranking | FAISS · spaCy · SQLite |
| `differential-privacy-llm` | DP-SGD from first principles, Rényi DP budget accounting, privacy-utility tradeoffs | PyTorch · Opacus |
| `llm-code-archaeologist` | LLM mines git history for intent classification, tech debt scoring, commit clustering | GitPython · KMeans · Plotly |

### ML Systems & Data Engineering

| Project | Description | Stack |
|---------|-------------|-------|
| [`sql-to-dag-compiler`](https://github.com/shaikn6/sql-to-dag-compiler) | Oracle SQL/PLSQL + dbt models → Airflow 2.x DAGs · edge-case handler · Mermaid/DOT lineage export | sqlparse · Jinja2 · Airflow |
| [`kafka-stream-feature-store`](https://github.com/shaikn6/kafka-stream-feature-store) | Real-time feature store: Kafka → Redis, sub-60s freshness, FastAPI serving layer | Kafka · Redis · FastAPI |
| [`agentic-pipeline-healer`](https://github.com/shaikn6/agentic-pipeline-healer) | LangGraph agent monitors N Airflow DAGs · diagnoses failures · AST fixes · Slack alerts · SQLite audit log | LangGraph · Airflow · AST |
| `ml-fairness-audit` | ECOA/Fair Lending bias detection, SHAP explanations, automated PDF audit report | fairlearn · SHAP · ReportLab |
| [`federated-credit-risk`](https://github.com/shaikn6/federated-credit-risk) | 3-institution federated training · zero raw data sharing · model poisoning guard · ECOA compliance | Flower · PyTorch · FastAPI |
| `evidently-llm-sentinel` | Extends Evidently AI to LLM outputs: semantic drift + hallucination risk scoring | Evidently · Grafana |
| `redshift-wlm-optimizer` | XGBoost predicts query cost, auto-routes to WLM queue with SHAP explanations | XGBoost · SHAP · FastAPI |
| `etl-lineage-graph` | Column-level SQL lineage via CTE-aware parser + interactive networkx graph | sqlparse · networkx · Plotly |
| `healthcare-claims-anomaly` | Insurance fraud: Isolation Forest + Autoencoder ensemble with SHAP explainability | PyTorch · SHAP · FastAPI |

### AI Research & Cognitive Systems

| Project | Description | Stack |
|---------|-------------|-------|
| `cognitive-load-adaptive-ai` | Keystroke dynamics → real-time cognitive load → adaptive LLM response verbosity | RandomForest · FastAPI |
| `regulatory-compliance-copilot` | Policy gap RAG across 5 compliance domains, risk scoring, remediation roadmap | FAISS · ReportLab · Streamlit |
| `ml-drift-monitoring` | PSI + KS test drift detection with automated statistical retraining gates | scikit-learn · Evidently |
| `mlops-retraining-pipeline` | Welch t-test + chi-squared gates → drift-triggered automated model retraining | Airflow · MLflow · FastAPI |

### Domain Showcases

| Project | Description | Stack |
|---------|-------------|-------|
| [`flight-ops-intelligence`](https://github.com/shaikn6/flight-ops-intelligence) | ML delay predictor · Open-Meteo live weather · real-time FastAPI endpoint · Folium route risk map | XGBoost · FastAPI · Folium |
| `high-traffic-ticket-engine` | Redis Lua atomic inventory, 10K concurrent users, zero oversell guarantee | Redis · FastAPI · Kafka |
| `sign-language-translator` | MediaPipe + PyTorch ASL recognition — 96.7% accuracy, real-trained CNN | MediaPipe · PyTorch · OpenCV |
| `stellar-birth-chart` | Keplerian orbital mechanics: precise planetary positions for any birth date/location | Skyfield · Plotly · Streamlit |
| `ocean-sail-navigator` | Dijkstra over lat/lon grid + J/24 polar diagram + storm-avoidance routing | networkx · Folium · SciPy |
| `music-mood-engine` | librosa audio features → UMAP → KMeans mood clusters → playlist recommender | librosa · UMAP · scikit-learn |
| `ant-colony-intelligence` | ACO for TSP, pheromone stigmergy simulation, quorum sensing model | numpy · networkx · Streamlit |
| `microbial-world-simulator` | Monod kinetics, phage Lotka-Volterra ODE, Steele algae photoinhibition model | scipy · matplotlib · Streamlit |
| `dayton-air-quality-intelligence` | SARIMA AQI forecasting + calendar effects (holidays = -25% PM2.5) + Folium hotspot map | statsmodels · Folium · Streamlit |
| `dayton-airport-analytics` | DAY airport: airline rankings, delay cause breakdown, passenger heatmap, route map | pandas · Folium · FastAPI |
| `ocean-health-monitor` | Ocean plastic biomagnification through marine food web + OHI scenario projections to 2050 | networkx · scipy · Folium |
| `human-embryo-genetics` | Mendelian inheritance engine, 40-week developmental timeline, sex determination model | numpy · plotly · Streamlit |

---

<div align="center">
<sub>10 public repositories above · remaining repos available on request · <a href="https://linkedin.com/in/nagizaazshaik">LinkedIn</a> · <a href="https://shaikn6.github.io/portfolio">Portfolio</a></sub>
</div>
