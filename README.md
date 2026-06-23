<div align="center">

# Nagizaaz Shaik

[![Typing SVG](https://readme-typing-svg.demolab.com/?font=Fraunces&size=30&duration=2800&pause=900&color=8B5CF6&center=true&vCenter=true&width=820&height=64&lines=AI+%2F+LLM+Engineer+%C2%B7+Fintech;I+build+LLM+systems+%C2%B7+AI+agents+%C2%B7+ML+platforms;Production+AI+that+passes+the+audit.)](https://nagizaaz.vercel.app)

</div>

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-nagizaazshaik-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/nagizaazshaik)
[![Portfolio](https://img.shields.io/badge/Portfolio-nagizaaz.vercel.app-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://nagizaaz.vercel.app)
[![Email](https://img.shields.io/badge/Email-nagizaazs-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:nagizaazs@gmail.com)

<br/>

![AWS ML Specialty](https://img.shields.io/badge/AWS-ML_Specialty-FF9900?style=flat&logo=amazonaws&logoColor=white)
![GCP PDE](https://img.shields.io/badge/GCP-Professional_Data_Engineer-4285F4?style=flat&logo=googlecloud&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat&logo=langchain&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazonaws&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat&logo=kubernetes&logoColor=white)

</div>

---

## About

AI / LLM Engineer building production AI for **financial services** — credit-risk models, multi-agent pipelines, and the secure ML platforms that ship them. 5+ years from data engineering to ML and cloud architecture, now focused on applied LLM systems. **AWS ML Specialty** certified.

---

## Architecture

<div align="center">
<img src="https://raw.githubusercontent.com/shaikn6/shaikn6/main/stack-diagram.svg" width="100%" alt="Production LLM Stack"/>
</div>

*End-to-end production stack — streaming data ingestion → feature store → LLM gateway → multi-agent layer → observability*

---

## Certifications

| Certification | Issuer |
|---|---|
| AWS Certified Machine Learning – Specialty | Amazon Web Services |
| AWS Certified Cloud Practitioner | Amazon Web Services |
| GCP Professional Data Engineer | Google Cloud |
| Snowflake SnowPro Advanced: Data Engineer | Snowflake |
| dbt Analytics Engineering Certification | dbt Labs |

---

## Skills

| Domain | Technologies |
|--------|-------------|
| **LLM Engineering** | LangChain · LangGraph · RAG · MCP · Prompt Engineering · OWASP LLM Top 10 · Semantic Caching · LLM Gateways |
| **AgentOps** | Multi-Agent Systems · LangGraph · CrewAI · Tool Use · Agent Memory · Autonomous Coding Agents · n8n |
| **MLOps / LLMOps** | MLflow · Feast · Airflow · Evidently AI · RAGAS · Prompt Versioning · A/B Testing · Model Registry |
| **DevSecOps** | GitHub Actions · ArgoCD · Crossplane · OPA Gatekeeper · Trivy · Semgrep · Cosign · SBOM · SLSA |
| **Data Engineering** | Apache Kafka · Apache Spark · dbt · Redis · Airflow · SQL lineage · ETL/ELT · PostgreSQL |
| **ML / Deep Learning** | PyTorch · XGBoost · scikit-learn · HuggingFace · Federated Learning · Differential Privacy · Fairlearn |
| **Cloud & Infra** | AWS (EKS · SageMaker · Redshift · S3 · IAM · CloudWatch) · GCP · Terraform · Docker · Kubernetes |
| **Security & Governance** | ECOA / Fair Lending · GDPR · PCI-DSS · SOC2 · Model Cards · Audit Trails · Data Lineage |

---

## Featured Projects

### LLM Engineering & Agents

| Repo | What it does | Stack |
|------|-------------|-------|
| [`agent-autopsy`](https://github.com/shaikn6/agent-autopsy) | Framework-agnostic agent interceptor — builds execution flame graphs, detects failure patterns (loops, token storms, goal drift), generates structured post-mortems | Python · networkx · FastAPI |
| [`llm-gateway`](https://github.com/shaikn6/llm-gateway) | Production LLM gateway: OpenAI-compatible API, semantic caching, multi-model routing, cost analytics dashboard | FastAPI · Redis · sentence-transformers |
| [`llm-safety-auditor`](https://github.com/shaikn6/llm-safety-auditor) [![Live Demo](https://img.shields.io/badge/demo-live-FF9E0F?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/9mark9/llm-safety-auditor) | 250+ adversarial attack payloads, OWASP LLM Top 10 scoring, red-team report generation, Streamlit dashboard | FastAPI · sentence-transformers · reportlab |
| [`nano-finbert`](https://github.com/shaikn6/nano-finbert) | Tiny transformer (~2M params) trained from scratch on financial text — outputs structured market signals: sentiment, entities, sectors, event type, impact score | PyTorch · FastAPI |

### Multi-Agent Systems

| Repo | What it does | Stack |
|------|-------------|-------|
| [`autonomous-coding-agent`](https://github.com/shaikn6/autonomous-coding-agent) | Reads GitHub issues → understands codebase → generates fix → runs tests → opens PR | Claude · LangGraph · PyGitHub · FastAPI |
| [`mcp-diagram-agent`](https://github.com/shaikn6/mcp-diagram-agent) | MCP server: describe any system → production-ready Excalidraw architecture diagram via Claude. Fully typed tool surface, 97%+ coverage | MCP · Claude · FastAPI · mypy |
| [`finance-agent-crew`](https://github.com/shaikn6/finance-agent-crew) | Multi-agent financial intelligence: SEC EDGAR + earnings calls + news → investment brief | LangGraph · Claude · n8n · FastAPI |
| [`ops-autopilot`](https://github.com/shaikn6/ops-autopilot) | Autonomous SRE agent: receives alerts → LLM root-cause diagnosis → creates GitHub fix PR + Slack summary + runbook entry | FastAPI · Anthropic · PyGithub |
| [`multi-agent-research-crew`](https://github.com/shaikn6/multi-agent-research-crew) | LangGraph crew (researcher + analyst + writer + critic) producing structured research reports with SSE streaming progress | LangGraph · Claude · FastAPI |

### MLOps & Production Infrastructure

| Repo | What it does | Stack |
|------|-------------|-------|
| [`llmops-eval-platform`](https://github.com/shaikn6/llmops-eval-platform) | RAGAS RAG evaluation + Claude-as-judge + A/B testing + prompt versioning + cost tracking | Python · RAGAS · FastAPI · SQLAlchemy |
| [`nvidia-nim-rag-techniques`](https://github.com/shaikn6/nvidia-nim-rag-techniques) | 5 production RAG techniques on NVIDIA NIM: hybrid search/RRF, cross-encoder reranking, query rewriting (HyDE/multi-query/step-back), context compression, corrective RAG | NVIDIA NIM · LangGraph · FastAPI |
| [`mlops-feature-platform`](https://github.com/shaikn6/mlops-feature-platform) | Enterprise MLOps for financial ML: Feast feature store + MLflow registry + Airflow pipelines + Evidently AI monitoring | Feast · MLflow · Airflow · Evidently |
| [`kafka-stream-feature-store`](https://github.com/shaikn6/kafka-stream-feature-store) | Real-time ML feature store: Kafka → Redis pipeline with sub-60s feature freshness for low-latency inference | Kafka · Redis · FastAPI |
| [`financial-model-governance`](https://github.com/shaikn6/financial-model-governance) | ML governance for finance: ECOA fairness testing, disparate-impact analysis, model cards, audit trails, approval workflows | Python · ECOA · Model Cards |

### DevSecOps & Applied RAG

| Repo | What it does | Stack |
|------|-------------|-------|
| [`fintech-devsecops-pipeline`](https://github.com/shaikn6/fintech-devsecops-pipeline) | End-to-end DevSecOps: SBOM generation, Trivy scanning, OPA policy enforcement, Cosign image signing, SLSA Level 3 provenance | GitHub Actions · Trivy · Cosign · OPA · ArgoCD |
| [`k8s-platform-engineering`](https://github.com/shaikn6/k8s-platform-engineering) | Internal Developer Platform: ArgoCD GitOps + Crossplane infra provisioning + OPA Gatekeeper policies + golden-path templates | Kubernetes · ArgoCD · Crossplane · OPA |
| [`adaptive-cognitive-rag`](https://github.com/shaikn6/adaptive-cognitive-rag) | RAG that measures cognitive state from behavioral signals and adapts explanation depth; builds a per-session knowledge graph of mastery gaps | RAG · Knowledge Graph · LangChain · FastAPI |

---

## Domain Coverage

```
LLM Engineering    ████████████████████  15 repos
Agents & Agentic   ███████████████████░  12 repos
MLOps / LLMOps     ██████████████████░░  10 repos
Fintech AI         ████████████████░░░░   8 repos
DevSecOps          ██████████████░░░░░░   6 repos
Platform Eng       ████████████░░░░░░░░   5 repos
Computer Vision    ██████████░░░░░░░░░░   4 repos
```

---

## Other Public Repos

Full portfolio spans computer vision, AIOps, n8n automation, Kubernetes platform engineering, RAG assistants — all with CI/CD, Docker, and automated testing.

**Private repos available on request** — federated learning for credit risk, LLM red-teaming, on-device LLM optimization, SQL-to-DAG compilers. Contact via [LinkedIn](https://linkedin.com/in/nagizaazshaik).

---

## GitHub Stats

<div align="center">

<a href="https://github.com/shaikn6">
  <img height="180em" src="https://github-readme-stats.vercel.app/api?username=shaikn6&show_icons=true&theme=github_dark&hide_border=true&count_private=true&include_all_commits=true&hide_rank=true&hide=stars,issues&bg_color=0d1117&title_color=6366f1&icon_color=6366f1&text_color=94a3b8&ring_color=6366f1" />
  <img height="180em" src="https://github-readme-stats.vercel.app/api/top-langs/?username=shaikn6&theme=github_dark&hide_border=true&layout=compact&bg_color=0d1117&title_color=6366f1&text_color=94a3b8&langs_count=8" />
</a>

</div>

<div align="center">

<img src="https://streak-stats.demolab.com?user=shaikn6&theme=github-dark-blue&hide_border=true&date_format=M%20j%5B%2C%20Y%5D&mode=weekly&background=0d1117&ring=6366f1&fire=a855f7&currStreakLabel=6366f1" />

</div>

<div align="center">

<img src="https://github-readme-activity-graph.vercel.app/graph?username=shaikn6&theme=github-compact&hide_border=true&area=true&bg_color=0d1117&color=6366f1&line=6366f1&point=a855f7&custom_title=Contribution+Activity" />

</div>

---

<div align="center">
<sub>75 public repos · CI/CD green across all repos · Docker · AWS · <a href="https://nagizaaz.vercel.app">nagizaaz.vercel.app</a> · <a href="https://linkedin.com/in/nagizaazshaik">LinkedIn</a></sub>
</div>
