<img src="https://capsule-render.vercel.app/api?type=waving&color=0:c8b08a,100:b59b73&height=210&section=header&text=Nagizaaz%20Shaik&fontColor=0a0a0b&fontSize=58&fontAlignY=36&desc=AI%20%2F%20LLM%20Systems%20Engineer%20%C2%B7%20Production%20GenAI%20for%20Fintech&descSize=18&descAlignY=58&descColor=141414" width="100%" alt="Nagizaaz Shaik"/>

<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com/?font=Fraunces&size=26&duration=2800&pause=900&color=C8B08A&center=true&vCenter=true&width=840&height=52&lines=AI+%2F+LLM+Systems+Engineer+%C2%B7+production+GenAI+for+fintech;RAG+%C2%B7+multi-agent+orchestration+%C2%B7+LLM+gateways+%C2%B7+LLMOps;Go+%2B+Python+%C2%B7+Postgres+%C2%B7+Kubernetes+%C2%B7+built+to+pass+the+audit)](https://nagizaaz.vercel.app)

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/nagizaazshaik)
[![Portfolio](https://img.shields.io/badge/Portfolio-141414?style=for-the-badge&logo=vercel&logoColor=white)](https://nagizaaz.vercel.app)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:nagizaazs@gmail.com)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/9mark9)

</div>

---

## About

I build **production GenAI systems for fintech** — LLM gateways, RAG pipelines, multi-agent
orchestration, and the LLMOps and DevSecOps platforms that ship them into regulated
environments where correctness and compliance are not optional.

The through-line in my work is **backend rigor applied to AI**: idempotency and ACID
guarantees for money movement, constant-time auth, structured audit logging, ordered
locking for concurrency safety, OWASP LLM Top-10 hardening, and CI that actually gates
(`-race`, `govulncheck`, container smoke tests). 5+ years across data engineering, ML, and
cloud architecture; 2+ years shipping production LLM systems. Currently a **doctoral
researcher in Applied AI** and open to LLM / AI systems roles.

<div align="center">

![AWS ML Specialty](https://img.shields.io/badge/AWS-ML_Specialty-FF9900?style=flat&logo=amazonaws&logoColor=white)
![GCP PDE](https://img.shields.io/badge/GCP-Professional_Data_Engineer-4285F4?style=flat&logo=googlecloud&logoColor=white)
![SnowPro](https://img.shields.io/badge/Snowflake-SnowPro_Advanced-29B5E8?style=flat&logo=snowflake&logoColor=white)
![dbt](https://img.shields.io/badge/dbt-Analytics_Engineering-FF694B?style=flat&logo=dbt&logoColor=white)

</div>

---

## Featured Work

### Production Services

| Project | What it is |
|---------|-----------|
| **[ledger-service](https://github.com/shaikn6/ledger-service)** &nbsp;`Go · Postgres` | Double-entry accounting ledger microservice. Idempotent transfers keyed on `Idempotency-Key`, one-shot reversals, ordered `SELECT … FOR UPDATE` locking (deadlock-free) + retry on serialization failures, append-only postings, integer minor units. OpenAPI 3.1, optional bearer auth, Prometheus metrics, distroless image. **31 tests** incl. a 40-goroutine balance-conservation test — all green with `-race`; `govulncheck` clean. |
| **[llm-gateway](https://github.com/shaikn6/llm-gateway)** &nbsp;`FastAPI · Redis` | One OpenAI-compatible endpoint in front of Anthropic / OpenAI / Ollama. Redis semantic cache, sliding-window rate limiting, deterministic A/B routing, per-key cost accounting, API-key auth, and metadata-only JSON audit logging. **~330 tests, 99% line coverage.** |
| **[fintech-devsecops-pipeline](https://github.com/shaikn6/fintech-devsecops-pipeline)** &nbsp;`Terraform · EKS · ArgoCD` | Shift-left DevSecOps reference platform: hardened Terraform (VPC + private EKS with IRSA), an OPA/Rego admission-policy set, a multi-scanner CI security gate, and ArgoCD GitOps delivery — with a documented Disaster Recovery design (RTO/RPO by scenario, failover strategy, runbook). |

### LLM Engineering & Agents

- **[nvidia-nim-rag-techniques](https://github.com/shaikn6/nvidia-nim-rag-techniques)** — 5 production RAG optimizations on NVIDIA NIM: hybrid search / RRF, cross-encoder reranking, query rewriting (HyDE / multi-query / step-back), context compression, corrective RAG (LangGraph). &nbsp;`NVIDIA NIM · LangGraph`
- **[mcp-diagram-agent](https://github.com/shaikn6/mcp-diagram-agent)** — MCP server: describe a system in plain English → production-ready architecture diagram via Claude. &nbsp;`MCP · LangGraph · FastAPI`
- **[llm-safety-auditor](https://github.com/shaikn6/llm-safety-auditor)** &nbsp;([live demo](https://huggingface.co/spaces/9mark9/llm-safety-auditor)) — 250+ adversarial payloads, OWASP LLM Top-10 scoring, red-team report generation. &nbsp;`FastAPI · sentence-transformers`
- **[finance-agent-crew](https://github.com/shaikn6/finance-agent-crew)** — Multi-agent financial intelligence: async analyst agents (fundamentals, sentiment, risk, SEC filings, news) synthesized into an investment brief. &nbsp;`LangGraph · Claude`

### Applied ML

- **[on-device-llm-optimizer](https://github.com/shaikn6/on-device-llm-optimizer)** — Knowledge distillation: Phi-3 Mini (3.8B) → 236M student on Apple MLX, INT4 quantization, CoreML export. &nbsp;`MLX · Distillation · CoreML`
- **[nano-finbert](https://github.com/shaikn6/nano-finbert)** — Transformer built from scratch on financial text, plus a production [MiniLM sentiment model](https://huggingface.co/9mark9/finbert-minilm-sentiment) at 95.3% test accuracy. &nbsp;`PyTorch · HuggingFace`

### Regulated-Domain Rigor

- **[phi-shield](https://github.com/shaikn6/phi-shield)** — One-line HIPAA Safe-Harbor PHI de-identification for clinical text, benchmarked on recall *and* clinical-utility preservation — the same PII-handling and compliance discipline that fintech workloads demand. &nbsp;`Python · NER · benchmarks`

---

## Tech Stack

| Domain | Technologies |
|--------|-------------|
| **LLM Engineering** | LangChain · LangGraph · RAG · MCP · Prompt Engineering · OWASP LLM Top 10 · Semantic Caching · LLM Gateways · Model Distillation |
| **Agents / AgentOps** | Multi-Agent Systems · LangGraph · CrewAI · Tool Use · Agent Memory · Autonomous Coding Agents · n8n |
| **Backend** | Go · Python · FastAPI · PostgreSQL · Redis · gRPC/REST · OpenAPI · Idempotency · Concurrency Control |
| **MLOps / LLMOps** | MLflow · Feast · Airflow · Evidently AI · RAGAS · Prompt Versioning · A/B Testing · Model Registry |
| **DevSecOps / Cloud** | Docker · Kubernetes · GitHub Actions · Terraform · ArgoCD · OPA/Rego · Trivy · govulncheck · SBOM · SLSA · AWS (EKS · SageMaker · S3 · IAM) · GCP |
| **Data Engineering** | Apache Kafka · Apache Spark · dbt · Airflow · SQL lineage · ETL/ELT |
| **ML / Deep Learning** | PyTorch · XGBoost · scikit-learn · HuggingFace · Federated Learning · Differential Privacy · Fairlearn |
| **Security & Governance** | ECOA / Fair Lending · PCI-DSS · SOC 2 · GDPR · HIPAA Safe Harbor · Model Cards · Audit Trails · Data Lineage |

---

## Activity

<div align="center">

<img src="https://raw.githubusercontent.com/shaikn6/shaikn6/output/activity-graph.svg" alt="Contribution activity graph" width="100%"/>

<br/><br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/shaikn6/shaikn6/output/github-snake-dark.svg">
  <img src="https://raw.githubusercontent.com/shaikn6/shaikn6/output/github-snake.svg" alt="Contribution snake animation" width="100%"/>
</picture>

</div>

---

## Production Architecture

<div align="center">
<img src="https://raw.githubusercontent.com/shaikn6/shaikn6/main/stack-diagram.svg" width="100%" alt="Production LLM stack — streaming ingestion → feature store → LLM gateway → multi-agent layer → observability"/>
</div>

---

## Certifications

| Certification | Issuer |
|---|---|
| AWS Certified Machine Learning – Specialty | Amazon Web Services |
| AWS Certified Cloud Practitioner | Amazon Web Services |
| GCP Professional Data Engineer | Google Cloud |
| Snowflake SnowPro Advanced: Data Engineer | Snowflake |
| dbt Analytics Engineering Certification | dbt Labs |

<br/>

> Additional work under private repos — federated learning for credit risk, LLMOps evaluation platforms, SQL-to-DAG compilers, ML fairness auditing. Ask via [LinkedIn](https://linkedin.com/in/nagizaazshaik).

<div align="center">
<sub><!--REPO_COUNT_START-->16 public repos<!--REPO_COUNT_END--> · CI green · Docker · Kubernetes · AWS · <a href="https://nagizaaz.vercel.app">Portfolio</a> · <a href="https://linkedin.com/in/nagizaazshaik">LinkedIn</a> · <a href="https://huggingface.co/9mark9">Hugging Face</a></sub>
</div>
