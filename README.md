<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/header-dark.svg">
  <img src="assets/header-light.svg" width="100%" alt="Nagizaaz Shaik, AI / LLM Systems Engineer"/>
</picture>

<div align="center">

[![Portfolio](https://img.shields.io/badge/Portfolio-141414?style=flat-square&logo=vercel&logoColor=white)](https://nagizaaz.vercel.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/nagizaaz-shaik)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:nagizaazs@gmail.com)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-FFD21E?style=flat-square&logoColor=black)](https://huggingface.co/9mark9)

</div>

I build **production GenAI systems for fintech**: LLM gateways, RAG pipelines and multi-agent
orchestration, with the backend rigor these workloads need (idempotency, ACID, audit logging,
and CI that gates). 5+ years across data engineering, ML and cloud; 2+ shipping LLM systems.
**AWS ML Specialty** certified.

**Now:** doctoral research in Applied AI on two questions: how easily small financial language
models can be fooled, and how AI-agent code sandboxes get bypassed. Open to full-time roles and
research internships.

### Featured

| | |
|---|---|
| **[ledger-service](https://github.com/shaikn6/ledger-service)** `Go` | Double-entry ledger microservice: idempotent transfers, reversals, deadlock-free row locking, append-only postings. 31 tests, `-race`, `govulncheck` clean. |
| **[llm-gateway](https://github.com/shaikn6/llm-gateway)** `FastAPI` | One OpenAI-style chat endpoint over Anthropic / OpenAI / Ollama: per-tenant response cache, rate limiting, A/B routing, cost tracking, audit logging. 415 tests, 99% coverage, SDK contract tests. |
| **[fintech-devsecops-pipeline](https://github.com/shaikn6/fintech-devsecops-pipeline)** `Terraform` | DevSecOps reference platform: hardened Terraform + private EKS, OPA/Rego admission policies with 49 policy tests, Helm chart gated by conftest in CI, ArgoCD GitOps. |
| **[fin-lora](https://github.com/shaikn6/fin-lora)** `PyTorch` · [adapters](https://huggingface.co/9mark9/fin-lora-news-qwen2.5-0.5b) | LoRA-tuned Qwen2.5-0.5B for financial sentiment against TF-IDF, FinBERT and zero-shot on tweets, headlines and news. A confidence cascade cuts average latency 63% for 0.4 points of accuracy; a follow-up study measures how easily each model is fooled across five markets. |
| **[llm-safety-auditor](https://github.com/shaikn6/llm-safety-auditor)** `FastAPI` · [demo](https://huggingface.co/spaces/9mark9/llm-safety-auditor) | LLM red-teaming: 250+ adversarial payloads, a 6-layer detector, OWASP LLM Top 10 scoring, report generation. 479 tests. |
| **[credit-arena](https://github.com/shaikn6/credit-arena)** `scikit-learn` | Credit-default risk: 6 model families on 30K accounts, compared on AUC, calibration, business cost and fair-lending behaviour with bootstrap significance tests. |

<details>
<summary><b>More projects</b></summary>
<br/>

| | |
|---|---|
| **[georisk-agent](https://github.com/shaikn6/georisk-agent)** · [live demo](https://georisk-agent.vercel.app) | Flood and wildfire risk for a US site from Sentinel-2 imagery and terrain, with calibrated probabilities validated against FEMA and USGS records (flood AUC 0.866 on held-out regions). |
| **[nano-finbert](https://github.com/shaikn6/nano-finbert)** | A 1.88M-parameter transformer built from scratch, plus a measured study of how far distillation and pretraining can take it (55% → 65% on a fixed test split, and why it stops there). |
| **[finance-agent-crew](https://github.com/shaikn6/finance-agent-crew)** | Turns a stock ticker into a sourced research brief: concurrent data gathering from SEC EDGAR and market APIs, then three analyst agents. Runs offline in demo mode. |
| **[finarena](https://github.com/shaikn6/finarena)** | Model-serving API with a web UI: cost-routed sentiment and credit scoring behind authentication, with measured capacity numbers. |
| **[trade-arena](https://github.com/shaikn6/trade-arena)** | Walk-forward backtest of rules and ML models on 10 assets, net of costs, with a shuffled-label control. Result: nothing beat buy-and-hold. |
| **[exec-rl](https://github.com/shaikn6/exec-rl)** | Optimal trade execution: a from-scratch PPO agent against TWAP and Almgren-Chriss on a simulated market. |
| **[sigdet](https://github.com/shaikn6/sigdet)** | Document signature detection with YOLO11, with a robustness study under blur, JPEG compression, noise and low light. |
| **[event-impact-signals](https://github.com/shaikn6/event-impact-signals)** | Classifies real-world events from live news and maps them to the sectors likely to gain or lose, with the reasoning shown. |
| **[mcp-diagram-agent](https://github.com/shaikn6/mcp-diagram-agent)** | MCP server that turns a plain-text system description into an Excalidraw architecture diagram. |
| **[nvidia-nim-rag-techniques](https://github.com/shaikn6/nvidia-nim-rag-techniques)** | Five RAG optimization techniques on NVIDIA NIM: hybrid search, reranking, query rewriting, context compression, corrective RAG. |
| **[on-device-llm-optimizer](https://github.com/shaikn6/on-device-llm-optimizer)** | Distillation, INT4 quantization and CoreML export pipeline for running a small LLM on Apple hardware. |

</details>

### Open source

**Merged:** [jackc/pgx #2638](https://github.com/jackc/pgx/pull/2638) ·
[mlflow #26613](https://github.com/mlflow/mlflow/pull/26613) ·
[langroid #1209](https://github.com/langroid/langroid/pull/1209) ·
[pyproj #1637](https://github.com/pyproj4/pyproj/pull/1637) ·
[nibabel #1550](https://github.com/nipy/nibabel/pull/1550) ·
[pystac #1803](https://github.com/stac-utils/pystac/pull/1803) ·
[pystac-client #931](https://github.com/stac-utils/pystac-client/pull/931) ·
[fhir.resources #211](https://github.com/nazrulworld/fhir.resources/pull/211)<br/>
**Security:** reporter credit on [GHSA-m5v3-6ccr-cfgx](https://github.com/femto/minion/security/advisories/GHSA-m5v3-6ccr-cfgx), a sandbox bypass in an AI-agent code executor; further reports are under coordinated disclosure.<br/>
**In review:** native Groq / Fireworks / Together providers for [crewAI](https://github.com/crewAIInc/crewAI/pulls?q=author%3Ashaikn6), and fixes to broken documentation samples in Ray, MLflow, Dagster, Hugging Face Datasets, Feast, LlamaIndex and others ([all open pull requests](https://github.com/search?q=author%3Ashaikn6+is%3Apr+is%3Aopen+-user%3Ashaikn6&type=pullrequests)).

### Stack

**Backend** Go · Python · FastAPI · PostgreSQL · Redis &nbsp;•&nbsp; **LLM / Agents** LangGraph · RAG · MCP · OWASP LLM Top 10 · LLMOps<br/>
**Infra** Docker · Kubernetes · Terraform · ArgoCD · GitHub Actions · AWS &nbsp;•&nbsp; **ML / Data** PyTorch · MLflow · Kafka · dbt

<details>
<summary><b>Activity</b></summary>
<br/>
<div align="center">

<img src="https://raw.githubusercontent.com/shaikn6/shaikn6/output/activity-graph.svg" alt="Contribution activity over the last year" width="100%"/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/shaikn6/shaikn6/output/github-snake-dark.svg">
  <img src="https://raw.githubusercontent.com/shaikn6/shaikn6/output/github-snake.svg" alt="Contribution snake" width="100%"/>
</picture>

</div>
</details>

<div align="center">

<sub><!--REPO_COUNT_START-->19 public repos<!--REPO_COUNT_END--> · <a href="https://nagizaaz.vercel.app">Portfolio</a> · <a href="https://linkedin.com/in/nagizaaz-shaik">LinkedIn</a> · <a href="https://huggingface.co/9mark9">Hugging Face</a></sub>

</div>
</content>
</invoke>
