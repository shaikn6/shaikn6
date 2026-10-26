<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/header-dark.svg">
  <img src="assets/header-light.svg" width="100%" alt="Nagizaaz Shaik, Generative AI Engineer: Agentic AI, RAG, AWS, Fintech"/>
</picture>

<div align="center">

[![Portfolio](https://img.shields.io/badge/Portfolio-141414?style=flat-square&logo=vercel&logoColor=white)](https://nagizaaz.vercel.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nagizaazshaik)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:nagizaazs@gmail.com)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-FFD21E?style=flat-square&logoColor=black)](https://huggingface.co/9mark9)

</div>

I build **generative AI systems for fintech**: RAG pipelines, LLM gateways, agentic workflows, and the AWS ML
platforms underneath them. 5 years in data and ML engineering, the last 2 in production at regulated credit
unions. **AWS ML Specialty** certified.

**Now:** doctoral research (DBA, Applied AI) on AI-agent security. I traced a sandbox bypass into 25 open-source
projects (23 still unpatched) and earned reporter credit on
[GHSA-m5v3-6ccr-cfgx](https://github.com/femto/minion/security/advisories/GHSA-m5v3-6ccr-cfgx). Open to GenAI and
agentic AI engineering roles, internship or full-time.

### Featured

| | |
|---|---|
| **[llm-gateway](https://github.com/shaikn6/llm-gateway)** `FastAPI` | One OpenAI-style chat endpoint over Anthropic / OpenAI / Ollama: per-tenant response cache, rate limiting, A/B routing, cost tracking, audit logging. 415 tests, 99% coverage, SDK contract tests. |
| **[lethe](https://github.com/shaikn6/lethe)** `Go` | Memory server for AI agents that learns from feedback which memories to return. The first learner lost to a hand-set rule (0.611 vs 0.641 evidence in the top 10); forced exploration fixed it: 0.707 on held-out LoCoMo conversations, at 7,119 requests/s. Python, LangChain and MCP adapters. |
| **[llm-safety-auditor](https://github.com/shaikn6/llm-safety-auditor)** `FastAPI` · [demo](https://huggingface.co/spaces/9mark9/llm-safety-auditor) | LLM red-teaming: 250+ adversarial payloads, a 6-layer detector, OWASP LLM Top 10 scoring, report generation. 479 tests. |
| **[fin-lora](https://github.com/shaikn6/fin-lora)** `PyTorch` · [adapters](https://huggingface.co/9mark9/fin-lora-news-qwen2.5-0.5b) | LoRA-tuned Qwen2.5-0.5B for financial sentiment: 84.7% on news vs 73.6% for FinBERT; a confidence cascade cuts average latency 63% for 0.4 points of accuracy; a follow-up study measures how easily it is fooled. |
| **[finance-agent-crew](https://github.com/shaikn6/finance-agent-crew)** `asyncio` | Multi-agent research: concurrent SEC EDGAR, market and news gathering feeds parallel fundamental, sentiment and risk agents; a writer agent produces a sourced brief. 333 tests. |
| **[guard-gap-detector](https://github.com/shaikn6/guard-gap-detector)** `Python` | Static detector for a sandbox-guard gap in AST-walking restricted Python interpreters used by AI agents; artifact for the paper *When Vendoring Defeats the Patch*. |

<details>
<summary><b>More LLM work</b></summary>
<br/>

| | |
|---|---|
| **[mcp-diagram-agent](https://github.com/shaikn6/mcp-diagram-agent)** | MCP server that turns a plain-text system description into an Excalidraw architecture diagram. |
| **[nvidia-nim-rag-techniques](https://github.com/shaikn6/nvidia-nim-rag-techniques)** | Five RAG techniques on NVIDIA NIM: hybrid search, reranking, query rewriting, context compression, corrective RAG. |
| **[on-device-llm-optimizer](https://github.com/shaikn6/on-device-llm-optimizer)** | Distillation, INT4 quantization and CoreML export for running a small LLM on Apple hardware. |
| **[nano-finbert](https://github.com/shaikn6/nano-finbert)** | A 1.88M-parameter transformer built from scratch, with a measured study of how far distillation and pretraining can take it. |

</details>

### Open source

**Merged:** [crewAI #8061](https://github.com/crewAIInc/crewAI/pull/8061) (streaming event fix, from a bug I reported) ·
[GENesis-AGI #2960](https://github.com/WingedGuardian/GENesis-AGI/pull/2960) (path-containment fix) ·
[prefect #23367](https://github.com/PrefectHQ/prefect/pull/23367) (ECS guide import) ·
[crewAI #8033](https://github.com/crewAIInc/crewAI/pull/8033) ·
[temporal sdk-python #1955](https://github.com/temporalio/sdk-python/pull/1955) ·
[ray #66918](https://github.com/ray-project/ray/pull/66918) (RLlib / Tune doc samples) ·
[litestar #5126](https://github.com/litestar-org/litestar/pull/5126) ·
[goose #1121](https://github.com/pressly/goose/pull/1121) ·
[jackc/pgx #2638](https://github.com/jackc/pgx/pull/2638) ·
[mlflow #26613](https://github.com/mlflow/mlflow/pull/26613) ·
[langroid #1209](https://github.com/langroid/langroid/pull/1209) ·
[pyproj #1637](https://github.com/pyproj4/pyproj/pull/1637) ·
[nibabel #1550](https://github.com/nipy/nibabel/pull/1550) ·
[pystac #1803](https://github.com/stac-utils/pystac/pull/1803) ·
[pystac-client #931](https://github.com/stac-utils/pystac-client/pull/931) ·
[fhir.resources #211](https://github.com/nazrulworld/fhir.resources/pull/211) ·
[backtest-integrity-guard #33](https://github.com/suguobin2021/backtest-integrity-guard/pull/33) ·
[hwspec #75](https://github.com/jiegui2025/hwspec/pull/75)<br/>
**Security:** reporter credit on [GHSA-m5v3-6ccr-cfgx](https://github.com/femto/minion/security/advisories/GHSA-m5v3-6ccr-cfgx), a sandbox bypass in an AI-agent code executor; further reports are under coordinated disclosure.<br/>
**In review:** native Groq / Fireworks / Together providers for [crewAI](https://github.com/crewAIInc/crewAI/pulls?q=author%3Ashaikn6), and fixes to broken documentation samples in Ray, MLflow, Dagster, Hugging Face Datasets, Feast, LlamaIndex and others ([all open pull requests](https://github.com/search?q=author%3Ashaikn6+is%3Apr+is%3Aopen+-user%3Ashaikn6&type=pullrequests)).

### Stack

**LLMs** Claude · OpenAI · Qwen2.5 · LoRA / PEFT · vLLM · Ollama &nbsp;•&nbsp; **Agents** LangGraph · CrewAI · MCP · tool calling · agent memory<br/>
**RAG** LangChain · Pinecone · FAISS · hybrid search · reranking &nbsp;•&nbsp; **Evals** RAGAS · LLM-as-judge · OWASP LLM Top 10<br/>
**AWS** SageMaker · Bedrock · Lambda · S3 · Glue · Redshift · EKS · IAM &nbsp;•&nbsp; **Platform** Databricks · MLflow · Airflow · Kafka · Terraform · Docker · Kubernetes · ArgoCD · GitHub Actions &nbsp;•&nbsp; **Languages** Python · Go · SQL

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

<sub><!--REPO_COUNT_START-->22 public repos<!--REPO_COUNT_END--> · <a href="https://nagizaaz.vercel.app">Portfolio</a> · <a href="https://www.linkedin.com/in/nagizaazshaik">LinkedIn</a> · <a href="https://huggingface.co/9mark9">Hugging Face</a></sub>

</div>
