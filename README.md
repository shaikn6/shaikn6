<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/header-dark.svg">
  <img src="assets/header-light.svg" width="100%" alt="Nagizaaz Shaik, generative and agentic AI engineer"/>
</picture>

<div align="center">

[![Portfolio](https://img.shields.io/badge/Portfolio-141414?style=flat-square&logo=vercel&logoColor=white)](https://nagizaaz.vercel.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nagizaazshaik)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:nagizaazs@gmail.com)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-FFD21E?style=flat-square&logoColor=black)](https://huggingface.co/9mark9)

</div>

**Generative and agentic AI engineer.** I started in data engineering, moved to ML platform engineering, and now
build LLM systems: agent memory, LLM gateways, fine-tuned models and the evaluations that check them. Each project
number below comes from the linked repository. **AWS ML Specialty** certified.

- **20 pull requests merged upstream**, in crewAI, Ray, MLflow, DSPy and 14 other open-source projects ([list below](#open-source)).
- **Agent memory that beats a hand-tuned rule on held-out data:** [lethe](https://github.com/shaikn6/lethe) puts 0.707 of the evidence in the top 10 on LoCoMo vs 0.641 for the best hand-set rule, and only ties it on LongMemEval.
- **Security advisory with credit:** I reported a sandbox bypass in an AI-agent code executor through a GitHub security advisory; femto/minion adopted the fix and credited the report ([GHSA-m5v3-6ccr-cfgx](https://github.com/femto/minion/security/advisories/GHSA-m5v3-6ccr-cfgx)).

**Now:** doctoral research (DBA, Applied AI) on AI-agent security and LLM robustness. Open to LLM, GenAI and agentic
AI engineering roles, internship or full-time.

### Start here

| | |
|---|---|
| **[lethe](https://github.com/shaikn6/lethe)** `Go` · [demo](https://huggingface.co/spaces/9mark9/lethe-agent-memory) | **0.707 of the evidence in the top 10 on held-out LoCoMo vs 0.641 for the best hand-set rule; on LongMemEval it only ties that rule.** A memory server for AI agents that learns from feedback which memories to return, with Python, LangChain and MCP adapters. |
| **[llm-gateway](https://github.com/shaikn6/llm-gateway)** `Python` `FastAPI` `Redis` | **One OpenAI-compatible endpoint over Anthropic, OpenAI and Ollama; 428 tests at 99% line coverage, CI fails under 95%.** Response caching, sliding-window rate limits, deterministic A/B routing and per-key cost tracking; contract tests bind each provider request to the real SDK signatures. |
| **[fin-lora](https://github.com/shaikn6/fin-lora)** `PyTorch` `PEFT` · [demo](https://huggingface.co/spaces/9mark9/fin-lora-attack-demo) | **84.7% on LLM-labelled news vs 73.6% for zero-shot FinBERT (76.8% vs 71.2% on 267 human-labelled headlines).** LoRA-tuned Qwen2.5-0.5B for financial sentiment; a confidence cascade cuts average latency 63% for 0.4 points of accuracy, and a follow-up study measures how easily planted labels flip it. |

### More work

| | |
|---|---|
| **[llm-safety-auditor](https://github.com/shaikn6/llm-safety-auditor)** · [demo](https://huggingface.co/spaces/9mark9/llm-safety-auditor) | **Scores a model against the OWASP LLM Top 10; 635 tests.** 50 curated attacks (350 with mutations), a 6-layer detector and PDF reports. Detector recall is measured on a seeded mock, and the README shows where it is weaker on real text. |
| **[readme-truth](https://github.com/shaikn6/readme-truth)** · engineering tools | **Catches READMEs that no longer match the code; a hand audit of 22 repositories found 15 had drifted.** Turns a README into checkable claims (test counts, versions, paths, commands, links) and checks each against the repository. Offline CLI or GitHub Action; 52 tests. |
| **[finance-agent-crew](https://github.com/shaikn6/finance-agent-crew)** | **Turns a stock ticker into a sourced research brief; runs offline with no API keys; 333 tests.** Concurrent SEC EDGAR, market and news gathering feeds parallel fundamental, sentiment and risk agents, then a writer agent. |
| **[nvidia-nim-rag-techniques](https://github.com/shaikn6/nvidia-nim-rag-techniques)** | **Cross-encoder reranking lifted recall@1 from 0.857 to 0.964 over naive dense retrieval** on a small 28-query set (too small to rank techniques). Five RAG techniques on NVIDIA NIM with LangChain and LangGraph: hybrid search, reranking, query rewriting, context compression, corrective RAG. |
| **[sigdet](https://github.com/shaikn6/sigdet)** · computer vision | **0.939 ± 0.005 test mAP50-95 over 3 seeds for YOLO11n** on a leakage-aware split (26 test images); the larger model scored lower and varied more. Document signature detection with a robustness study under scan corruptions. |
| **[exec-rl](https://github.com/shaikn6/exec-rl)** · reinforcement learning | **0.22 bps mean shortfall vs 18.8 bps for TWAP** on 20,000 held-out simulated paths, 3 seeds, with an alpha signal; without the signal it is no better than TWAP. From-scratch PPO for trade execution. Simulation only. |
| **[guard-gap-detector](https://github.com/shaikn6/guard-gap-detector)** | **Separates all 3 vulnerable fixtures from all 5 clean ones on its synthetic corpus.** Read-only static detector for a sandbox-guard gap in AST-walking restricted Python interpreters used by AI agents; artifact for the working paper *When Vendoring Defeats the Patch*. |

<details>
<summary><b>Earlier and side projects</b></summary>
<br/>

| | |
|---|---|
| **[event-impact-signals](https://github.com/shaikn6/event-impact-signals)** | 94.4% (102 of 108) on held-out headlines with a sentence-embedding classifier vs 44.4% for keyword rules. The headlines are synthetic and written by one author, so this is a sanity check, not a benchmark. |
| **[finarena](https://github.com/shaikn6/finarena)** | One API in front of several fintech models: a cost-routed sentiment cascade (TF-IDF, then the LoRA LLM only when needed) and explainable credit scoring with reason codes. |
| **[mcp-diagram-agent](https://github.com/shaikn6/mcp-diagram-agent)** | MCP server that turns a plain-text system description into an Excalidraw architecture diagram. |
| **[on-device-llm-optimizer](https://github.com/shaikn6/on-device-llm-optimizer)** | Distillation, INT4 quantization and CoreML export pipeline for running a small LLM on Apple hardware. |
| **[nano-finbert](https://github.com/shaikn6/nano-finbert)** | A 1.88M-parameter transformer built from scratch, with a measured study of how far distillation and pretraining can take it. |
| **[credit-arena](https://github.com/shaikn6/credit-arena)** · **[trade-arena](https://github.com/shaikn6/trade-arena)** | Honest model comparisons for credit default and trading, with bootstrap confidence intervals; in trading, nothing beat buy-and-hold. |
| **[ledger-service](https://github.com/shaikn6/ledger-service)** | Double-entry ledger service in Go and Postgres with idempotent transfers and a concurrency test that checks balances are conserved. |

</details>

### Experience highlights

From my data, ML platform and ML engineering roles (2020 to 2026), not from the repositories above:

- **Deployment failures down 85% in 60 days:** built an AWS ML platform with SageMaker, an MLflow registry, gated promotion and automated rollback.
- **Policy research cut from about 2 hours to under 20 minutes per question:** built a RAG system over 500+ policy documents.
- **Sub-200 ms p99:** fine-tuned and served a 14-class BERT intent classifier to route support requests.
- **2 regulatory audits, zero findings:** owned ML data security (encryption, least-privilege IAM, audit logging).
- **Query runtime down 60%:** led an Oracle-to-Redshift migration, rebuilding stored procedures as Airflow DAGs.

### Open source

**Merged upstream (20):** [crewAI #8061](https://github.com/crewAIInc/crewAI/pull/8061) (streaming event fix, from a bug I reported) ·
[GENesis-AGI #2960](https://github.com/WingedGuardian/GENesis-AGI/pull/2960) (path-containment fix) ·
[prefect #23367](https://github.com/PrefectHQ/prefect/pull/23367) (ECS guide import) ·
[crewAI #8033](https://github.com/crewAIInc/crewAI/pull/8033) ·
[temporal sdk-python #1955](https://github.com/temporalio/sdk-python/pull/1955) ·
[ray #66918](https://github.com/ray-project/ray/pull/66918) (RLlib / Tune doc samples) ·
[litestar #5126](https://github.com/litestar-org/litestar/pull/5126) ·
[goose #1121](https://github.com/pressly/goose/pull/1121) ·
[dspy #10608](https://github.com/stanfordnlp/dspy/pull/10608) ·
[jackc/pgx #2638](https://github.com/jackc/pgx/pull/2638) ·
[mlflow #26613](https://github.com/mlflow/mlflow/pull/26613) ·
[mlflow #26634](https://github.com/mlflow/mlflow/pull/26634) ·
[langroid #1209](https://github.com/langroid/langroid/pull/1209) ·
[pyproj #1637](https://github.com/pyproj4/pyproj/pull/1637) ·
[nibabel #1550](https://github.com/nipy/nibabel/pull/1550) ·
[pystac #1803](https://github.com/stac-utils/pystac/pull/1803) ·
[pystac-client #931](https://github.com/stac-utils/pystac-client/pull/931) ·
[fhir.resources #211](https://github.com/nazrulworld/fhir.resources/pull/211) ·
[backtest-integrity-guard #33](https://github.com/suguobin2021/backtest-integrity-guard/pull/33) ·
[hwspec #75](https://github.com/jiegui2025/hwspec/pull/75)<br/>
**Security:** reported a sandbox bypass in an AI-agent code executor through a GitHub security advisory; a downstream project, femto/minion, adopted the fix and credited the report ([GHSA-m5v3-6ccr-cfgx](https://github.com/femto/minion/security/advisories/GHSA-m5v3-6ccr-cfgx)).<br/>
**In review:** native Groq / Fireworks / Together providers for [crewAI](https://github.com/crewAIInc/crewAI/pulls?q=author%3Ashaikn6), and fixes to broken documentation samples in Dagster, Hugging Face Datasets, Feast, LlamaIndex, sqlc and others ([all open pull requests](https://github.com/search?q=author%3Ashaikn6+is%3Apr+is%3Aopen+-user%3Ashaikn6&type=pullrequests)).

### Skills, with the repo that shows each

**Agents and agent memory** [lethe](https://github.com/shaikn6/lethe) (learned memory ranking, MCP and LangChain adapters) · [finance-agent-crew](https://github.com/shaikn6/finance-agent-crew) (multi-agent fan-out / fan-in) · [mcp-diagram-agent](https://github.com/shaikn6/mcp-diagram-agent) (MCP server)<br/>
**LLM serving and routing** [llm-gateway](https://github.com/shaikn6/llm-gateway) (Anthropic, OpenAI, Ollama, caching, rate limits, A/B routing, cost tracking) · [finarena](https://github.com/shaikn6/finarena) (cost-routed model cascade)<br/>
**Fine-tuning and small models** [fin-lora](https://github.com/shaikn6/fin-lora) (LoRA / PEFT on Qwen2.5) · [on-device-llm-optimizer](https://github.com/shaikn6/on-device-llm-optimizer) (distillation, INT4, MLX) · [nano-finbert](https://github.com/shaikn6/nano-finbert) (transformer from scratch)<br/>
**RAG and retrieval** [nvidia-nim-rag-techniques](https://github.com/shaikn6/nvidia-nim-rag-techniques) (hybrid search, reranking, corrective RAG with LangGraph) · [lethe](https://github.com/shaikn6/lethe) (BM25 + embeddings, learned ranking)<br/>
**Evaluation and LLM security** [llm-safety-auditor](https://github.com/shaikn6/llm-safety-auditor) (OWASP LLM Top 10 red-teaming) · [guard-gap-detector](https://github.com/shaikn6/guard-gap-detector) (sandbox static analysis) · [credit-arena](https://github.com/shaikn6/credit-arena) (paired bootstrap model comparison)<br/>
**Computer vision and reinforcement learning** [sigdet](https://github.com/shaikn6/sigdet) (YOLO11, multi-seed evaluation) · [exec-rl](https://github.com/shaikn6/exec-rl) (PPO from scratch)<br/>
**Platform and tooling** [fintech-devsecops-pipeline](https://github.com/shaikn6/fintech-devsecops-pipeline) (Terraform, EKS, ArgoCD, OPA) · [readme-truth](https://github.com/shaikn6/readme-truth) (GitHub Action) · GitHub Actions CI in every repo<br/>
**Languages** Python · Go · SQL · TypeScript ([portfolio](https://github.com/shaikn6/portfolio))

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

<sub>Contact: <a href="mailto:nagizaazs@gmail.com">nagizaazs@gmail.com</a> · <!--REPO_COUNT_START-->23 public repos<!--REPO_COUNT_END--> · <a href="https://nagizaaz.vercel.app">Portfolio</a> · <a href="https://www.linkedin.com/in/nagizaazshaik">LinkedIn</a> · <a href="https://huggingface.co/9mark9">Hugging Face</a></sub>

</div>
