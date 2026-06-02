# Public Repos (Live)

Approved public as of 2026-05-30. All have: MIT license · shields.io badges · Quick Start · no PII.

## Status: V1 (polished, not V2)

All 5 are V1 with polish applied (badges, Quick Start, license). V2 upgrade pending.

| Repo | Domain | What It Does | Score |
|------|--------|-------------|-------|
| `flight-ops-intelligence` | Data Engineering | ML delay predictor + weather correlation + Folium route map (500 flights) | 8/10 |
| `sql-to-dag-compiler` | Data Engineering | Parses Oracle SQL/PLSQL → auto-generates Airflow 2.x DAGs with lineage | 9/10 |
| `llm-safety-auditor` | LLM Safety | 50-attack adversarial red-teaming, OWASP LLM Top 10 scoring | 9/10 |
| `agentic-pipeline-healer` | MLOps / LLMOps | LangGraph agent monitors Airflow, diagnoses failures, AST-level fixes | 9/10 |
| `medical-imaging-ai` | Clinical AI | Chest X-ray CNN + Grad-CAM: Pneumonia/Cardiomegaly/Effusion detection | 8/10 |

## Polish Applied

Each public repo has:
- [x] `README.md` with shields.io badges (Python 3.11, MIT, Tests passing, Stack)
- [x] `LICENSE` (MIT)
- [x] Quick Start section (clone → install → test → run)
- [x] No 🔒 private banner
- [x] No email or PII anywhere

## What Could Still Be Improved (V2 Candidates)

See `v2-upgrades/changelog.md` for the 15 repos already upgraded.
These 5 are still on V1 — they can receive V2 treatment:

- `flight-ops-intelligence` → add live weather API integration, real-time delay predictor
- `sql-to-dag-compiler` → add dbt support, unit tests for edge-case SQL
- `llm-safety-auditor` → add automated mutation engine, more attack vectors
- `agentic-pipeline-healer` → add Slack alert integration, multi-DAG orchestration
- `medical-imaging-ai` → upgrade to BioBERT preprocessing, DICOM support
