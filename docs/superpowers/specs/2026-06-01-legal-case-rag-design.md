# Legal Case RAG — Design Spec

**Date:** 2026-06-01  
**Project:** `legal-case-rag`  
**Status:** Approved

---

## 1. Problem

Most RAG implementations use pure semantic (embedding) search. This fails on legal documents because:
- Exact legal terms ("17 U.S.C. § 512", statute numbers, case citations, party names) are missed by semantic search
- Ranking is inconsistent across long documents
- Irrelevant chunks surface when meaning overlaps but context differs

Hybrid retrieval — combining semantic embeddings with keyword (TF-IDF) search — addresses this by capturing both intent and exact terms.

---

## 2. Goal

Build a hybrid RAG system over 10 landmark federal court cases. Prove via a built-in eval dashboard that hybrid retrieval beats pure semantic on legal Q&A. Showcase on portfolio as a differentiated ML Engineer project.

**Resume bullet target:**
```
Engineered hybrid RAG pipeline over 10 landmark federal court cases
(FAISS semantic + TF-IDF keyword, α-weighted fusion); improved
NDCG@5 from ~0.61 → 0.84 vs embedding-only baseline on 50
ground-truth legal Q&A pairs.
```

---

## 3. Document Corpus

10 cases sourced from CourtListener (free) and Justia:

| ID | Case |
|----|------|
| `doj_google` | DOJ v. Google (search monopoly, 2024) |
| `epic_apple` | Epic v. Apple (App Store antitrust) |
| `ftc_meta` | FTC v. Meta (Facebook antitrust) |
| `sec_ripple` | SEC v. Ripple/XRP (crypto) |
| `us_maxwell` | US v. Ghislaine Maxwell (Epstein federal case) |
| `dominion_fox` | Dominion v. Fox News ($787M settlement) |
| `apple_samsung` | Apple v. Samsung (patent wars) |
| `google_oracle` | Google v. Oracle (API copyright, SCOTUS) |
| `us_boeing` | US v. Boeing (737 MAX criminal) |
| `twitter_musk` | Twitter v. Elon Musk (acquisition) |

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│  [Tab 1: Q&A]  [Tab 2: Eval Dashboard]  [Tab 3: About]│
└────────────┬───────────────────────────┬─────────────┘
             │ query                     │ metrics
             ▼                           ▼
┌────────────────────────┐   ┌───────────────────────┐
│   Generation Layer     │   │    Eval Layer (custom) │
│   LangChain LLM router │   │  Precision@5, NDCG@5  │
│   Claude Haiku default │   │  50 ground-truth Q&As │
└────────┬───────────────┘   └───────────────────────┘
         │ top-5 chunks
         ▼
┌────────────────────────────────────────────────────┐
│              Retrieval Layer (CUSTOM)               │
│  FAISS (dense)  +  TF-IDF (sparse)                 │
│  score = α × semantic + (1-α) × keyword            │
│  α = 0.6 default, tunable via UI slider            │
└────────┬───────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────┐
│               Indexing Layer (custom)               │
│  FAISS IndexFlatIP  |  sklearn TF-IDF matrix        │
│  SQLite metadata (case, doc_type, chunk_id, page)   │
└────────┬───────────────────────────────────────────┘
         ▼
┌────────────────────────────────────────────────────┐
│           Data Layer (LangChain loaders)            │
│  CourtListener API + Justia → PDF → pdfplumber      │
│  512-token chunks, 64-token overlap                 │
└────────────────────────────────────────────────────┘
```

---

## 5. Approach: B — LangChain Orchestration + Custom Retrieval Core

- **LangChain handles:** PDF loading, chunking, LLM routing (pluggable)
- **Custom code handles:** FAISS indexing, TF-IDF matrix, α-weighted fusion, eval metrics
- **Rationale:** Framework where it saves time; custom where the resume bullet lives

---

## 6. Technical Decisions

| Concern | Decision | Reason |
|---------|----------|--------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast, free, good domain quality |
| Dense index | FAISS `IndexFlatIP` | Exact search → reproducible benchmarks |
| Sparse index | sklearn `TfidfVectorizer` | No extra infra, well-understood |
| Fusion | α-weighted: `α×dense + (1-α)×sparse` | Simple, interpretable, tunable |
| α default | 0.6 | Empirically favors semantic for long docs |
| Chunking | 512 tokens, 64 overlap | Balances context vs. precision |
| Metadata store | SQLite | No extra infra, sufficient for 10 cases |
| LLM default | Claude Haiku (Anthropic SDK) | Cheap eval runs, Anthropic alignment |
| LLM abstraction | LangChain LLM factory | Pluggable: Claude / OpenAI / Ollama |
| UI | Streamlit | Fast to build, strong portfolio visual |

---

## 7. File Structure

```
legal-case-rag/
├── data/
│   ├── raw/                    # downloaded PDFs per case
│   └── processed/              # chunked JSON + metadata
├── src/
│   ├── ingestion/
│   │   ├── downloader.py       # CourtListener API + Justia scraper
│   │   ├── chunker.py          # LangChain text splitter wrapper
│   │   └── cases.yaml          # case registry (id, name, URLs, doc_type)
│   ├── retrieval/              # CUSTOM — the differentiator
│   │   ├── embedder.py         # sentence-transformers wrapper
│   │   ├── faiss_index.py      # build + query FAISS IndexFlatIP
│   │   ├── tfidf_index.py      # build + query sklearn TF-IDF
│   │   └── hybrid.py           # α-weighted fusion
│   ├── generation/
│   │   ├── llm_factory.py      # pluggable: Claude/OpenAI/Ollama
│   │   └── qa_chain.py         # prompt template + LangChain chain
│   ├── evaluation/
│   │   ├── metrics.py          # precision@k, recall@k, NDCG@k
│   │   └── runner.py           # runs semantic vs hybrid benchmark
│   └── app/
│       └── streamlit_app.py    # 3-tab Streamlit UI
├── indexes/                    # persisted FAISS + TF-IDF artifacts
├── eval/
│   └── qa_pairs.json           # 50 ground-truth Q&A pairs (5 per case)
├── tests/
│   ├── test_faiss_index.py
│   ├── test_tfidf_index.py
│   ├── test_hybrid.py
│   ├── test_metrics.py
│   └── test_qa_pipeline.py     # integration
├── scripts/
│   └── build_indexes.py        # one-shot ingestion + indexing CLI
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 8. Data Flow

**Ingestion (one-time, `scripts/build_indexes.py`):**
```
cases.yaml
→ downloader.py → data/raw/<case_id>/*.pdf
→ chunker.py (LangChain RecursiveCharacterTextSplitter, 512 tok, 64 overlap)
→ data/processed/<case_id>/chunks.json
→ embedder.py → FAISS IndexFlatIP → indexes/faiss.bin
→ TfidfVectorizer.fit_transform → indexes/tfidf.pkl
→ SQLite → indexes/metadata.db
```

**Query (runtime):**
```
user query
→ embedder.py → FAISS.search(top_k=20)
→ tfidf_index.py → cosine_similarity(top_k=20)
→ hybrid.py: α×semantic_score + (1-α)×tfidf_score → rerank → top-5
→ qa_chain.py → LLM prompt (context + query + cite instruction)
→ answer + citations (case, doc, page number)
```

---

## 9. Eval Dashboard (Tab 2)

**Dataset:** 50 hand-curated Q&A pairs, 5 per case  
**Metrics:** Precision@5, Recall@5, NDCG@5  
**Visualizations:**
- Bar chart: semantic-only vs hybrid per case (NDCG@5)
- α-sweep chart: NDCG@5 vs α ∈ [0.0, 1.0] — shows optimal blend
- Summary table: all metrics side-by-side

**CI gate:** `runner.py` asserts `hybrid_ndcg > semantic_ndcg` — fails build if hybrid doesn't win.

---

## 10. Streamlit UI Tabs

**Tab 1 — Q&A:**
- Case selector (all 10 + "All cases")
- Query text box
- α slider (0.0–1.0, default 0.6)
- Answer panel + collapsible source citations (case, doc, page)

**Tab 2 — Eval Dashboard:**
- Run eval button
- Bar charts + α-sweep
- Raw metrics table

**Tab 3 — About:**
- Architecture diagram (static image)
- Tech stack badges
- Link to GitHub

---

## 11. Testing

| Test | Type | Assertion |
|------|------|-----------|
| `test_faiss_index` | Unit | top-k returns k results, scores sorted desc |
| `test_tfidf_index` | Unit | exact keyword "goodwill impairment" ranks #1 |
| `test_hybrid` | Unit | α=1.0 → pure semantic; α=0.0 → pure keyword |
| `test_metrics` | Unit | NDCG, precision, recall correct on toy data |
| `test_qa_pipeline` | Integration | end-to-end query returns answer + citations |
| `runner.py` CI | Eval gate | hybrid NDCG > semantic NDCG |

---

## 12. Dependencies

```toml
[project]
dependencies = [
  "langchain",
  "langchain-community",
  "anthropic",
  "openai",
  "sentence-transformers",
  "faiss-cpu",
  "scikit-learn",
  "pdfplumber",
  "streamlit",
  "plotly",
  "pyyaml",
  "python-dotenv",
  "pytest",
]
```

---

## 13. Out of Scope

- Real-time document ingestion (new filings)
- Authentication / multi-user
- Document upload UI
- Fine-tuned embeddings
- Reranking with cross-encoder (future improvement)
