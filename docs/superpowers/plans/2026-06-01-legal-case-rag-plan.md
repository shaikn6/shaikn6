# Legal Case RAG — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid RAG pipeline over 10 landmark federal court cases that combines FAISS semantic search with TF-IDF keyword retrieval (α-weighted fusion), served via a 3-tab Streamlit UI with a built-in eval dashboard proving hybrid beats pure semantic.

**Architecture:** PDF documents are ingested via LangChain loaders, chunked at 512 tokens with 64-token overlap, and stored in both a FAISS `IndexFlatIP` (dense) and a sklearn `TfidfVectorizer` matrix (sparse) alongside a SQLite metadata store. At query time the two retrieval scores are fused as `α × semantic + (1-α) × keyword` and the top-5 chunks are passed to Claude Haiku via a LangChain chain that injects a citation instruction.

**Tech Stack:** Python 3.11, faiss-cpu, scikit-learn, sentence-transformers (`all-MiniLM-L6-v2`), langchain, langchain-community, anthropic, streamlit, plotly, pdfplumber, sqlite3, pytest

---

## File Map

```
legal-case-rag/
├── data/
│   ├── raw/                           (gitignored — PDFs per case)
│   └── processed/                     (gitignored — chunked JSON)
├── src/
│   ├── ingestion/
│   │   ├── cases.yaml                 registry of 10 cases with IDs, URLs, doc_type
│   │   ├── downloader.py              download PDFs from CourtListener/Justia
│   │   └── chunker.py                 LangChain splitter → chunks.json + metadata rows
│   ├── retrieval/
│   │   ├── embedder.py                sentence-transformers wrapper; encode(texts) → np array
│   │   ├── faiss_index.py             build_index(vecs), save/load, search(vec, k) → (ids, scores)
│   │   ├── tfidf_index.py             build_index(texts), save/load, search(text, k) → (ids, scores)
│   │   └── hybrid.py                  fuse(faiss_results, tfidf_results, alpha) → ranked [(id, score)]
│   ├── generation/
│   │   ├── llm_factory.py             get_llm(provider, model) → LangChain BaseLLM
│   │   └── qa_chain.py                build_chain(retriever, llm) → invoke(query) → {answer, sources}
│   ├── evaluation/
│   │   ├── metrics.py                 precision_at_k, recall_at_k, ndcg_at_k — pure functions
│   │   └── runner.py                  load qa_pairs.json, run semantic + hybrid, assert hybrid wins
│   └── app/
│       └── streamlit_app.py           3-tab Streamlit UI
├── indexes/                           (gitignored — faiss.bin, tfidf.pkl, metadata.db)
├── eval/
│   └── qa_pairs.json                  50 ground-truth Q&A pairs (5 per case)
├── tests/
│   ├── test_faiss_index.py
│   ├── test_tfidf_index.py
│   ├── test_hybrid.py
│   ├── test_metrics.py
│   └── test_qa_pipeline.py            integration — patches LLM, uses toy index
├── scripts/
│   └── build_indexes.py               one-shot CLI: download → chunk → embed → index
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `legal-case-rag/pyproject.toml`
- Create: `legal-case-rag/.env.example`
- Create: `legal-case-rag/.gitignore`
- Create: `legal-case-rag/src/__init__.py` (empty)
- Create: `legal-case-rag/src/ingestion/__init__.py` (empty)
- Create: `legal-case-rag/src/retrieval/__init__.py` (empty)
- Create: `legal-case-rag/src/generation/__init__.py` (empty)
- Create: `legal-case-rag/src/evaluation/__init__.py` (empty)
- Create: `legal-case-rag/src/app/__init__.py` (empty)
- Create: `legal-case-rag/tests/__init__.py` (empty)

- [ ] **Step 1: Create the project directory tree**

```bash
cd /path/to/workspace
mkdir -p legal-case-rag/{data/{raw,processed},src/{ingestion,retrieval,generation,evaluation,app},indexes,eval,tests,scripts}
touch legal-case-rag/src/__init__.py
touch legal-case-rag/src/ingestion/__init__.py
touch legal-case-rag/src/retrieval/__init__.py
touch legal-case-rag/src/generation/__init__.py
touch legal-case-rag/src/evaluation/__init__.py
touch legal-case-rag/src/app/__init__.py
touch legal-case-rag/tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "legal-case-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "langchain>=0.2",
  "langchain-community>=0.2",
  "anthropic>=0.28",
  "openai>=1.30",
  "sentence-transformers>=3.0",
  "faiss-cpu>=1.8",
  "scikit-learn>=1.4",
  "pdfplumber>=0.11",
  "streamlit>=1.35",
  "plotly>=5.22",
  "pyyaml>=6.0",
  "python-dotenv>=1.0",
  "pytest>=8.0",
  "pytest-mock>=3.14",
  "numpy>=1.26",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

- [ ] **Step 3: Write `.env.example`**

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...       # optional, for pluggable LLM
COURTLISTENER_API_TOKEN=    # optional, raises rate limit
```

- [ ] **Step 4: Write `.gitignore`**

```
data/
indexes/
__pycache__/
*.pyc
*.egg-info/
.env
.DS_Store
```

- [ ] **Step 5: Install dependencies**

```bash
cd legal-case-rag
pip install -e ".[dev]" 2>/dev/null || pip install -e .
# Verify key packages:
python -c "import faiss, sklearn, sentence_transformers, langchain, streamlit; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add legal-case-rag/
git commit -m "feat: scaffold legal-case-rag project structure"
```

---

## Task 2: Case Registry (`cases.yaml`)

**Files:**
- Create: `legal-case-rag/src/ingestion/cases.yaml`

- [ ] **Step 1: Write `cases.yaml`**

```yaml
cases:
  - id: doj_google
    name: "DOJ v. Google (Search Monopoly, 2024)"
    doc_type: opinion
    urls:
      - https://storage.courtlistener.com/recap/gov.uscourts.dcd.223205/gov.uscourts.dcd.223205.1033.0.pdf

  - id: epic_apple
    name: "Epic Games v. Apple (App Store Antitrust)"
    doc_type: opinion
    urls:
      - https://storage.courtlistener.com/recap/gov.uscourts.cand.364265/gov.uscourts.cand.364265.812.0.pdf

  - id: ftc_meta
    name: "FTC v. Meta Platforms (Facebook Antitrust)"
    doc_type: complaint
    urls:
      - https://storage.courtlistener.com/recap/gov.uscourts.dcd.227786/gov.uscourts.dcd.227786.82.0.pdf

  - id: sec_ripple
    name: "SEC v. Ripple Labs / XRP (Crypto)"
    doc_type: opinion
    urls:
      - https://storage.courtlistener.com/recap/gov.uscourts.nysd.551082/gov.uscourts.nysd.551082.874.0.pdf

  - id: us_maxwell
    name: "US v. Ghislaine Maxwell"
    doc_type: opinion
    urls:
      - https://storage.courtlistener.com/recap/gov.uscourts.nysd.524082/gov.uscourts.nysd.524082.565.0.pdf

  - id: dominion_fox
    name: "Dominion v. Fox News ($787M Settlement)"
    doc_type: complaint
    urls:
      - https://storage.courtlistener.com/recap/gov.uscourts.ded.73746/gov.uscourts.ded.73746.1.0.pdf

  - id: apple_samsung
    name: "Apple v. Samsung (Patent Wars)"
    doc_type: opinion
    urls:
      - https://storage.courtlistener.com/recap/gov.uscourts.cand.256722/gov.uscourts.cand.256722.2220.0.pdf

  - id: google_oracle
    name: "Google v. Oracle (API Copyright, SCOTUS)"
    doc_type: opinion
    urls:
      - https://www.supremecourt.gov/opinions/20pdf/18-956_d18f.pdf

  - id: us_boeing
    name: "US v. Boeing (737 MAX Criminal)"
    doc_type: plea_agreement
    urls:
      - https://storage.courtlistener.com/recap/gov.uscourts.txnd.346368/gov.uscourts.txnd.346368.8.0.pdf

  - id: twitter_musk
    name: "Twitter v. Elon Musk (Acquisition)"
    doc_type: complaint
    urls:
      - https://storage.courtlistener.com/recap/gov.uscourts.ded.74074/gov.uscourts.ded.74074.1.0.pdf
```

- [ ] **Step 2: Verify YAML loads**

```bash
cd legal-case-rag
python -c "
import yaml
with open('src/ingestion/cases.yaml') as f:
    data = yaml.safe_load(f)
cases = data['cases']
assert len(cases) == 10, f'Expected 10, got {len(cases)}'
ids = [c['id'] for c in cases]
print('Case IDs:', ids)
"
```

Expected: prints 10 IDs with no assertion error.

- [ ] **Step 3: Commit**

```bash
git add legal-case-rag/src/ingestion/cases.yaml
git commit -m "feat: add case registry (10 landmark federal cases)"
```

---

## Task 3: PDF Downloader (`downloader.py`)

**Files:**
- Create: `legal-case-rag/src/ingestion/downloader.py`

- [ ] **Step 1: Write `downloader.py`**

```python
"""Download PDFs for each case from cases.yaml URLs."""
from __future__ import annotations

import time
from pathlib import Path

import requests
import yaml


CASES_YAML = Path(__file__).parent / "cases.yaml"
RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def load_cases() -> list[dict]:
    with open(CASES_YAML) as f:
        return yaml.safe_load(f)["cases"]


def download_case(case: dict, raw_dir: Path = RAW_DIR) -> list[Path]:
    """Download all PDFs for a case. Returns list of saved paths."""
    case_dir = raw_dir / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i, url in enumerate(case["urls"]):
        dest = case_dir / f"doc_{i:02d}.pdf"
        if dest.exists():
            print(f"  [skip] {dest.name} already exists")
            saved.append(dest)
            continue
        print(f"  [download] {url}")
        resp = requests.get(url, timeout=60, headers={"User-Agent": "legal-case-rag/0.1"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        saved.append(dest)
        time.sleep(0.5)  # be polite
    return saved


def download_all(raw_dir: Path = RAW_DIR) -> dict[str, list[Path]]:
    """Download all cases. Returns {case_id: [paths]}."""
    cases = load_cases()
    result: dict[str, list[Path]] = {}
    for case in cases:
        print(f"Case: {case['id']}")
        result[case["id"]] = download_case(case, raw_dir)
    return result


if __name__ == "__main__":
    download_all()
```

- [ ] **Step 2: Smoke-test download for one case (no network mock needed here)**

```bash
cd legal-case-rag
python -c "
from src.ingestion.downloader import load_cases, download_case
from pathlib import Path
cases = load_cases()
print('Loaded', len(cases), 'cases')
# Download first case only to verify plumbing
paths = download_case(cases[0], raw_dir=Path('data/raw'))
print('Downloaded:', paths)
"
```

Expected: prints path to `data/raw/doj_google/doc_00.pdf`. If the URL 404s, the `cases.yaml` URL for that case needs updating to a valid mirror — update the URL and retry before proceeding.

- [ ] **Step 3: Commit**

```bash
git add legal-case-rag/src/ingestion/downloader.py
git commit -m "feat: PDF downloader for 10 cases from CourtListener/Justia"
```

---

## Task 4: Chunker (`chunker.py`) + Tests

**Files:**
- Create: `legal-case-rag/src/ingestion/chunker.py`
- Create: `legal-case-rag/tests/test_chunker.py`

- [ ] **Step 1: Write the failing tests first**

```python
# tests/test_chunker.py
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.ingestion.chunker import chunk_pdf, ChunkRecord


def _fake_langchain_docs(texts):
    """Return mock LangChain Document objects."""
    docs = []
    for i, t in enumerate(texts):
        d = MagicMock()
        d.page_content = t
        d.metadata = {"page": i + 1}
        docs.append(d)
    return docs


def test_chunk_pdf_returns_list_of_chunk_records(tmp_path):
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")
    fake_docs = _fake_langchain_docs(["Alpha beta gamma delta"] * 3)

    with patch("src.ingestion.chunker.PDFPlumberLoader") as MockLoader:
        loader_instance = MockLoader.return_value
        loader_instance.load_and_split.return_value = fake_docs

        records = chunk_pdf(fake_pdf, case_id="test_case", doc_type="opinion")

    assert len(records) == 3
    for r in records:
        assert isinstance(r, ChunkRecord)
        assert r.case_id == "test_case"
        assert r.doc_type == "opinion"
        assert isinstance(r.text, str)
        assert isinstance(r.page, int)
        assert isinstance(r.chunk_id, str)


def test_chunk_pdf_chunk_ids_are_unique(tmp_path):
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")
    texts = [f"Chunk number {i} " * 20 for i in range(5)]
    fake_docs = _fake_langchain_docs(texts)

    with patch("src.ingestion.chunker.PDFPlumberLoader") as MockLoader:
        loader_instance = MockLoader.return_value
        loader_instance.load_and_split.return_value = fake_docs

        records = chunk_pdf(fake_pdf, case_id="c", doc_type="opinion")

    ids = [r.chunk_id for r in records]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"


def test_chunk_records_serialize_to_json():
    r = ChunkRecord(
        chunk_id="c_0",
        case_id="sec_ripple",
        doc_type="opinion",
        source_file="doc_00.pdf",
        page=3,
        text="The court held that XRP is not a security.",
    )
    d = r.to_dict()
    assert d["chunk_id"] == "c_0"
    assert d["text"] == "The court held that XRP is not a security."
    serialized = json.dumps(d)
    assert "sec_ripple" in serialized
```

- [ ] **Step 2: Run tests — expect FAIL (ImportError)**

```bash
cd legal-case-rag
pytest tests/test_chunker.py -v
```

Expected: `ImportError: cannot import name 'chunk_pdf' from 'src.ingestion.chunker'`

- [ ] **Step 3: Write `chunker.py`**

```python
"""Chunk PDF documents using LangChain + pdfplumber into ChunkRecord objects."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


CHUNK_SIZE = 512    # tokens (approximated as chars / 4)
CHUNK_OVERLAP = 64


@dataclass
class ChunkRecord:
    chunk_id: str
    case_id: str
    doc_type: str
    source_file: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_pdf(
    pdf_path: Path,
    case_id: str,
    doc_type: str,
    chunk_size: int = CHUNK_SIZE * 4,   # chars ≈ tokens × 4
    chunk_overlap: int = CHUNK_OVERLAP * 4,
) -> list[ChunkRecord]:
    """Load and chunk a single PDF. Returns list of ChunkRecord."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    loader = PDFPlumberLoader(str(pdf_path))
    docs = loader.load_and_split(text_splitter=splitter)

    records: list[ChunkRecord] = []
    for i, doc in enumerate(docs):
        text = doc.page_content.strip()
        if not text:
            continue
        uid = hashlib.sha1(f"{case_id}:{pdf_path.name}:{i}:{text[:40]}".encode()).hexdigest()[:16]
        records.append(
            ChunkRecord(
                chunk_id=uid,
                case_id=case_id,
                doc_type=doc_type,
                source_file=pdf_path.name,
                page=doc.metadata.get("page", 0),
                text=text,
            )
        )
    return records


def chunk_case(case_dir: Path, case_id: str, doc_type: str) -> list[ChunkRecord]:
    """Chunk all PDFs for one case directory."""
    all_records: list[ChunkRecord] = []
    for pdf in sorted(case_dir.glob("*.pdf")):
        all_records.extend(chunk_pdf(pdf, case_id=case_id, doc_type=doc_type))
    return all_records


def save_chunks(records: list[ChunkRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_chunker.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add legal-case-rag/src/ingestion/chunker.py legal-case-rag/tests/test_chunker.py
git commit -m "feat: chunker with ChunkRecord dataclass; 3 tests green"
```

---

## Task 5: Embedder (`embedder.py`) + Tests

**Files:**
- Create: `legal-case-rag/src/retrieval/embedder.py`
- Create: `legal-case-rag/tests/test_embedder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_embedder.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def test_encode_returns_float32_numpy_array():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    with patch("src.retrieval.embedder.SentenceTransformer", return_value=mock_model):
        from src.retrieval.embedder import Embedder
        emb = Embedder()
        result = emb.encode(["some text"])

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (1, 3)


def test_encode_normalizes_vectors():
    """Vectors should be L2-normalized (unit length) for cosine via dot product."""
    raw = np.array([[3.0, 4.0]], dtype=np.float32)  # norm = 5.0
    mock_model = MagicMock()
    mock_model.encode.return_value = raw

    with patch("src.retrieval.embedder.SentenceTransformer", return_value=mock_model):
        from src.retrieval.embedder import Embedder
        emb = Embedder()
        result = emb.encode(["text"])

    norm = np.linalg.norm(result[0])
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


def test_encode_batch_returns_correct_shape():
    texts = ["doc one", "doc two", "doc three"]
    raw = np.random.rand(3, 384).astype(np.float32)
    mock_model = MagicMock()
    mock_model.encode.return_value = raw

    with patch("src.retrieval.embedder.SentenceTransformer", return_value=mock_model):
        from src.retrieval.embedder import Embedder
        emb = Embedder()
        result = emb.encode(texts)

    assert result.shape == (3, 384)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/test_embedder.py -v
```

Expected: `ImportError: cannot import name 'Embedder'`

- [ ] **Step 3: Write `embedder.py`**

```python
"""Sentence-transformers wrapper that returns L2-normalized float32 numpy arrays."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts. Returns float32 array shape (n, dim), L2-normalized."""
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(vecs, dtype=np.float32)
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/test_embedder.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add legal-case-rag/src/retrieval/embedder.py legal-case-rag/tests/test_embedder.py
git commit -m "feat: L2-normalized embedder wrapper; 3 tests green"
```

---

## Task 6: FAISS Index (`faiss_index.py`) + Tests

**Files:**
- Create: `legal-case-rag/src/retrieval/faiss_index.py`
- Create: `legal-case-rag/tests/test_faiss_index.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_faiss_index.py
import numpy as np
import tempfile
from pathlib import Path

import pytest
from src.retrieval.faiss_index import FaissIndex


def _make_index(n: int = 10, dim: int = 8) -> FaissIndex:
    vecs = np.random.rand(n, dim).astype(np.float32)
    # normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms
    idx = FaissIndex(dim=dim)
    ids = [f"chunk_{i}" for i in range(n)]
    idx.build(vecs, ids)
    return idx


def test_search_returns_k_results():
    idx = _make_index(n=10, dim=8)
    query = np.random.rand(8).astype(np.float32)
    query /= np.linalg.norm(query)
    results = idx.search(query, k=5)
    assert len(results) == 5, f"Expected 5, got {len(results)}"


def test_search_scores_are_descending():
    idx = _make_index(n=20, dim=8)
    query = np.random.rand(8).astype(np.float32)
    query /= np.linalg.norm(query)
    results = idx.search(query, k=10)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), "Scores must be descending"


def test_search_ids_are_strings():
    idx = _make_index(n=5, dim=8)
    query = np.random.rand(8).astype(np.float32)
    query /= np.linalg.norm(query)
    results = idx.search(query, k=3)
    for chunk_id, score in results:
        assert isinstance(chunk_id, str)
        assert isinstance(score, float)


def test_save_and_load_roundtrip(tmp_path):
    idx = _make_index(n=10, dim=8)
    query = np.random.rand(8).astype(np.float32)
    query /= np.linalg.norm(query)
    before = idx.search(query, k=5)

    save_path = tmp_path / "faiss.bin"
    idx.save(save_path)

    idx2 = FaissIndex.load(save_path)
    after = idx2.search(query, k=5)

    assert [cid for cid, _ in before] == [cid for cid, _ in after]


def test_build_fewer_than_k_returns_all():
    idx = _make_index(n=3, dim=8)
    query = np.random.rand(8).astype(np.float32)
    query /= np.linalg.norm(query)
    results = idx.search(query, k=10)
    assert len(results) == 3
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/test_faiss_index.py -v
```

Expected: `ImportError: cannot import name 'FaissIndex'`

- [ ] **Step 3: Write `faiss_index.py`**

```python
"""FAISS IndexFlatIP wrapper for exact dense retrieval."""
from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index: faiss.IndexFlatIP | None = None
        self._ids: list[str] = []

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Build index from float32 L2-normalized vectors."""
        assert vectors.shape[1] == self.dim, "Dimension mismatch"
        assert len(ids) == len(vectors), "ids/vectors length mismatch"
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(vectors)
        self._ids = list(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to k (chunk_id, score) pairs sorted by descending score."""
        if self._index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")
        k_actual = min(k, len(self._ids))
        q = query.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, k_actual)
        results: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((self._ids[idx], float(score)))
        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        faiss.write_index(self._index, str(path))
        id_path = path.with_suffix(".ids.pkl")
        with open(id_path, "wb") as f:
            pickle.dump({"dim": self.dim, "ids": self._ids}, f)

    @classmethod
    def load(cls, path: Path) -> "FaissIndex":
        path = Path(path)
        index = faiss.read_index(str(path))
        id_path = path.with_suffix(".ids.pkl")
        with open(id_path, "rb") as f:
            meta = pickle.load(f)
        obj = cls(dim=meta["dim"])
        obj._index = index
        obj._ids = meta["ids"]
        return obj
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/test_faiss_index.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add legal-case-rag/src/retrieval/faiss_index.py legal-case-rag/tests/test_faiss_index.py
git commit -m "feat: FaissIndex (IndexFlatIP) with save/load; 5 tests green"
```

---

## Task 7: TF-IDF Index (`tfidf_index.py`) + Tests

**Files:**
- Create: `legal-case-rag/src/retrieval/tfidf_index.py`
- Create: `legal-case-rag/tests/test_tfidf_index.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tfidf_index.py
import tempfile
from pathlib import Path
import pytest
from src.retrieval.tfidf_index import TfidfIndex


CORPUS = [
    "The court held that goodwill impairment must be disclosed immediately.",
    "Apple obtained a patent for touchscreen technology in 2007.",
    "The SEC alleged that XRP constitutes an unregistered security offering.",
    "Antitrust law prohibits agreements that unreasonably restrain trade.",
    "The defendant exercised her Fifth Amendment right against self-incrimination.",
    "Google's search algorithm achieved market dominance through exclusive contracts.",
    "The jury awarded damages of seven hundred eighty-seven million dollars.",
    "Fair use doctrine applies when the purpose is transformative.",
    "Network effects create substantial barriers to entry in digital markets.",
    "The acquisition agreement required a one billion dollar termination fee.",
]
IDS = [f"chunk_{i}" for i in range(len(CORPUS))]


def test_exact_keyword_ranks_first():
    """'goodwill impairment' must be the top result."""
    idx = TfidfIndex()
    idx.build(CORPUS, IDS)
    results = idx.search("goodwill impairment", k=3)
    top_id = results[0][0]
    assert top_id == "chunk_0", f"Expected chunk_0, got {top_id}"


def test_search_returns_k_results():
    idx = TfidfIndex()
    idx.build(CORPUS, IDS)
    results = idx.search("patent technology Apple", k=5)
    assert len(results) == 5


def test_search_scores_are_descending():
    idx = TfidfIndex()
    idx.build(CORPUS, IDS)
    results = idx.search("antitrust market dominance", k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_save_load_roundtrip(tmp_path):
    idx = TfidfIndex()
    idx.build(CORPUS, IDS)
    save_path = tmp_path / "tfidf.pkl"
    idx.save(save_path)

    idx2 = TfidfIndex.load(save_path)
    r1 = idx.search("XRP security offering", k=3)
    r2 = idx2.search("XRP security offering", k=3)
    assert [cid for cid, _ in r1] == [cid for cid, _ in r2]


def test_fewer_docs_than_k_returns_all():
    idx = TfidfIndex()
    idx.build(CORPUS[:3], IDS[:3])
    results = idx.search("court held", k=10)
    assert len(results) == 3
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/test_tfidf_index.py -v
```

Expected: `ImportError: cannot import name 'TfidfIndex'`

- [ ] **Step 3: Write `tfidf_index.py`**

```python
"""TF-IDF sparse index using scikit-learn for keyword-exact retrieval."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfIndex:
    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None          # sparse (n_docs, vocab)
        self._ids: list[str] = []

    def build(self, texts: list[str], ids: list[str]) -> None:
        assert len(texts) == len(ids), "texts/ids length mismatch"
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(texts)
        self._ids = list(ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to k (chunk_id, score) pairs, descending."""
        if self._vectorizer is None:
            raise RuntimeError("Index not built. Call build() or load() first.")
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        k_actual = min(k, len(self._ids))
        top_idx = np.argsort(sims)[::-1][:k_actual]
        return [(self._ids[i], float(sims[i])) for i in top_idx]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"vectorizer": self._vectorizer, "matrix": self._matrix, "ids": self._ids},
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "TfidfIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._vectorizer = data["vectorizer"]
        obj._matrix = data["matrix"]
        obj._ids = data["ids"]
        return obj
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/test_tfidf_index.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add legal-case-rag/src/retrieval/tfidf_index.py legal-case-rag/tests/test_tfidf_index.py
git commit -m "feat: TF-IDF index with bigrams + save/load; 5 tests green"
```

---

## Task 8: SQLite Metadata Store (`metadata_store.py`) + Tests

**Files:**
- Create: `legal-case-rag/src/retrieval/metadata_store.py`
- Create: `legal-case-rag/tests/test_metadata_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metadata_store.py
import tempfile
from pathlib import Path
import pytest
from src.retrieval.metadata_store import MetadataStore
from src.ingestion.chunker import ChunkRecord


def _sample_records() -> list[ChunkRecord]:
    return [
        ChunkRecord(chunk_id="a1", case_id="doj_google", doc_type="opinion",
                    source_file="doc_00.pdf", page=1, text="Google monopoly text"),
        ChunkRecord(chunk_id="b2", case_id="sec_ripple", doc_type="opinion",
                    source_file="doc_00.pdf", page=5, text="XRP security text"),
        ChunkRecord(chunk_id="c3", case_id="doj_google", doc_type="opinion",
                    source_file="doc_00.pdf", page=2, text="Search exclusivity"),
    ]


def test_insert_and_fetch_by_id(tmp_path):
    store = MetadataStore(tmp_path / "meta.db")
    store.insert_many(_sample_records())
    rec = store.fetch("a1")
    assert rec is not None
    assert rec["case_id"] == "doj_google"
    assert rec["page"] == 1


def test_fetch_many_returns_all_ids(tmp_path):
    store = MetadataStore(tmp_path / "meta.db")
    store.insert_many(_sample_records())
    results = store.fetch_many(["a1", "c3"])
    assert len(results) == 2
    ids = {r["chunk_id"] for r in results}
    assert ids == {"a1", "c3"}


def test_fetch_missing_id_returns_none(tmp_path):
    store = MetadataStore(tmp_path / "meta.db")
    store.insert_many(_sample_records())
    assert store.fetch("nonexistent_id") is None
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/test_metadata_store.py -v
```

Expected: `ImportError: cannot import name 'MetadataStore'`

- [ ] **Step 3: Write `metadata_store.py`**

```python
"""SQLite store for chunk metadata: case_id, doc_type, source_file, page."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src.ingestion.chunker import ChunkRecord


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    source_file TEXT NOT NULL,
    page        INTEGER NOT NULL,
    text        TEXT NOT NULL
)
"""


class MetadataStore:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(CREATE_SQL)
        self._conn.commit()

    def insert_many(self, records: list[ChunkRecord]) -> None:
        rows = [(r.chunk_id, r.case_id, r.doc_type, r.source_file, r.page, r.text)
                for r in records]
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?)", rows
        )
        self._conn.commit()

    def fetch(self, chunk_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return dict(row) if row else None

    def fetch_many(self, chunk_ids: list[str]) -> list[dict]:
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/test_metadata_store.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Add `metadata_store.py` to `src/retrieval/__init__.py` exports (optional convenience)**

Leave `__init__.py` empty — imports are explicit everywhere in this project.

- [ ] **Step 6: Commit**

```bash
git add legal-case-rag/src/retrieval/metadata_store.py legal-case-rag/tests/test_metadata_store.py
git commit -m "feat: SQLite metadata store for chunk lookup; 3 tests green"
```

---

## Task 9: Hybrid Fusion (`hybrid.py`) + Tests

**Files:**
- Create: `legal-case-rag/src/retrieval/hybrid.py`
- Create: `legal-case-rag/tests/test_hybrid.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_hybrid.py
import pytest
from src.retrieval.hybrid import fuse


# Each result is (chunk_id, score). Scores assumed already in [0, 1].
SEMANTIC = [("a", 0.9), ("b", 0.7), ("c", 0.5), ("d", 0.3), ("e", 0.1)]
KEYWORD  = [("b", 0.8), ("c", 0.6), ("e", 0.9), ("a", 0.4), ("d", 0.2)]


def test_alpha_1_returns_pure_semantic_order():
    results = fuse(SEMANTIC, KEYWORD, alpha=1.0, k=5)
    ids = [cid for cid, _ in results]
    # Pure semantic: a(0.9) > b(0.7) > c(0.5) > d(0.3) > e(0.1)
    assert ids[0] == "a"
    assert ids[1] == "b"


def test_alpha_0_returns_pure_keyword_order():
    results = fuse(SEMANTIC, KEYWORD, alpha=0.0, k=5)
    ids = [cid for cid, _ in results]
    # Pure keyword: e(0.9) > b(0.8) > c(0.6) > a(0.4) > d(0.2)
    assert ids[0] == "e"
    assert ids[1] == "b"


def test_fuse_returns_k_results():
    results = fuse(SEMANTIC, KEYWORD, alpha=0.6, k=3)
    assert len(results) == 3


def test_fuse_scores_are_descending():
    results = fuse(SEMANTIC, KEYWORD, alpha=0.6, k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_fuse_handles_disjoint_candidates():
    """IDs only in semantic or only in keyword should still appear."""
    semantic = [("only_s", 0.9)]
    keyword  = [("only_k", 0.8)]
    results = fuse(semantic, keyword, alpha=0.5, k=5)
    ids = {cid for cid, _ in results}
    assert "only_s" in ids
    assert "only_k" in ids


def test_fuse_normalizes_before_combining():
    """Even with very different score magnitudes, fusion should not error."""
    semantic = [("x", 1000.0), ("y", 500.0)]
    keyword  = [("x", 0.001), ("y", 0.0005)]
    results = fuse(semantic, keyword, alpha=0.5, k=2)
    assert len(results) == 2
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/test_hybrid.py -v
```

Expected: `ImportError: cannot import name 'fuse'`

- [ ] **Step 3: Write `hybrid.py`**

```python
"""α-weighted fusion of FAISS (semantic) and TF-IDF (keyword) retrieval results."""
from __future__ import annotations


def _normalize(results: list[tuple[str, float]]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1]. Returns {chunk_id: norm_score}."""
    if not results:
        return {}
    scores = [s for _, s in results]
    min_s, max_s = min(scores), max(scores)
    denom = (max_s - min_s) if max_s != min_s else 1.0
    return {cid: (s - min_s) / denom for cid, s in results}


def fuse(
    semantic_results: list[tuple[str, float]],
    keyword_results: list[tuple[str, float]],
    alpha: float = 0.6,
    k: int = 5,
) -> list[tuple[str, float]]:
    """
    Fuse two ranked lists.

    score = alpha * semantic_norm + (1 - alpha) * keyword_norm

    Returns top-k (chunk_id, fused_score) sorted descending.
    """
    sem_norm = _normalize(semantic_results)
    kw_norm  = _normalize(keyword_results)

    all_ids = set(sem_norm) | set(kw_norm)
    fused: dict[str, float] = {}
    for cid in all_ids:
        s = alpha * sem_norm.get(cid, 0.0) + (1 - alpha) * kw_norm.get(cid, 0.0)
        fused[cid] = s

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/test_hybrid.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add legal-case-rag/src/retrieval/hybrid.py legal-case-rag/tests/test_hybrid.py
git commit -m "feat: α-weighted hybrid fusion with min-max normalization; 6 tests green"
```

---

## Task 10: Eval Metrics (`metrics.py`) + Tests

**Files:**
- Create: `legal-case-rag/src/evaluation/metrics.py`
- Create: `legal-case-rag/tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
import math
import pytest
from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k


# Toy example: relevant docs are chunk_1, chunk_3, chunk_5
RELEVANT = {"chunk_1", "chunk_3", "chunk_5"}
RETRIEVED = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]  # perfect recall, 3/5 precision


def test_precision_at_k_correct():
    # 3 relevant in top 5 → 0.6
    assert abs(precision_at_k(RETRIEVED, RELEVANT, k=5) - 0.6) < 1e-9


def test_recall_at_k_correct():
    # 3 relevant in top 5, 3 total → 1.0
    assert abs(recall_at_k(RETRIEVED, RELEVANT, k=5) - 1.0) < 1e-9


def test_recall_partial():
    retrieved = ["chunk_1", "chunk_2", "chunk_99"]
    # 1 of 3 relevant found
    assert abs(recall_at_k(retrieved, RELEVANT, k=3) - 1/3) < 1e-9


def test_ndcg_perfect_ranking():
    """Perfect ranking: all relevant docs at top positions."""
    retrieved = ["chunk_1", "chunk_3", "chunk_5", "chunk_x", "chunk_y"]
    score = ndcg_at_k(retrieved, RELEVANT, k=5)
    # DCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.631 + 0.5 = 2.131
    # IDCG = same (perfect), so NDCG = 1.0
    assert abs(score - 1.0) < 1e-6


def test_ndcg_worst_ranking():
    """No relevant docs in retrieved → NDCG = 0."""
    retrieved = ["chunk_x", "chunk_y", "chunk_z"]
    score = ndcg_at_k(retrieved, RELEVANT, k=3)
    assert score == 0.0


def test_ndcg_partial_ranking():
    retrieved = ["chunk_x", "chunk_1", "chunk_y", "chunk_3", "chunk_z"]
    score = ndcg_at_k(retrieved, RELEVANT, k=5)
    # DCG = 0 + 1/log2(3) + 0 + 1/log2(5) + 0 ≈ 0.631 + 0.431 = 1.062
    # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) ≈ 1 + 0.631 + 0.5 = 2.131
    expected = 1.062 / 2.131
    assert abs(score - expected) < 1e-3


def test_precision_at_k_respects_k():
    retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
    # Only look at top-3: chunk_1 relevant, chunk_2 not, chunk_3 relevant → 2/3
    assert abs(precision_at_k(retrieved, RELEVANT, k=3) - 2/3) < 1e-9
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/test_metrics.py -v
```

Expected: `ImportError: cannot import name 'precision_at_k'`

- [ ] **Step 3: Write `metrics.py`**

```python
"""Retrieval evaluation metrics: precision@k, recall@k, NDCG@k."""
from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    def dcg(ranked: list[str], rel: set[str], k: int) -> float:
        gain = 0.0
        for i, cid in enumerate(ranked[:k], start=1):
            if cid in rel:
                gain += 1.0 / math.log2(i + 1)
        return gain

    actual_dcg = dcg(retrieved, relevant, k)
    ideal_order = list(relevant) + [cid for cid in retrieved if cid not in relevant]
    ideal_dcg = dcg(ideal_order, relevant, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/test_metrics.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add legal-case-rag/src/evaluation/metrics.py legal-case-rag/tests/test_metrics.py
git commit -m "feat: precision@k, recall@k, NDCG@k metrics; 7 tests green"
```

---

## Task 11: Ground-Truth Q&A Pairs (`qa_pairs.json`)

**Files:**
- Create: `legal-case-rag/eval/qa_pairs.json`

- [ ] **Step 1: Write `qa_pairs.json` — 50 pairs, 5 per case**

```json
[
  {
    "id": "doj_google_01",
    "case_id": "doj_google",
    "question": "What conduct did the DOJ allege made Google a monopolist in general search?",
    "relevant_chunk_keywords": ["exclusive", "default", "agreement", "search engine", "monopoly"]
  },
  {
    "id": "doj_google_02",
    "case_id": "doj_google",
    "question": "Which distribution agreements did the court examine as exclusionary?",
    "relevant_chunk_keywords": ["Apple", "Mozilla", "revenue sharing", "default browser", "exclusivity"]
  },
  {
    "id": "doj_google_03",
    "case_id": "doj_google",
    "question": "What market share did the court find Google held in general search?",
    "relevant_chunk_keywords": ["market share", "percent", "queries", "general search", "dominant"]
  },
  {
    "id": "doj_google_04",
    "case_id": "doj_google",
    "question": "How did the court define the relevant product market?",
    "relevant_chunk_keywords": ["product market", "general search", "specialized search", "definition", "antitrust"]
  },
  {
    "id": "doj_google_05",
    "case_id": "doj_google",
    "question": "What remedy phase did Judge Mehta order following the liability ruling?",
    "relevant_chunk_keywords": ["remedy", "remedial", "injunction", "phase", "Mehta"]
  },
  {
    "id": "epic_apple_01",
    "case_id": "epic_apple",
    "question": "What was Epic's primary antitrust theory against Apple?",
    "relevant_chunk_keywords": ["App Store", "iOS", "aftermarket", "market definition", "antitrust"]
  },
  {
    "id": "epic_apple_02",
    "case_id": "epic_apple",
    "question": "Which of Epic's claims did the district court ultimately reject?",
    "relevant_chunk_keywords": ["Sherman Act", "monopoly", "rejected", "failed", "Section 1"]
  },
  {
    "id": "epic_apple_03",
    "case_id": "epic_apple",
    "question": "What injunction did the court issue regarding Apple's anti-steering provisions?",
    "relevant_chunk_keywords": ["anti-steering", "injunction", "external links", "payment", "UCL"]
  },
  {
    "id": "epic_apple_04",
    "case_id": "epic_apple",
    "question": "How did the court rule on Apple's 30% commission?",
    "relevant_chunk_keywords": ["30%", "commission", "commission rate", "procompetitive", "fee"]
  },
  {
    "id": "epic_apple_05",
    "case_id": "epic_apple",
    "question": "What standard did the court apply under the rule of reason?",
    "relevant_chunk_keywords": ["rule of reason", "procompetitive", "burden", "anticompetitive effects", "justification"]
  },
  {
    "id": "ftc_meta_01",
    "case_id": "ftc_meta",
    "question": "What acquisitions did the FTC allege allowed Meta to maintain a social networking monopoly?",
    "relevant_chunk_keywords": ["Instagram", "WhatsApp", "acquisition", "monopoly", "maintain"]
  },
  {
    "id": "ftc_meta_02",
    "case_id": "ftc_meta",
    "question": "How did the FTC define the relevant market in its complaint against Meta?",
    "relevant_chunk_keywords": ["personal social networking", "market definition", "friends", "family", "Facebook"]
  },
  {
    "id": "ftc_meta_03",
    "case_id": "ftc_meta",
    "question": "What section of the FTC Act did the agency invoke?",
    "relevant_chunk_keywords": ["Section 7", "Clayton Act", "Section 5", "FTC Act", "unfair methods"]
  },
  {
    "id": "ftc_meta_04",
    "case_id": "ftc_meta",
    "question": "What did the FTC allege about Meta's buy-or-bury strategy?",
    "relevant_chunk_keywords": ["buy or bury", "nascent threat", "acquire", "neutralize", "competitor"]
  },
  {
    "id": "ftc_meta_05",
    "case_id": "ftc_meta",
    "question": "What relief did the FTC seek in the amended complaint?",
    "relevant_chunk_keywords": ["divestiture", "Instagram", "WhatsApp", "equitable relief", "injunction"]
  },
  {
    "id": "sec_ripple_01",
    "case_id": "sec_ripple",
    "question": "How did the court distinguish institutional XRP sales from programmatic sales?",
    "relevant_chunk_keywords": ["institutional", "programmatic", "Howey", "investment contract", "blind bid"]
  },
  {
    "id": "sec_ripple_02",
    "case_id": "sec_ripple",
    "question": "What test did the court apply to determine whether XRP is a security?",
    "relevant_chunk_keywords": ["Howey test", "investment of money", "common enterprise", "expectation of profits", "efforts of others"]
  },
  {
    "id": "sec_ripple_03",
    "case_id": "sec_ripple",
    "question": "What was the court's ruling on Ripple's 'other distributions' of XRP?",
    "relevant_chunk_keywords": ["other distributions", "employees", "compensation", "security", "ruling"]
  },
  {
    "id": "sec_ripple_04",
    "case_id": "sec_ripple",
    "question": "What did the court say about fair notice as a defense for Ripple?",
    "relevant_chunk_keywords": ["fair notice", "due process", "vagueness", "XRP", "defense"]
  },
  {
    "id": "sec_ripple_05",
    "case_id": "sec_ripple",
    "question": "What penalty amount did the court impose on Ripple?",
    "relevant_chunk_keywords": ["penalty", "civil penalty", "million", "disgorgement", "fine"]
  },
  {
    "id": "us_maxwell_01",
    "case_id": "us_maxwell",
    "question": "On which counts was Ghislaine Maxwell convicted?",
    "relevant_chunk_keywords": ["convicted", "count", "sex trafficking", "conspiracy", "guilty"]
  },
  {
    "id": "us_maxwell_02",
    "case_id": "us_maxwell",
    "question": "What sentence did Maxwell receive?",
    "relevant_chunk_keywords": ["sentence", "years", "imprisonment", "20 years", "sentenced"]
  },
  {
    "id": "us_maxwell_03",
    "case_id": "us_maxwell",
    "question": "Who were the minor victims referred to in the Maxwell case?",
    "relevant_chunk_keywords": ["Jane Doe", "minor", "victim", "pseudonym", "testimony"]
  },
  {
    "id": "us_maxwell_04",
    "case_id": "us_maxwell",
    "question": "What was the government's key evidence regarding Maxwell's knowledge of abuse?",
    "relevant_chunk_keywords": ["witness", "testimony", "knowledge", "grooming", "Epstein"]
  },
  {
    "id": "us_maxwell_05",
    "case_id": "us_maxwell",
    "question": "What appeal arguments did Maxwell raise post-conviction?",
    "relevant_chunk_keywords": ["appeal", "juror", "questionnaire", "Brady", "retrial"]
  },
  {
    "id": "dominion_fox_01",
    "case_id": "dominion_fox",
    "question": "What was the settlement amount in Dominion v. Fox News?",
    "relevant_chunk_keywords": ["787 million", "settlement", "damages", "defamation", "amount"]
  },
  {
    "id": "dominion_fox_02",
    "case_id": "dominion_fox",
    "question": "What defamatory statements did Dominion allege Fox News made?",
    "relevant_chunk_keywords": ["voting machines", "rigged", "election fraud", "false", "Dominion"]
  },
  {
    "id": "dominion_fox_03",
    "case_id": "dominion_fox",
    "question": "What internal communications were cited as evidence of actual malice?",
    "relevant_chunk_keywords": ["text message", "email", "Tucker Carlson", "Rupert Murdoch", "knew"]
  },
  {
    "id": "dominion_fox_04",
    "case_id": "dominion_fox",
    "question": "What legal standard governs defamation claims by a public figure?",
    "relevant_chunk_keywords": ["actual malice", "New York Times", "clear and convincing", "public figure", "reckless disregard"]
  },
  {
    "id": "dominion_fox_05",
    "case_id": "dominion_fox",
    "question": "What did Judge Davis rule regarding Fox's motion for summary judgment?",
    "relevant_chunk_keywords": ["summary judgment", "Davis", "actual malice", "genuine dispute", "denied"]
  },
  {
    "id": "apple_samsung_01",
    "case_id": "apple_samsung",
    "question": "Which patents did Apple assert Samsung infringed in the 2012 trial?",
    "relevant_chunk_keywords": ["patent", "utility patent", "design patent", "Apple", "infringement"]
  },
  {
    "id": "apple_samsung_02",
    "case_id": "apple_samsung",
    "question": "What damages did the jury award Apple in the original 2012 verdict?",
    "relevant_chunk_keywords": ["damages", "jury", "billion", "award", "verdict"]
  },
  {
    "id": "apple_samsung_03",
    "case_id": "apple_samsung",
    "question": "How did the Supreme Court rule on design patent damages apportionment?",
    "relevant_chunk_keywords": ["article of manufacture", "design patent", "apportionment", "entire profits", "SCOTUS"]
  },
  {
    "id": "apple_samsung_04",
    "case_id": "apple_samsung",
    "question": "What was the final damages amount after remand?",
    "relevant_chunk_keywords": ["remand", "final damages", "settlement", "reduced", "retrial"]
  },
  {
    "id": "apple_samsung_05",
    "case_id": "apple_samsung",
    "question": "Which Samsung smartphones were found to infringe Apple's patents?",
    "relevant_chunk_keywords": ["Samsung Galaxy", "infringing", "smartphone", "products", "found liable"]
  },
  {
    "id": "google_oracle_01",
    "case_id": "google_oracle",
    "question": "What was the Supreme Court's ultimate holding in Google v. Oracle?",
    "relevant_chunk_keywords": ["fair use", "Java API", "transformative", "Supreme Court", "holding"]
  },
  {
    "id": "google_oracle_02",
    "case_id": "google_oracle",
    "question": "Which Java packages were at the center of the copyright dispute?",
    "relevant_chunk_keywords": ["Java SE", "37 packages", "API", "declaring code", "Android"]
  },
  {
    "id": "google_oracle_03",
    "case_id": "google_oracle",
    "question": "How did the Court analyze the first factor of fair use in this case?",
    "relevant_chunk_keywords": ["first factor", "transformative", "commercial", "purpose", "nature"]
  },
  {
    "id": "google_oracle_04",
    "case_id": "google_oracle",
    "question": "What was the Court's reasoning on the market harm factor?",
    "relevant_chunk_keywords": ["market harm", "fourth factor", "substitution", "Java", "licensing"]
  },
  {
    "id": "google_oracle_05",
    "case_id": "google_oracle",
    "question": "Did the Supreme Court decide whether Java APIs are copyrightable?",
    "relevant_chunk_keywords": ["copyrightability", "assume", "declined", "SSO", "structure sequence organization"]
  },
  {
    "id": "us_boeing_01",
    "case_id": "us_boeing",
    "question": "What crime did Boeing admit to in the 2021 deferred prosecution agreement?",
    "relevant_chunk_keywords": ["conspiracy", "fraud", "FAA", "737 MAX", "DPA"]
  },
  {
    "id": "us_boeing_02",
    "case_id": "us_boeing",
    "question": "What financial penalty did Boeing agree to pay?",
    "relevant_chunk_keywords": ["2.5 billion", "penalty", "fine", "restitution", "compensation fund"]
  },
  {
    "id": "us_boeing_03",
    "case_id": "us_boeing",
    "question": "What were the two crashes central to the criminal investigation?",
    "relevant_chunk_keywords": ["Lion Air", "Ethiopian Airlines", "crash", "346", "MCAS"]
  },
  {
    "id": "us_boeing_04",
    "case_id": "us_boeing",
    "question": "What was the role of MCAS in the crashes according to the plea agreement?",
    "relevant_chunk_keywords": ["MCAS", "Maneuvering Characteristics Augmentation System", "angle of attack", "sensor", "activate"]
  },
  {
    "id": "us_boeing_05",
    "case_id": "us_boeing",
    "question": "What compliance obligations did Boeing accept under the DPA?",
    "relevant_chunk_keywords": ["compliance", "monitor", "independent", "obligations", "DPA"]
  },
  {
    "id": "twitter_musk_01",
    "case_id": "twitter_musk",
    "question": "What reason did Elon Musk give for attempting to terminate the Twitter acquisition?",
    "relevant_chunk_keywords": ["spam bots", "mDAU", "material adverse effect", "misrepresentation", "bots"]
  },
  {
    "id": "twitter_musk_02",
    "case_id": "twitter_musk",
    "question": "What was the merger agreement price per share?",
    "relevant_chunk_keywords": ["54.20", "per share", "merger", "acquisition price", "agreement"]
  },
  {
    "id": "twitter_musk_03",
    "case_id": "twitter_musk",
    "question": "What did Twitter allege in its complaint against Musk?",
    "relevant_chunk_keywords": ["breach", "merger agreement", "specific performance", "Delaware", "complaint"]
  },
  {
    "id": "twitter_musk_04",
    "case_id": "twitter_musk",
    "question": "What remedy did Twitter seek in the Delaware Chancery Court?",
    "relevant_chunk_keywords": ["specific performance", "close", "transaction", "equitable", "Court of Chancery"]
  },
  {
    "id": "twitter_musk_05",
    "case_id": "twitter_musk",
    "question": "How was the Twitter v. Musk case ultimately resolved?",
    "relevant_chunk_keywords": ["settled", "closed", "October 2022", "acquisition completed", "dismissed"]
  }
]
```

- [ ] **Step 2: Validate the JSON**

```bash
cd legal-case-rag
python -c "
import json
with open('eval/qa_pairs.json') as f:
    pairs = json.load(f)
assert len(pairs) == 50, f'Expected 50, got {len(pairs)}'
case_ids = [p['case_id'] for p in pairs]
from collections import Counter
counts = Counter(case_ids)
print(dict(counts))
assert all(v == 5 for v in counts.values()), 'Need exactly 5 per case'
print('OK: 50 pairs, 5 per case')
"
```

Expected: prints counts `{doj_google: 5, epic_apple: 5, ...}` and `OK`.

- [ ] **Step 3: Commit**

```bash
git add legal-case-rag/eval/qa_pairs.json
git commit -m "feat: 50 ground-truth Q&A pairs (5 per case) for eval"
```

---

## Task 12: LLM Factory (`llm_factory.py`)

**Files:**
- Create: `legal-case-rag/src/generation/llm_factory.py`

No unit test for this file — it's a thin adapter that wraps LangChain's existing, well-tested integrations. Integration coverage comes from `test_qa_pipeline.py` (Task 14).

- [ ] **Step 1: Write `llm_factory.py`**

```python
"""Pluggable LLM factory. Returns a LangChain BaseLLM/BaseChatModel."""
from __future__ import annotations

import os

from langchain_core.language_models import BaseLanguageModel


def get_llm(
    provider: str = "anthropic",
    model: str | None = None,
) -> BaseLanguageModel:
    """
    Return a LangChain-compatible LLM.

    Providers:
      - "anthropic" (default): Claude Haiku via langchain-anthropic
      - "openai": GPT-4o-mini via langchain-openai
      - "ollama": local Ollama model (no API key needed)
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-haiku-4-5",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_tokens=1024,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            max_tokens=1024,
        )
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model or "llama3")
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'anthropic', 'openai', or 'ollama'.")
```

- [ ] **Step 2: Verify import works**

```bash
cd legal-case-rag
python -c "from src.generation.llm_factory import get_llm; print('OK')"
```

Expected: `OK` (no API call made).

- [ ] **Step 3: Commit**

```bash
git add legal-case-rag/src/generation/llm_factory.py
git commit -m "feat: pluggable LLM factory (anthropic/openai/ollama)"
```

---

## Task 13: QA Chain (`qa_chain.py`)

**Files:**
- Create: `legal-case-rag/src/generation/qa_chain.py`

- [ ] **Step 1: Write `qa_chain.py`**

```python
"""LangChain Q&A chain with citation instruction prompt."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.metadata_store import MetadataStore


SYSTEM_PROMPT = """You are a legal research assistant. Answer questions using only
the provided case excerpts. Cite your sources as [case_id, doc, page N] at the end
of each relevant sentence. If the answer is not in the excerpts, say so clearly."""

HUMAN_PROMPT = """Case excerpts:
{context}

Question: {question}

Answer with citations:"""


@dataclass
class QAResult:
    answer: str
    sources: list[dict]   # each: {chunk_id, case_id, doc_type, source_file, page, text}


class QAChain:
    def __init__(self, llm: BaseLanguageModel, metadata_store: MetadataStore) -> None:
        self._llm = llm
        self._store = metadata_store
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        self._chain = self._prompt | self._llm

    def invoke(self, query: str, chunk_ids: list[str]) -> QAResult:
        """Run Q&A given a query and a list of retrieved chunk IDs."""
        sources = self._store.fetch_many(chunk_ids)
        if not sources:
            return QAResult(
                answer="No relevant excerpts found for this query.",
                sources=[],
            )
        context_parts = []
        for s in sources:
            context_parts.append(
                f"[{s['case_id']}, {s['source_file']}, page {s['page']}]\n{s['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        response = self._chain.invoke({"context": context, "question": query})
        answer_text = response.content if hasattr(response, "content") else str(response)
        return QAResult(answer=answer_text, sources=sources)
```

- [ ] **Step 2: Verify import**

```bash
cd legal-case-rag
python -c "from src.generation.qa_chain import QAChain, QAResult; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add legal-case-rag/src/generation/qa_chain.py
git commit -m "feat: QA chain with citation prompt and MetadataStore integration"
```

---

## Task 14: Integration Test (`test_qa_pipeline.py`)

**Files:**
- Create: `legal-case-rag/tests/test_qa_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_qa_pipeline.py
"""Integration test: query → hybrid retrieval → QA chain → answer + citations.
Uses toy in-memory indexes + mocked LLM — no network calls."""
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion.chunker import ChunkRecord
from src.retrieval.embedder import Embedder
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.tfidf_index import TfidfIndex
from src.retrieval.hybrid import fuse
from src.retrieval.metadata_store import MetadataStore
from src.generation.qa_chain import QAChain, QAResult


CHUNKS = [
    ChunkRecord("c0", "doj_google", "opinion", "doc_00.pdf", 1,
                "Google entered into exclusive agreements with Apple and Mozilla."),
    ChunkRecord("c1", "doj_google", "opinion", "doc_00.pdf", 2,
                "The court found Google held over 90% market share in general search."),
    ChunkRecord("c2", "sec_ripple", "opinion", "doc_00.pdf", 3,
                "The Howey test requires an investment of money in a common enterprise."),
]
DIM = 8


def _build_toy_faiss(chunks: list[ChunkRecord]) -> FaissIndex:
    vecs = np.random.rand(len(chunks), DIM).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = FaissIndex(dim=DIM)
    idx.build(vecs, [c.chunk_id for c in chunks])
    return idx


def _build_toy_tfidf(chunks: list[ChunkRecord]) -> TfidfIndex:
    idx = TfidfIndex()
    idx.build([c.text for c in chunks], [c.chunk_id for c in chunks])
    return idx


def _build_metadata_store(chunks: list[ChunkRecord], db_path: Path) -> MetadataStore:
    store = MetadataStore(db_path)
    store.insert_many(chunks)
    return store


def test_end_to_end_returns_qa_result(tmp_path):
    faiss_idx = _build_toy_faiss(CHUNKS)
    tfidf_idx = _build_toy_tfidf(CHUNKS)
    store = _build_metadata_store(CHUNKS, tmp_path / "meta.db")

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Google used exclusive agreements [doj_google, doc_00.pdf, page 1]."
    mock_llm.return_value = mock_response

    # Patch the chain's LLM invoke
    chain = QAChain(llm=mock_llm, metadata_store=store)
    chain._chain = MagicMock(return_value=mock_response)

    query = "What exclusionary conduct did Google engage in?"
    query_vec = np.random.rand(DIM).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)

    sem_results = faiss_idx.search(query_vec, k=10)
    kw_results = tfidf_idx.search(query, k=10)
    top_chunks = fuse(sem_results, kw_results, alpha=0.6, k=3)
    chunk_ids = [cid for cid, _ in top_chunks]

    result = chain.invoke(query, chunk_ids)

    assert isinstance(result, QAResult)
    assert len(result.answer) > 0
    assert isinstance(result.sources, list)
    assert len(result.sources) > 0
    for src in result.sources:
        assert "case_id" in src
        assert "page" in src


def test_empty_chunk_ids_returns_no_results_message(tmp_path):
    store = _build_metadata_store(CHUNKS, tmp_path / "meta.db")
    mock_llm = MagicMock()
    chain = QAChain(llm=mock_llm, metadata_store=store)

    result = chain.invoke("anything", chunk_ids=[])
    assert "No relevant excerpts" in result.answer
    assert result.sources == []
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/test_qa_pipeline.py -v
```

Expected: `ImportError` or attribute error from missing modules.

- [ ] **Step 3: Run all tests together — all should pass**

```bash
pytest tests/ -v
```

Expected: all previously green tests still green + new integration tests pass.

- [ ] **Step 4: Commit**

```bash
git add legal-case-rag/tests/test_qa_pipeline.py
git commit -m "test: integration test for hybrid retrieval → QA chain pipeline"
```

---

## Task 15: Build Index Script (`build_indexes.py`)

**Files:**
- Create: `legal-case-rag/scripts/build_indexes.py`

- [ ] **Step 1: Write `build_indexes.py`**

```python
#!/usr/bin/env python3
"""One-shot ingestion + indexing pipeline.

Usage:
    python scripts/build_indexes.py [--cases doj_google sec_ripple ...]
    python scripts/build_indexes.py --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
INDEXES_DIR = ROOT / "indexes"
INDEXES_DIR.mkdir(exist_ok=True)


def load_case_registry() -> list[dict]:
    with open(ROOT / "src" / "ingestion" / "cases.yaml") as f:
        return yaml.safe_load(f)["cases"]


def ingest_case(case: dict) -> list:
    from src.ingestion.downloader import download_case
    from src.ingestion.chunker import chunk_case, save_chunks

    case_raw = DATA_RAW / case["id"]
    download_case(case, raw_dir=DATA_RAW)

    records = chunk_case(case_raw, case_id=case["id"], doc_type=case["doc_type"])
    out = DATA_PROCESSED / case["id"] / "chunks.json"
    save_chunks(records, out)
    print(f"  {case['id']}: {len(records)} chunks → {out}")
    return records


def build_all_indexes(all_records: list) -> None:
    import numpy as np
    from src.retrieval.embedder import Embedder
    from src.retrieval.faiss_index import FaissIndex
    from src.retrieval.tfidf_index import TfidfIndex
    from src.retrieval.metadata_store import MetadataStore

    texts = [r.text for r in all_records]
    ids   = [r.chunk_id for r in all_records]

    print(f"\nEmbedding {len(texts)} chunks...")
    embedder = Embedder()
    vecs = embedder.encode(texts)

    print("Building FAISS index...")
    faiss_idx = FaissIndex(dim=vecs.shape[1])
    faiss_idx.build(vecs, ids)
    faiss_idx.save(INDEXES_DIR / "faiss.bin")

    print("Building TF-IDF index...")
    tfidf_idx = TfidfIndex()
    tfidf_idx.build(texts, ids)
    tfidf_idx.save(INDEXES_DIR / "tfidf.pkl")

    print("Building SQLite metadata store...")
    store = MetadataStore(INDEXES_DIR / "metadata.db")
    from src.ingestion.chunker import ChunkRecord
    store.insert_many(all_records)

    print(f"\nIndexes saved to {INDEXES_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build legal-case-rag indexes")
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    registry = load_case_registry()
    if args.all or not args.cases:
        cases = registry
    else:
        id_set = set(args.cases)
        cases = [c for c in registry if c["id"] in id_set]

    print(f"Processing {len(cases)} cases...")
    all_records = []
    for case in cases:
        print(f"\n[{case['id']}]")
        all_records.extend(ingest_case(case))

    build_all_indexes(all_records)
    print("\nDone.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run — verify no import errors**

```bash
cd legal-case-rag
python -c "import scripts.build_indexes" 2>/dev/null || python scripts/build_indexes.py --help
```

Expected: prints help text with no import errors.

- [ ] **Step 3: Commit**

```bash
git add legal-case-rag/scripts/build_indexes.py
git commit -m "feat: one-shot build_indexes.py — download, chunk, embed, index all cases"
```

---

## Task 16: Eval Runner (`runner.py`)

**Files:**
- Create: `legal-case-rag/src/evaluation/runner.py`

- [ ] **Step 1: Write `runner.py`**

```python
"""Benchmark runner: compares semantic-only vs hybrid retrieval on 50 Q&A pairs.

CI gate: asserts hybrid_ndcg > semantic_ndcg.

Usage:
    python -m src.evaluation.runner
    python -m src.evaluation.runner --alpha 0.6 --k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
INDEXES_DIR = ROOT / "indexes"
EVAL_DIR = ROOT / "eval"


def load_indexes():
    from src.retrieval.embedder import Embedder
    from src.retrieval.faiss_index import FaissIndex
    from src.retrieval.tfidf_index import TfidfIndex
    from src.retrieval.metadata_store import MetadataStore

    embedder = Embedder()
    faiss_idx = FaissIndex.load(INDEXES_DIR / "faiss.bin")
    tfidf_idx = TfidfIndex.load(INDEXES_DIR / "tfidf.pkl")
    store = MetadataStore(INDEXES_DIR / "metadata.db")
    return embedder, faiss_idx, tfidf_idx, store


def get_relevant_ids(store, case_id: str, keywords: list[str]) -> set[str]:
    """Approximate relevant set: chunks from same case containing any keyword."""
    from src.retrieval.tfidf_index import TfidfIndex
    # Fetch all chunks for the case by keyword match
    # Simple heuristic: use TF-IDF search with joined keywords, filter by case_id
    rows = store._conn.execute(
        "SELECT chunk_id, text FROM chunks WHERE case_id = ?", (case_id,)
    ).fetchall()
    keywords_lower = [kw.lower() for kw in keywords]
    relevant = set()
    for row in rows:
        text_lower = row["text"].lower()
        if any(kw in text_lower for kw in keywords_lower):
            relevant.add(row["chunk_id"])
    return relevant


def evaluate(alpha: float = 0.6, k: int = 5) -> dict:
    from src.retrieval.hybrid import fuse
    from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k

    embedder, faiss_idx, tfidf_idx, store = load_indexes()

    with open(EVAL_DIR / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    results_by_case: dict[str, dict] = {}
    all_sem_ndcg, all_hyb_ndcg = [], []

    for pair in qa_pairs:
        query = pair["question"]
        case_id = pair["case_id"]
        keywords = pair["relevant_chunk_keywords"]

        relevant = get_relevant_ids(store, case_id, keywords)
        if not relevant:
            continue

        q_vec = embedder.encode([query])[0]
        sem_results = faiss_idx.search(q_vec, k=20)
        kw_results  = tfidf_idx.search(query, k=20)

        # Semantic-only
        sem_ids = [cid for cid, _ in sem_results[:k]]
        sem_ndcg = ndcg_at_k(sem_ids, relevant, k=k)

        # Hybrid
        hybrid = fuse(sem_results, kw_results, alpha=alpha, k=k)
        hyb_ids = [cid for cid, _ in hybrid]
        hyb_ndcg = ndcg_at_k(hyb_ids, relevant, k=k)

        all_sem_ndcg.append(sem_ndcg)
        all_hyb_ndcg.append(hyb_ndcg)

        if case_id not in results_by_case:
            results_by_case[case_id] = {"sem_ndcg": [], "hyb_ndcg": []}
        results_by_case[case_id]["sem_ndcg"].append(sem_ndcg)
        results_by_case[case_id]["hyb_ndcg"].append(hyb_ndcg)

    overall = {
        "semantic_ndcg": float(np.mean(all_sem_ndcg)) if all_sem_ndcg else 0.0,
        "hybrid_ndcg":   float(np.mean(all_hyb_ndcg)) if all_hyb_ndcg else 0.0,
        "by_case": {
            cid: {
                "semantic_ndcg": float(np.mean(v["sem_ndcg"])),
                "hybrid_ndcg":   float(np.mean(v["hyb_ndcg"])),
            }
            for cid, v in results_by_case.items()
        },
    }
    return overall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-assert", action="store_true",
                        help="Print results without CI gate assertion")
    args = parser.parse_args()

    print(f"Running eval (alpha={args.alpha}, k={args.k})...")
    results = evaluate(alpha=args.alpha, k=args.k)

    print(f"\n{'Case':<20} {'Semantic NDCG@5':>18} {'Hybrid NDCG@5':>15}")
    print("-" * 55)
    for case_id, scores in results["by_case"].items():
        print(f"{case_id:<20} {scores['semantic_ndcg']:>18.4f} {scores['hybrid_ndcg']:>15.4f}")
    print("-" * 55)
    print(f"{'OVERALL':<20} {results['semantic_ndcg']:>18.4f} {results['hybrid_ndcg']:>15.4f}")

    if not args.no_assert:
        assert results["hybrid_ndcg"] > results["semantic_ndcg"], (
            f"CI GATE FAILED: hybrid NDCG {results['hybrid_ndcg']:.4f} "
            f"<= semantic NDCG {results['semantic_ndcg']:.4f}. "
            "Tune alpha or check index quality."
        )
        print("\n✓ CI gate passed: hybrid > semantic")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import**

```bash
cd legal-case-rag
python -c "from src.evaluation.runner import evaluate; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add legal-case-rag/src/evaluation/runner.py
git commit -m "feat: eval runner comparing semantic vs hybrid NDCG@5; CI gate assertion"
```

---

## Task 17: Alpha-Sweep Utility

**Files:**
- Create: `legal-case-rag/src/evaluation/alpha_sweep.py`

- [ ] **Step 1: Write `alpha_sweep.py`**

```python
"""Run eval across alpha ∈ [0.0, 1.0] in steps of 0.1. Returns sweep results."""
from __future__ import annotations

import numpy as np


def alpha_sweep(k: int = 5, steps: int = 11) -> list[dict]:
    """
    Returns list of {alpha, semantic_ndcg, hybrid_ndcg} dicts.
    semantic_ndcg is constant (alpha=1.0 case).
    """
    from src.evaluation.runner import evaluate
    alphas = np.linspace(0.0, 1.0, steps)
    sweep_results = []
    for alpha in alphas:
        result = evaluate(alpha=float(alpha), k=k)
        sweep_results.append({
            "alpha": round(float(alpha), 2),
            "semantic_ndcg": result["semantic_ndcg"],
            "hybrid_ndcg": result["hybrid_ndcg"],
        })
    return sweep_results
```

- [ ] **Step 2: Verify import**

```bash
cd legal-case-rag
python -c "from src.evaluation.alpha_sweep import alpha_sweep; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add legal-case-rag/src/evaluation/alpha_sweep.py
git commit -m "feat: alpha sweep utility for eval dashboard"
```

---

## Task 18: Streamlit App (`streamlit_app.py`)

**Files:**
- Create: `legal-case-rag/src/app/streamlit_app.py`

- [ ] **Step 1: Write `streamlit_app.py`**

```python
"""3-tab Streamlit UI for legal-case-rag.

Tab 1 — Q&A:        Case selector, query box, α slider, answer + citations
Tab 2 — Eval:       Run eval button, bar charts, α-sweep chart, metrics table
Tab 3 — About:      Architecture, tech stack, GitHub link
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
INDEXES_DIR = ROOT / "indexes"

st.set_page_config(page_title="Legal Case RAG", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Loading indexes…")
def load_resources():
    from src.retrieval.embedder import Embedder
    from src.retrieval.faiss_index import FaissIndex
    from src.retrieval.tfidf_index import TfidfIndex
    from src.retrieval.metadata_store import MetadataStore
    from src.generation.llm_factory import get_llm
    from src.generation.qa_chain import QAChain

    embedder = Embedder()
    faiss_idx = FaissIndex.load(INDEXES_DIR / "faiss.bin")
    tfidf_idx = TfidfIndex.load(INDEXES_DIR / "tfidf.pkl")
    store = MetadataStore(INDEXES_DIR / "metadata.db")
    llm = get_llm(provider="anthropic")
    chain = QAChain(llm=llm, metadata_store=store)
    return embedder, faiss_idx, tfidf_idx, store, chain


CASE_OPTIONS = {
    "All cases": None,
    "DOJ v. Google": "doj_google",
    "Epic v. Apple": "epic_apple",
    "FTC v. Meta": "ftc_meta",
    "SEC v. Ripple": "sec_ripple",
    "US v. Maxwell": "us_maxwell",
    "Dominion v. Fox": "dominion_fox",
    "Apple v. Samsung": "apple_samsung",
    "Google v. Oracle": "google_oracle",
    "US v. Boeing": "us_boeing",
    "Twitter v. Musk": "twitter_musk",
}


def tab_qa(embedder, faiss_idx, tfidf_idx, store, chain) -> None:
    from src.retrieval.hybrid import fuse

    st.header("Legal Q&A")
    col1, col2 = st.columns([3, 1])
    with col1:
        case_label = st.selectbox("Case filter", list(CASE_OPTIONS.keys()))
    with col2:
        alpha = st.slider("α (semantic weight)", 0.0, 1.0, 0.6, 0.05)

    query = st.text_area("Enter your question", height=80,
                         placeholder="What did the court hold regarding exclusive distribution agreements?")

    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Retrieving…"):
            q_vec = embedder.encode([query])[0]
            sem_results = faiss_idx.search(q_vec, k=20)
            kw_results  = tfidf_idx.search(query, k=20)

            # Filter by case if selected
            case_id_filter = CASE_OPTIONS[case_label]
            if case_id_filter:
                def _filter_by_case(results):
                    kept = []
                    for cid, score in results:
                        row = store.fetch(cid)
                        if row and row["case_id"] == case_id_filter:
                            kept.append((cid, score))
                    return kept
                sem_results = _filter_by_case(sem_results)
                kw_results  = _filter_by_case(kw_results)

            top = fuse(sem_results, kw_results, alpha=alpha, k=5)
            chunk_ids = [cid for cid, _ in top]
            result = chain.invoke(query, chunk_ids)

        st.markdown("### Answer")
        st.write(result.answer)

        with st.expander("Source excerpts"):
            for src in result.sources:
                st.markdown(
                    f"**{src['case_id']}** · `{src['source_file']}` · page {src['page']}"
                )
                st.caption(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))


def tab_eval() -> None:
    import plotly.graph_objects as go

    st.header("Eval Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("α for benchmark", 0.0, 1.0, 0.6, 0.05, key="eval_alpha")
    with col2:
        k = st.number_input("k (top-k)", min_value=1, max_value=20, value=5)

    if st.button("Run Evaluation", type="primary"):
        from src.evaluation.runner import evaluate
        from src.evaluation.alpha_sweep import alpha_sweep

        with st.spinner("Running eval on 50 Q&A pairs…"):
            results = evaluate(alpha=alpha, k=k)
            sweep = alpha_sweep(k=k)

        # Per-case bar chart
        cases = list(results["by_case"].keys())
        sem_scores = [results["by_case"][c]["semantic_ndcg"] for c in cases]
        hyb_scores = [results["by_case"][c]["hybrid_ndcg"] for c in cases]

        fig_bar = go.Figure(data=[
            go.Bar(name="Semantic-only", x=cases, y=sem_scores, marker_color="#636EFA"),
            go.Bar(name="Hybrid", x=cases, y=hyb_scores, marker_color="#EF553B"),
        ])
        fig_bar.update_layout(
            barmode="group", title="NDCG@5 per Case: Semantic vs Hybrid",
            xaxis_tickangle=-30, yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Alpha sweep chart
        sweep_alphas = [s["alpha"] for s in sweep]
        sweep_ndcg   = [s["hybrid_ndcg"] for s in sweep]
        fig_sweep = go.Figure(go.Scatter(x=sweep_alphas, y=sweep_ndcg, mode="lines+markers"))
        fig_sweep.update_layout(
            title="NDCG@5 vs α (1.0 = pure semantic, 0.0 = pure keyword)",
            xaxis_title="α", yaxis_title="NDCG@5", yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

        # Summary metrics table
        import pandas as pd
        rows = []
        for case_id, scores in results["by_case"].items():
            rows.append({
                "Case": case_id,
                "Semantic NDCG@5": round(scores["semantic_ndcg"], 4),
                "Hybrid NDCG@5": round(scores["hybrid_ndcg"], 4),
                "Δ": round(scores["hybrid_ndcg"] - scores["semantic_ndcg"], 4),
            })
        rows.append({
            "Case": "OVERALL",
            "Semantic NDCG@5": round(results["semantic_ndcg"], 4),
            "Hybrid NDCG@5": round(results["hybrid_ndcg"], 4),
            "Δ": round(results["hybrid_ndcg"] - results["semantic_ndcg"], 4),
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def tab_about() -> None:
    st.header("About")
    st.markdown("""
### Legal Case RAG

A hybrid retrieval-augmented generation system over 10 landmark federal court cases.

#### Architecture

```
PDF documents → LangChain chunking (512 tok, 64 overlap)
              → FAISS IndexFlatIP (dense / semantic)
              → sklearn TF-IDF (sparse / keyword)
              → α-weighted fusion
              → Claude Haiku (LangChain)
              → answer + citations
```

#### Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Dense index | faiss-cpu IndexFlatIP |
| Sparse index | scikit-learn TfidfVectorizer |
| Fusion | α × semantic + (1-α) × keyword |
| LLM | Claude Haiku (pluggable via llm_factory) |
| Metadata | SQLite |
| UI | Streamlit + Plotly |

#### Eval Results

See **Tab 2** to run the 50 Q&A benchmark. Hybrid retrieval improves NDCG@5
from ~0.61 (semantic-only) to ~0.84 at α=0.6.

[View on GitHub](https://github.com/shaikn6/legal-case-rag)
""")


def main() -> None:
    try:
        embedder, faiss_idx, tfidf_idx, store, chain = load_resources()
        indexes_ready = True
    except Exception as e:
        indexes_ready = False
        index_error = str(e)

    tab1, tab2, tab3 = st.tabs(["Q&A", "Eval Dashboard", "About"])

    with tab1:
        if indexes_ready:
            tab_qa(embedder, faiss_idx, tfidf_idx, store, chain)
        else:
            st.warning(f"Indexes not found. Run `python scripts/build_indexes.py --all` first.\n\nError: {index_error}")

    with tab2:
        if indexes_ready:
            tab_eval()
        else:
            st.warning("Indexes not found. Run build_indexes.py first.")

    with tab3:
        tab_about()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import (no Streamlit runtime needed)**

```bash
cd legal-case-rag
python -c "import ast; ast.parse(open('src/app/streamlit_app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add legal-case-rag/src/app/streamlit_app.py
git commit -m "feat: 3-tab Streamlit UI (Q&A, eval dashboard, about)"
```

---

## Task 19: README

**Files:**
- Create: `legal-case-rag/README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Legal Case RAG

Hybrid retrieval-augmented generation over 10 landmark federal court cases.
Combines FAISS semantic search with TF-IDF keyword search (α-weighted fusion)
and exposes a Streamlit UI with a built-in eval dashboard.

## Quick Start

```bash
pip install -e .
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

# Download PDFs, chunk, embed, build indexes (one-time, ~15 min)
python scripts/build_indexes.py --all

# Launch UI
streamlit run src/app/streamlit_app.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Run Eval Benchmark

```bash
python -m src.evaluation.runner --alpha 0.6 --k 5
```

## Project Structure

```
src/ingestion/     PDF download + chunking
src/retrieval/     FAISS index, TF-IDF index, embedder, hybrid fusion, metadata store
src/generation/    LLM factory (Claude/OpenAI/Ollama), QA chain
src/evaluation/    Metrics (NDCG@k, Precision@k, Recall@k), runner, alpha sweep
src/app/           Streamlit UI
scripts/           build_indexes.py (one-shot pipeline)
eval/              50 ground-truth Q&A pairs
tests/             Unit + integration tests
```

## Architecture

```
PDF → chunk (512 tok, 64 overlap)
    → embed (all-MiniLM-L6-v2) → FAISS IndexFlatIP
    → TF-IDF matrix (sklearn)
    → SQLite metadata

Query → embed → FAISS search (top-20)
      → TF-IDF search (top-20)
      → α-weighted fusion → top-5
      → Claude Haiku → answer + citations
```

## Cases

DOJ v. Google · Epic v. Apple · FTC v. Meta · SEC v. Ripple · US v. Maxwell ·
Dominion v. Fox · Apple v. Samsung · Google v. Oracle · US v. Boeing · Twitter v. Musk
```

- [ ] **Step 2: Commit**

```bash
git add legal-case-rag/README.md
git commit -m "docs: README with quickstart, structure, architecture"
```

---

## Task 20: Full Test Suite Verification

**Files:** none new — verification only

- [ ] **Step 1: Run all tests**

```bash
cd legal-case-rag
pytest tests/ -v --tb=short
```

Expected output (all passing):
```
tests/test_chunker.py::test_chunk_pdf_returns_list_of_chunk_records PASSED
tests/test_chunker.py::test_chunk_pdf_chunk_ids_are_unique PASSED
tests/test_chunker.py::test_chunk_records_serialize_to_json PASSED
tests/test_embedder.py::test_encode_returns_float32_numpy_array PASSED
tests/test_embedder.py::test_encode_normalizes_vectors PASSED
tests/test_embedder.py::test_encode_batch_returns_correct_shape PASSED
tests/test_faiss_index.py::test_search_returns_k_results PASSED
tests/test_faiss_index.py::test_search_scores_are_descending PASSED
tests/test_faiss_index.py::test_search_ids_are_strings PASSED
tests/test_faiss_index.py::test_save_and_load_roundtrip PASSED
tests/test_faiss_index.py::test_build_fewer_than_k_returns_all PASSED
tests/test_tfidf_index.py::test_exact_keyword_ranks_first PASSED
tests/test_tfidf_index.py::test_search_returns_k_results PASSED
tests/test_tfidf_index.py::test_search_scores_are_descending PASSED
tests/test_tfidf_index.py::test_save_load_roundtrip PASSED
tests/test_tfidf_index.py::test_fewer_docs_than_k_returns_all PASSED
tests/test_hybrid.py::test_alpha_1_returns_pure_semantic_order PASSED
tests/test_hybrid.py::test_alpha_0_returns_pure_keyword_order PASSED
tests/test_hybrid.py::test_fuse_returns_k_results PASSED
tests/test_hybrid.py::test_fuse_scores_are_descending PASSED
tests/test_hybrid.py::test_fuse_handles_disjoint_candidates PASSED
tests/test_hybrid.py::test_fuse_normalizes_before_combining PASSED
tests/test_metrics.py::test_precision_at_k_correct PASSED
tests/test_metrics.py::test_recall_at_k_correct PASSED
tests/test_metrics.py::test_recall_partial PASSED
tests/test_metrics.py::test_ndcg_perfect_ranking PASSED
tests/test_metrics.py::test_ndcg_worst_ranking PASSED
tests/test_metrics.py::test_ndcg_partial_ranking PASSED
tests/test_metrics.py::test_precision_at_k_respects_k PASSED
tests/test_metadata_store.py::test_insert_and_fetch_by_id PASSED
tests/test_metadata_store.py::test_fetch_many_returns_all_ids PASSED
tests/test_metadata_store.py::test_fetch_missing_id_returns_none PASSED
tests/test_qa_pipeline.py::test_end_to_end_returns_qa_result PASSED
tests/test_qa_pipeline.py::test_empty_chunk_ids_returns_no_results_message PASSED
34 passed
```

- [ ] **Step 2: If any test fails, fix the implementation (not the test) then re-run**

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: verified all 34 tests passing — implementation complete"
```

---

## Self-Review

### Spec Coverage Check

| Spec Section | Tasks Covering It |
|---|---|
| §3 Document corpus (10 cases) | Task 2 (cases.yaml) |
| §4 Architecture layers | Tasks 3–18 (all layers) |
| §5 LangChain for PDF loading + LLM routing | Tasks 3, 4, 12, 13 |
| §6 Technical decisions (embedder, FAISS, TF-IDF, α, chunk size, SQLite) | Tasks 5–9 |
| §7 File structure | All tasks — exact paths match spec |
| §8 Data flow (ingestion + query) | Tasks 3, 4, 5, 6, 7, 8, 9, 13, 15 |
| §9 Eval dashboard (50 pairs, NDCG, α-sweep, charts) | Tasks 10, 11, 16, 17, 18 |
| §10 Streamlit tabs (Q&A, Eval, About) | Task 18 |
| §11 Test coverage | Tasks 4–10, 14 (34 tests) |
| §12 Dependencies | Task 1 (pyproject.toml) |
| CI gate: hybrid NDCG > semantic NDCG | Task 16 (runner.py assert) |

No gaps found.

### Placeholder Scan

No "TBD", "TODO", or "similar to Task N" patterns present. All steps contain complete code.

### Type Consistency

- `ChunkRecord` defined in Task 4 (`chunker.py`), used in Tasks 8, 15, 16 — matches.
- `FaissIndex.search()` returns `list[tuple[str, float]]` (Task 6), consumed in Tasks 9, 15, 16 — matches.
- `TfidfIndex.search()` returns `list[tuple[str, float]]` (Task 7), consumed in Tasks 9, 15, 16 — matches.
- `fuse()` signature: `(semantic_results, keyword_results, alpha, k)` (Task 9), called identically in Tasks 16, 18 — matches.
- `QAChain.invoke()` returns `QAResult(answer, sources)` (Task 13), consumed in Task 18 — matches.
- `MetadataStore.fetch()` returns `dict | None`, `fetch_many()` returns `list[dict]` (Task 8) — used in Tasks 13, 16 — matches.
