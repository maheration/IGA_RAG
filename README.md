# Clinical RAG Assistant — KDIGO IgAN/IgAV Guideline

A production-style Retrieval-Augmented Generation (RAG) system that allows nephrologists and fellows to query the KDIGO 2025 Clinical Practice Guideline for IgA Nephropathy (IgAN) and IgA Vasculitis (IgAV) in natural language and receive accurate, cited, guideline-grounded answers.

---

## Why this project exists

This is a portfolio and learning project built to apply RAG concepts on a real clinical document, following a structured production blueprint. The goal is to practice every phase of a RAG system end-to-end — from document ingestion through evaluation and production monitoring — not just to build a demo. The document scope is a single guideline PDF for now, but the architecture is designed to extend to the full KDIGO guideline library without rework.

---

## Folder structure

```
IGA_RAG/
│
├── docs/
│   ├── BLUEPRINT_DECISIONS.md   # Every architectural decision, with rationale. Read this first.
│   └── test_set.jsonl           # 30-question evaluation test set (Phase 0.5)
│
├── data/
│   ├── raw/                     # Original source PDFs (unmodified)
│   └── cleaned/                 # Extracted and cleaned text, ready for chunking
│
├── src/
│   ├── ingestion/               # Phase 1 — extraction, chunking, embedding, indexing
│   ├── query/                   # Phase 2 — query rewriting, retrieval, reranking, generation
│   └── eval/                    # Phase 3 — retrieval metrics, LLM-judge, scoring scripts
│
└── README.md                    # This file
```

---

## How to run it

> This section is updated as each phase is implemented. Check back after each phase.

### Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd IGA_RAG

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Mac/Linux

# Install dependencies
uv pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Fill in your API keys in .env (OpenAI, etc.)
```

### Phase 1 — Ingest documents

```bash
# TODO: will be added after Phase 1 implementation
```

### Phase 2 — Run a query

```bash
# TODO: will be added after Phase 2 implementation
```

### Phase 3 — Run evaluation

```bash
# TODO: will be added after Phase 3 implementation
```

---

## Key design decisions

All architectural decisions (chunking strategy, embedding model, vector store, retrieval method, reranking approach, success metrics, etc.) are documented with their rationale in:

**`docs/BLUEPRINT_DECISIONS.md`**

Read that file before touching any code — it explains not just *what* was decided but *why*, which is what the Cursor agent (or any contributor) needs to implement things correctly.

---

## Tech stack

> Finalized incrementally as each phase is decided. Current status:

| Component | Choice | Status |
|---|---|---|
| Source document | KDIGO 2025 IgAN/IgAV Executive Summary PDF | Confirmed |
| Package manager | uv | Confirmed |
| PDF extraction | TBD — Phase 1.1 | Pending |
| Chunking | TBD — Phase 1.2 | Pending |
| Embedding model | TBD — Phase 1.5 | Pending |
| Vector store | TBD — Phase 1.6 | Pending |
| Reranking | LLM-based reranking | Confirmed |
| Generation model | TBD — Phase 2.8 | Pending |
| Evaluation | LLM-as-judge, Recall@10, keyword coverage | Confirmed |

---

## Success targets

| Metric | Target |
|---|---|
| Retrieval — Recall@10 | ≥ 95% |
| Answer quality — LLM judge | ≥ 4.5 / 5.0 |
| Latency | ≤ 5s total, streamed, first token < 1s |
| Monthly API cost | ≤ $20 |
