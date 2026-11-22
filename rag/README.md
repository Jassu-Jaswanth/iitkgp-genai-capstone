# rag/: the RAG service ("the librarian")

Serves `POST /retrieve` for the agent's `search_policy` tool (contract: `backend/app/retrieval.py`).
Input is the team corpus **`data/corpus/clauses.jsonl`** (FedHR-Policy, 7,574 clauses; `docs/DATASET.md`).

## Models (upGrad-provided only, plus the reranker)
| Role | Model | Where it runs |
|---|---|---|
| Embeddings (index + queries) | `bge-small-en-v1.5`, 384-d | **upGrad embedding endpoint** (`EMBEDDING_*` in `.env`) |
| Embedding fallback | same `bge-small-en-v1.5` weights (fastembed ONNX) | local CPU, only if the endpoint fails at query time; vectors verified identical (cosine 1.00000) |
| Keyword search | BM25 (`rank-bm25`) | local, no model |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder (22M params) | local CPU; upGrad provides no reranking endpoint |
| LLM | none: the RAG service never calls Qwen3-8B, so it doesn't use the shared 1-request-at-a-time LLM key | |

## Decisions
| Date | Decision | Why |
|---|---|---|
| 21 Nov 2026 | **Reranker = local cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2`**, the only model not supplied by upGrad (decided by Rajesh) | The brief requires a reranker and upGrad offers no reranking endpoint. Alternatives rejected: Qwen3-8B as reranker (one extra call per search on the shared 1-request-at-a-time LLM key) and no reranker (breaks the requirement; −5–10 points recall). 22M params, ~0.5 s/query on CPU. |
| 21 Nov 2026 | Embeddings only via the upGrad endpoint (`bge-small-en-v1.5`); local copy of the same weights only as a query-time fallback | Use the upGrad-provided model; fallback keeps the demo up if the shared endpoint is busy (identical vectors). |
| 21 Nov 2026 | Corpus = team dataset `data/corpus/clauses.jsonl` (no separate curation in `rag/`) | One source of truth; 38/38 gold clause IDs resolve. |

## Pipeline
| Step | File | Output |
|---|---|---|
| 1. Chunk clauses (per strategy) | `chunking.py` | chunks with `clause_ids`, `[citation] heading` header |
| 2. Embed + save index | `build_index.py` | `data/index/<strategy>/{chunks.jsonl,dense.npy,meta.json}` (git-ignored) |
| 3. Search | `retriever.py` | BM25 top 30 + dense top 30 → RRF (k=60) → rerank pool 20 → top k |
| 4. Serve | `service.py` | `POST /retrieve`, `GET /health`; LRU cache (standalone use) |
| 5. Measure | `experiments.py` | `eval/results/retrieval_<ts>.csv` (+ per-query JSONL) |

Metrics: `sec@k` (gold section in top k), `cl@k` (gold clauses found), `fact@k` (top-k text contains every
`must_contain` fact: credits the same rule found in another source), `MRR`. Scored on `data/queries/gold_v1.jsonl`
(dev) and `heldout_v1.jsonl` once written.

## Run
```bash
cd rag && uv sync
.venv/Scripts/python build_index.py --strategy clause-pack-200          # Windows; Linux: .venv/bin/python
.venv/Scripts/python build_index.py --strategy all                      # all chunking strategies (experiments)
.venv/Scripts/python experiments.py
.venv/Scripts/python -m uvicorn service:app --port 8200                 # then RAG_SERVICE_URL=http://localhost:8200
```
Docker: `docker compose up --build` runs this search inside the backend container (`RETRIEVAL_MODE=inprocess`), so
there is no separate RAG container. If the index is missing, the backend builds it on first start through the
embedding endpoint (~2 min). `service.py` still serves the same search over HTTP for standalone use.

## Gotchas (Windows)
- An expired Hugging Face token stored locally turns public model downloads into 401s; the code sets
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.
- `huggingface_hub` symlink warnings on Windows are harmless (`HF_HUB_DISABLE_SYMLINKS_WARNING=1`).
