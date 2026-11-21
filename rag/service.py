"""RAG service for the HR Policy Navigator backend.

    cd rag && .venv/Scripts/python -m uvicorn service:app --port 8200     (Windows)
    cd rag && .venv/bin/python -m uvicorn service:app --port 8200         (Linux/WSL/Docker)
Then set RAG_SERVICE_URL=http://localhost:8200 in the repo .env.

Contract (backend/app/retrieval.py):
    POST /retrieve {"query", "k", "filters"} -> {"chunks": [{chunk_id, text, citation, clause_ids, score, source_url, metadata}]}

Input: data/corpus/clauses.jsonl (docs/DATASET.md). If the index folder is missing, it is built on startup with the
upGrad embedding endpoint (~2 min), so `docker compose up` works on a fresh clone.

Env: RAG_STRATEGY (index/chunking, default clause-pack-200: best on gold_v1), RAG_MODE (hybrid_rerank | hybrid | bm25 | dense),
     RAG_CANDIDATES (30 per retriever), RAG_RERANK_POOL (20), EMBEDDER (api default | local),
     PHOENIX_COLLECTOR_ENDPOINT (optional: export spans; joins the backend trace via the traceparent header).
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager, nullcontext
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from retriever import INDEX, HybridRetriever, safe_name

STRATEGY = os.environ.get("RAG_STRATEGY", "clause-pack-200")
MODE = os.environ.get("RAG_MODE", "hybrid_rerank")
CANDIDATES = int(os.environ.get("RAG_CANDIDATES", "30"))
RERANK_POOL = int(os.environ.get("RAG_RERANK_POOL", "20"))
CACHE_SIZE = int(os.environ.get("RAG_CACHE_SIZE", "512"))

# ---------------------------------------------------------------- optional tracing
tracer = None
if os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
    try:
        from opentelemetry import trace
        from opentelemetry.propagate import extract
        from phoenix.otel import register
        provider = register(project_name=os.environ.get("PHOENIX_PROJECT_NAME", "hr-navigator"),
                            set_global_tracer_provider=False)
        tracer = provider.get_tracer("rag_service")
    except Exception as exc:  # tracing must never break retrieval
        print(f"tracing disabled: {exc}")


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] = Field(default_factory=dict)


class ChunkOut(BaseModel):
    chunk_id: str
    text: str
    citation: str
    clause_ids: list[str]
    score: float
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    chunks: list[ChunkOut]
    timings: dict[str, Any] = Field(default_factory=dict)


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    t0 = time.perf_counter()
    if not (INDEX / safe_name(STRATEGY) / "meta.json").exists():
        print(f"index {STRATEGY} missing -> building it from data/corpus/clauses.jsonl", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name("build_index.py")), "--strategy", STRATEGY,
                        "--embedder", os.environ.get("EMBEDDER", "api")], check=True)
    r = HybridRetriever(STRATEGY)
    r.search("annual leave warm-up", k=1, mode=MODE)  # load embedder + reranker once
    state.update(retriever=r, load_s=round(time.perf_counter() - t0, 1))
    yield


app = FastAPI(title="HR Navigator RAG service", lifespan=lifespan)


@lru_cache(maxsize=CACHE_SIZE)
def _cached(query: str, k: int, filters_key: tuple) -> tuple[tuple, dict]:
    timings: dict = {}
    res = state["retriever"].search(query, k=k, mode=MODE, n_candidates=CANDIDATES, rerank_pool=RERANK_POOL,
                                    filters=dict(filters_key) or None, timings=timings)
    return tuple(res), timings


@app.get("/health")
def health():
    r: HybridRetriever | None = state.get("retriever")
    return {"status": "ok" if r else "loading", "strategy": STRATEGY, "mode": MODE,
            "chunks": len(r.chunks) if r else 0, "embedder": r.meta.get("embedder") if r else None,
            "load_s": state.get("load_s"), "cache": _cached.cache_info()._asdict()}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, request: Request):
    if "retriever" not in state:
        raise HTTPException(503, "index loading")
    fkey = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in req.filters.items()))
    span_cm = nullcontext()
    if tracer is not None:
        ctx = extract(dict(request.headers))  # W3C traceparent from the backend -> same trace
        span_cm = tracer.start_as_current_span(f"rag_service.{MODE}", context=ctx)
    with span_cm as span:
        hits = _cached.cache_info().hits
        res, timings = _cached(req.query.strip(), req.k, fkey)
        timings = {**timings, "cache_hit": _cached.cache_info().hits > hits}
        if span is not None:
            span.set_attribute("openinference.span.kind", "RETRIEVER")
            span.set_attribute("input.value", req.query)
            for key, v in timings.items():
                span.set_attribute(f"rag.{key}", v)
            for i, c in enumerate(res):
                span.set_attribute(f"retrieval.documents.{i}.document.id", c["chunk_id"])
                span.set_attribute(f"retrieval.documents.{i}.document.score", c["score"])
                span.set_attribute(f"retrieval.documents.{i}.document.content", c["text"])
    return {"chunks": list(res), "timings": timings}
