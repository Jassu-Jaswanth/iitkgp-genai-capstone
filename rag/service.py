"""RAG service implementing the backend's retrieval contract.

    POST /retrieve {"query", "k", "filters": {"source_type"?: "statute|regulation|guidance"}}
      -> {"chunks": [{"chunk_id", "text", "citation", "clause_ids", "score", "source_url", "metadata"}]}

Run from the repo root:
    RAG_CHUNKING=clause-pack-350 RAG_RETRIEVER=hybrid+rerank rag/.venv/bin/uvicorn service:app --app-dir rag --port 8200
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from pipeline import (QueryInstructionEmbeddings, build_faiss, chunk_strategies, dedupe_by_chunk, load_sections,
                      retrievers_for, upgrad_embeddings)

CHUNKING = os.getenv("RAG_CHUNKING", "clause-pack-350")
RETRIEVER = os.getenv("RAG_RETRIEVER", "hybrid+rerank")
QUERY_INSTRUCTION = os.getenv("RAG_QUERY_INSTRUCTION", "1") == "1"
RERANKER = os.getenv("RAG_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MAX_K = 10


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=MAX_K)
    filters: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    sections = load_sections()
    docs = chunk_strategies()[CHUNKING](sections)
    base = upgrad_embeddings()
    vs = build_faiss(docs, base, "upgrad-bge-small__" + CHUNKING.replace("/", "_"))
    vs.embedding_function = QueryInstructionEmbeddings(base) if QUERY_INSTRUCTION else base
    app.state.retriever = retrievers_for(docs, vs, MAX_K, reranker_model=RERANKER)[RETRIEVER]
    app.state.info = {"chunking": CHUNKING, "retriever": RETRIEVER, "query_instruction": QUERY_INSTRUCTION,
                      "reranker": RERANKER if "rerank" in RETRIEVER else None, "chunks": len(docs),
                      "sections": len(sections), "startup_s": round(time.perf_counter() - t0, 1)}
    yield


app = FastAPI(title="HR Policy Navigator RAG service", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", **request.app.state.info}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest, request: Request) -> dict:
    allowed = req.filters.get("source_type")
    if allowed and allowed not in ("statute", "regulation", "guidance"):
        raise HTTPException(422, "filters.source_type must be statute, regulation or guidance")
    docs = request.app.state.retriever.invoke(req.query)
    if allowed:
        docs = [d for d in docs if d.metadata["source_type"] == allowed]
    docs = dedupe_by_chunk(docs, req.k)
    return {"chunks": [{"chunk_id": d.metadata["chunk_id"], "text": d.page_content, "citation": d.metadata["citation"],
                        "clause_ids": d.metadata["clause_ids"], "score": round(1.0 / rank, 4),  # rank-based score
                        "source_url": d.metadata.get("url"),
                        "metadata": {"section_id": d.metadata["section_id"], "heading": d.metadata["heading"],
                                     "source_type": d.metadata["source_type"], "rank": rank}}
                       for rank, d in enumerate(docs, 1)]}
