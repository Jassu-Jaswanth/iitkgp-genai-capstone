"""RAG service implementing the backend's retrieval contract.

    POST /retrieve {"query", "k", "filters": {"source_type"?: "statute|regulation|guidance"}}
      -> {"chunks": [{"chunk_id", "text", "citation", "clause_ids", "score", "source_url", "metadata"}]}

Run from the repo root:
    RAG_CHUNKING=token-256/32+header RAG_RETRIEVER=hybrid+rerank rag/.venv/bin/uvicorn service:app --app-dir rag --port 8200
"""

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from opentelemetry import context as otel_context
from opentelemetry.propagate import extract
from pydantic import BaseModel, Field

from pipeline import build_policy_search

CHUNKING = os.getenv("RAG_CHUNKING", "token-256/32+header")
RETRIEVER = os.getenv("RAG_RETRIEVER", "hybrid+rerank")
QUERY_INSTRUCTION = os.getenv("RAG_QUERY_INSTRUCTION", "0") == "1"
RERANKER = os.getenv("RAG_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MAX_K = 10


def setup_tracing() -> str | None:
    """Same Phoenix project as the backend; spans join the caller's trace via the W3C traceparent header.
    Not registered globally, so FastAPI's own HTTP spans stay out of the trace."""
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return None
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register
    provider = register(endpoint=endpoint, project_name=os.getenv("PHOENIX_PROJECT_NAME", "hr-policy-navigator"),
                        batch=True, set_global_tracer_provider=False, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return endpoint


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=MAX_K)
    filters: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracing = setup_tracing()
    app.state.search, info = build_policy_search(CHUNKING, RETRIEVER, query_instruction=QUERY_INSTRUCTION,
                                                 reranker=RERANKER, max_k=MAX_K)
    app.state.info = {**info, "tracing": tracing}
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
    token = otel_context.attach(extract(dict(request.headers)))  # continue the backend's trace
    try:
        docs = request.app.state.search.invoke(req.query, k=req.k, filters=req.filters,
                                               config={"run_name": "rag_service.search"})
    finally:
        otel_context.detach(token)
    return {"chunks": [{"chunk_id": d.metadata["chunk_id"], "text": d.page_content, "citation": d.metadata["citation"],
                        "clause_ids": d.metadata["clause_ids"], "score": d.metadata["score"],
                        "source_url": d.metadata["source_url"],
                        "metadata": {k: d.metadata[k] for k in ("section_id", "heading", "source_type", "rank")}}
                       for d in docs]}
