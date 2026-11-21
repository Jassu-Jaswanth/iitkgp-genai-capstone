"""Retrieval interface. The RAG index itself is built by the RAG teammate; the server only CALLS it.

Contract (see .context/analysis/data-curation-architecture.md):
    retrieve(query, k, filters) -> list[Chunk]
Each chunk carries the clause IDs it contains, so answers cite clauses and eval can score recall@k.

Retrieval runs through a LangChain BaseRetriever, so the trace (OpenInference) records a RETRIEVER span with
every returned chunk (text, score, citation, clause ids). The W3C trace context is forwarded to the RAG service,
so its internal steps (BM25, dense, RRF fusion, reranking) appear inside the same trace.
"""

from typing import Any, Protocol

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from opentelemetry import trace
from opentelemetry.propagate import inject
from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    chunk_id: str
    text: str
    citation: str  # e.g. "5 U.S.C. 6303(a)" or "OPM fact sheet: annual-leave"
    clause_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def chunk_to_doc(c: Chunk) -> Document:
    return Document(page_content=c.text, id=c.chunk_id,
                    metadata={"chunk_id": c.chunk_id, "citation": c.citation, "clause_ids": c.clause_ids,
                              "score": c.score, "source_url": c.source_url, **c.metadata})


def doc_to_chunk(d: Document) -> Chunk:
    m = dict(d.metadata)
    extra = {k: v for k, v in m.items() if k not in ("chunk_id", "citation", "clause_ids", "score", "source_url")}
    return Chunk(chunk_id=m.get("chunk_id") or d.id or "", text=d.page_content, citation=m.get("citation", ""),
                 clause_ids=m.get("clause_ids", []), score=float(m.get("score", 0.0)),
                 source_url=m.get("source_url"), metadata=extra)


class Retriever(Protocol):
    name: str

    async def retrieve(self, query: str, k: int = 5, filters: dict[str, Any] | None = None) -> list[Chunk]: ...


def _trace_headers() -> dict[str, str]:
    """W3C traceparent for the active OpenInference span (it isn't the OTel 'current' span, so look it up)."""
    headers: dict[str, str] = {}
    try:
        from openinference.instrumentation.langchain import get_current_span
        span = get_current_span()
    except Exception:  # tracing not installed/enabled
        span = None
    inject(headers, context=trace.set_span_in_context(span) if span else None)
    return headers


class PolicyRetriever(BaseRetriever):
    """LangChain retriever over the RAG service. Empty base_url -> placeholder that returns nothing
    (TODO(rag-integration): set RAG_SERVICE_URL once the RAG service is deployed)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    base_url: str = ""
    timeout_s: float = 30.0
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
                                k: int | None = None, filters: dict | None = None) -> list[Document]:
        if not self.base_url:
            return []
        headers = _trace_headers()  # W3C traceparent -> RAG service spans join this trace
        r = httpx.post(f"{self.base_url.rstrip('/')}/retrieve", headers=headers, timeout=self.timeout_s,
                       json={"query": query, "k": k or self.k, "filters": filters or {}})
        r.raise_for_status()
        return [chunk_to_doc(Chunk(**c)) for c in r.json().get("chunks", [])]

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun,
                                       k: int | None = None, filters: dict | None = None) -> list[Document]:
        if not self.base_url:
            return []
        headers = _trace_headers()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.base_url.rstrip('/')}/retrieve", headers=headers,
                                  json={"query": query, "k": k or self.k, "filters": filters or {}})
        r.raise_for_status()
        return [chunk_to_doc(Chunk(**c)) for c in r.json().get("chunks", [])]


class LangChainRetrieverAdapter:
    """Adapts any LangChain retriever to the server's retrieve() contract (keeps tests and callers simple)."""

    def __init__(self, retriever: BaseRetriever, name: str):
        self.lc, self.name = retriever, name

    async def retrieve(self, query: str, k: int = 5, filters: dict[str, Any] | None = None) -> list[Chunk]:
        docs = await self.lc.ainvoke(query, k=k, filters=filters or {})
        return [doc_to_chunk(d) for d in docs]


def build_retriever(rag_service_url: str) -> Retriever:
    name = "http" if rag_service_url else "placeholder"
    return LangChainRetrieverAdapter(PolicyRetriever(base_url=rag_service_url), name)
