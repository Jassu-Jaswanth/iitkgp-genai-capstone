"""Retrieval interface: one retrieve() contract, three interchangeable components (placeholder / http / inprocess).

Contract (see .context/analysis/data-curation-architecture.md):
    retrieve(query, k, filters) -> list[Chunk]
Each chunk carries the clause IDs it contains, so answers cite clauses and eval can score recall@k.

Retrieval runs through a LangChain BaseRetriever, so LangGraph's event stream (app/trace.py) records each search
with its query and every returned chunk (text, score, citation, clause ids).
"""

from typing import Any, Protocol

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ConfigDict, Field

from .cache import LRUCache, search_key


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


class PolicyRetriever(BaseRetriever):
    """LangChain retriever over the RAG service. Empty base_url -> placeholder that returns nothing
    (set RAG_SERVICE_URL to use the RAG service)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    base_url: str = ""
    timeout_s: float = 30.0
    k: int = 5
    cache: LRUCache = Field(default_factory=LRUCache)  # search cache (app/cache.py)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
                                k: int | None = None, filters: dict | None = None) -> list[Document]:
        if not self.base_url:
            return []
        key = search_key(query, k or self.k, filters)
        if (hit := self.cache.get(key)) is not None:
            return hit
        r = httpx.post(f"{self.base_url.rstrip('/')}/retrieve", timeout=self.timeout_s,
                       json={"query": query, "k": k or self.k, "filters": filters or {}})
        r.raise_for_status()
        docs = [chunk_to_doc(Chunk(**c)) for c in r.json().get("chunks", [])]
        self.cache.put(key, docs)
        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun,
                                       k: int | None = None, filters: dict | None = None) -> list[Document]:
        if not self.base_url:
            return []
        key = search_key(query, k or self.k, filters)
        if (hit := self.cache.get(key)) is not None:
            return hit
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.base_url.rstrip('/')}/retrieve",
                                  json={"query": query, "k": k or self.k, "filters": filters or {}})
        r.raise_for_status()
        docs = [chunk_to_doc(Chunk(**c)) for c in r.json().get("chunks", [])]
        self.cache.put(key, docs)
        return docs


class LangChainRetrieverAdapter:
    """Adapts any LangChain retriever to the server's retrieve() contract (keeps tests and callers simple)."""

    def __init__(self, retriever: BaseRetriever, name: str):
        self.lc, self.name = retriever, name

    def cache_stats(self) -> dict[str, int] | None:
        cache = getattr(self.lc, "cache", None)
        return cache.stats() if cache is not None else None

    async def retrieve(self, query: str, k: int = 5, filters: dict[str, Any] | None = None) -> list[Chunk]:
        docs = await self.lc.ainvoke(query, k=k, filters=filters or {})
        return [doc_to_chunk(d) for d in docs]


class LocalHybridRetriever(BaseRetriever):
    """rag/retriever.py's HybridRetriever (BM25 + dense + RRF + cross-encoder rerank) called in-process.
    A LangChain retriever, so the trace records a RETRIEVER span with every returned chunk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    engine: Any
    search_mode: str = "hybrid_rerank"
    k: int = 5
    cache: LRUCache = Field(default_factory=LRUCache)  # search cache (app/cache.py)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
                                k: int | None = None, filters: dict | None = None) -> list[Document]:
        key = search_key(query, k or self.k, filters)
        if (hit := self.cache.get(key)) is not None:
            return hit
        hits = self.engine.search(query, k=k or self.k, mode=self.search_mode, filters=filters or None)
        docs = [chunk_to_doc(Chunk(**h)) for h in hits]
        self.cache.put(key, docs)
        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun,
                                       k: int | None = None, filters: dict | None = None) -> list[Document]:
        from langchain_core.runnables.config import run_in_executor  # CPU work (BM25, rerank) off the event loop
        return await run_in_executor(None, self._get_relevant_documents, query, run_manager=run_manager.get_sync(),
                                     k=k, filters=filters)


def build_retriever(rag_service_url: str = "", mode: str = "", strategy: str = "clause-pack-200",
                    search_mode: str = "hybrid_rerank", cache_size: int = 512) -> Retriever:
    """Pick the retrieval component. All three satisfy the same retrieve() contract, so the agent can't tell them apart.
    - placeholder: returns nothing (agent still runs; answers "not covered")
    - http:        POST {rag_service_url}/retrieve (rag/service.py, or any service with the same contract)
    - inprocess:   rag/retriever.py called directly (no extra server)"""
    mode = mode or ("http" if rag_service_url else "placeholder")
    if mode == "inprocess":
        return LangChainRetrieverAdapter(LocalHybridRetriever(engine=_load_engine(strategy), search_mode=search_mode,
                                                              cache=LRUCache(cache_size)),
                                         f"inprocess:{strategy}/{search_mode}")
    if mode == "http" and not rag_service_url:
        raise ValueError("RETRIEVAL_MODE=http needs RAG_SERVICE_URL")
    return LangChainRetrieverAdapter(PolicyRetriever(base_url=rag_service_url if mode == "http" else "",
                                                      cache=LRUCache(cache_size)), mode)


def _load_engine(strategy: str):
    """Import rag/ and open its index (data/index/<strategy>); build the index first if it is missing."""
    import subprocess
    import sys
    from pathlib import Path
    rag_dir = Path(__file__).resolve().parents[2] / "rag"
    if str(rag_dir) not in sys.path:
        sys.path.insert(0, str(rag_dir))
    try:
        from retriever import INDEX, HybridRetriever, safe_name
    except ImportError as exc:
        raise RuntimeError("RETRIEVAL_MODE=inprocess needs the RAG libraries: cd backend && uv sync --extra rag") from exc
    if not (INDEX / safe_name(strategy) / "meta.json").exists():
        subprocess.run([sys.executable, str(rag_dir / "build_index.py"), "--strategy", strategy, "--embedder", "api"],
                       check=True, cwd=rag_dir)
    engine = HybridRetriever(strategy)
    engine.search("annual leave warm-up", k=1)  # load the embedder + reranker now, not on the first question
    return engine
