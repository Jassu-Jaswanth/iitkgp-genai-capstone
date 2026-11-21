"""Retrieval interface. The RAG index itself is built by the RAG teammate; the server only CALLS it.

Contract (see .context/analysis/data-curation-architecture.md):
    retrieve(query, k, filters) -> list[Chunk]
Each chunk carries the clause IDs it contains, so answers cite clauses and eval can score recall@k.
"""

from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    text: str
    citation: str  # e.g. "5 U.S.C. 6303(a)" or "OPM fact sheet: annual-leave"
    clause_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Retriever(Protocol):
    name: str

    async def retrieve(self, query: str, k: int = 5, filters: dict[str, Any] | None = None) -> list[Chunk]: ...


class PlaceholderRetriever:
    """Returns nothing. Used until the RAG service is available.

    TODO(rag-integration): replace with HttpRetriever (set RAG_SERVICE_URL) or an in-process retriever
    from the RAG teammate's package, keeping the same retrieve() signature.
    """

    name = "placeholder"

    async def retrieve(self, query: str, k: int = 5, filters: dict[str, Any] | None = None) -> list[Chunk]:
        return []


class HttpRetriever:
    """Calls the RAG team's service: POST {base}/retrieve {"query", "k", "filters"} -> {"chunks": [Chunk...]}."""

    name = "http"

    def __init__(self, base_url: str, timeout_s: float = 30.0):
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s)

    async def retrieve(self, query: str, k: int = 5, filters: dict[str, Any] | None = None) -> list[Chunk]:
        r = await self._http.post("/retrieve", json={"query": query, "k": k, "filters": filters or {}})
        r.raise_for_status()
        return [Chunk(**c) for c in r.json().get("chunks", [])]

    async def aclose(self) -> None:
        await self._http.aclose()


def build_retriever(rag_service_url: str) -> Retriever:
    return HttpRetriever(rag_service_url) if rag_service_url else PlaceholderRetriever()
