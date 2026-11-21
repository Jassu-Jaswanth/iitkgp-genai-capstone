"""HR Policy Navigator API.

Run from backend/:  uv run uvicorn app.main:app --reload --port 8100
"""

import re
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .config import get_settings
from .llm import LLMClient, LLMError
from .prompts import build_messages
from .retrieval import Chunk, build_retriever


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list)
    k: int | None = Field(default=None, ge=1, le=20)
    filters: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int | None = Field(default=None, ge=16, le=1024)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    chunks: list[Chunk]
    grounded: bool
    retriever: str
    usage: Usage
    latency_ms: dict[str, float]


_CITE = re.compile(r"\[([^\[\]]{3,120})\]")


def extract_citations(answer: str, used: list[Chunk]) -> list[str]:
    """Citations the model wrote that match a retrieved chunk, exactly or as a finer clause of it
    (e.g. "5 U.S.C. 6303(a)(2)" inside chunk "5 U.S.C. 6303(a)"). Invented citations are dropped."""
    known = [c.citation for c in used]
    seen: list[str] = []
    for m in (x.strip() for x in _CITE.findall(answer)):
        if any(m == k or (m.startswith(k) and m[len(k):len(k) + 1] == "(") for k in known) and m not in seen:
            seen.append(m)
    return seen


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.settings = s
    app.state.llm = LLMClient(s)
    app.state.retriever = build_retriever(s.rag_service_url)
    yield
    await app.state.llm.aclose()
    if hasattr(app.state.retriever, "aclose"):
        await app.state.retriever.aclose()


app = FastAPI(title="HR Policy Navigator", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    st = request.app.state
    try:
        models = await st.llm.models()
        llm = {"ok": st.settings.llm_model in models, "models": models}
    except Exception as exc:  # report, don't crash the health check
        llm = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok" if llm["ok"] else "degraded", "llm": llm, "retriever": st.retriever.name}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    st = request.app.state
    s = st.settings
    t0 = time.perf_counter()

    chunks = await st.retriever.retrieve(req.question, k=req.k or s.retrieval_top_k, filters=req.filters)
    t_retrieve = (time.perf_counter() - t0) * 1000

    max_tokens = req.max_tokens or s.llm_default_max_tokens
    messages, used = build_messages(
        req.question, chunks, [t.model_dump() for t in req.history],
        context_tokens=s.llm_context_tokens, max_answer_tokens=max_tokens,
    )
    try:
        result = await st.llm.chat(messages, max_tokens=max_tokens)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"LLM endpoint error: {exc}") from exc

    answer = result.content or ""
    return ChatResponse(
        answer=answer,
        citations=extract_citations(answer, used),
        chunks=used,
        grounded=bool(used),
        retriever=st.retriever.name,
        usage=Usage(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens, llm_calls=1),
        latency_ms={"retrieve": round(t_retrieve, 1), "llm": result.latency_ms,
                    "total": round((time.perf_counter() - t0) * 1000, 1)},
    )
