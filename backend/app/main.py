"""HR Policy Navigator API.

Run from backend/:  uv run uvicorn app.main:app --reload --port 8100

/chat   single-prompt RAG baseline (one model call), kept for the baseline-vs-agent comparison
/agent  LangGraph agent: planner -> tools -> critic, session memory, step trace
"""

import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from openai import APIError
from pydantic import BaseModel, Field

from .agent import build_agent, summarize_turn
from .config import get_settings
from .llm import build_chat_model
from .prompts import DECOMPOSE_PROMPT, STRATEGIES, build_messages, parse_strategy_output, system_for
from .retrieval import Chunk, build_retriever
from .tools import build_tools
from .tracing import setup_tracing


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list)
    k: int | None = Field(default=None, ge=1, le=20)
    filters: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int | None = Field(default=None, ge=16, le=1024)
    strategy: str = Field(default="grounded", description="grounded | few_shot_json | cot | decompose")


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int


class ChatResponse(BaseModel):
    answer: str
    strategy: str
    sub_queries: list[str] = Field(default_factory=list)
    citations: list[str]
    chunks: list[Chunk]
    grounded: bool
    retriever: str
    usage: Usage
    latency_ms: dict[str, float]


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="Reuse to continue a conversation")


class AgentResponse(BaseModel):
    session_id: str
    route: str
    answer: str
    citations: list[str]
    status: str  # completed | step_limit | llm_error
    retrieved_clause_ids: list[str] = Field(default_factory=list)
    retrieved_citations: list[str] = Field(default_factory=list)
    critique: list[str]
    tool_calls: int
    tool_errors: int
    trace: list[dict[str, Any]]
    usage: Usage
    latency_ms: float


_CITE = re.compile(r"\[([^\[\]]{3,120})\]")


def extract_citations(answer: str, known: list[str]) -> list[str]:
    """Citations the model wrote that match a known source, exactly or as a finer clause of it
    (e.g. "5 U.S.C. 6303(a)(2)" inside "5 U.S.C. 6303(a)"). Invented citations are dropped."""
    seen: list[str] = []
    for m in (x.strip() for x in _CITE.findall(answer)):
        if any(m == k or (m.startswith(k) and m[len(k):len(k) + 1] == "(") for k in known) and m not in seen:
            seen.append(m)
    return seen


def _known_citations_from_trace(trace: list[dict[str, Any]]) -> list[str]:
    """Every citation value returned by successful tool calls (tool results are JSON)."""
    found: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "citation" in k and isinstance(v, str):
                    found.extend(part.strip() for part in v.split(";"))
                else:
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    for t in trace:
        if t["type"] == "tool_result" and t["status"] != "error":
            try:
                walk(json.loads(t["full_content"]))
            except (ValueError, KeyError):
                pass
    return list(dict.fromkeys(found))


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    st = app.state
    st.settings = s
    st.tracing = setup_tracing()
    st.gate = asyncio.Semaphore(s.llm_max_concurrency)  # the key allows one active request
    st.llm = build_chat_model(s)
    st.retriever = build_retriever(s.rag_service_url)
    st.tools = build_tools(st.retriever, s.data_path())
    st.agent = build_agent(build_chat_model(s), build_chat_model(s, max_tokens=400), st.tools,
                           context_tokens=s.llm_context_tokens, max_answer_tokens=s.llm_default_max_tokens)
    yield
    if hasattr(st.retriever, "aclose"):
        await st.retriever.aclose()


app = FastAPI(title="HR Policy Navigator", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    st = request.app.state
    try:
        models = [m.id async for m in st.llm.root_async_client.models.list()]
        llm = {"ok": st.settings.llm_model in models, "models": models}
    except Exception as exc:  # report, don't crash the health check
        llm = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok" if llm["ok"] else "degraded", "llm": llm, "retriever": st.retriever.name,
            "tools": [t.name for t in st.tools], "tracing": st.tracing}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """Single-prompt RAG baseline with selectable prompting strategy (Track B part II)."""
    st = request.app.state
    s = st.settings
    if req.strategy not in STRATEGIES:
        raise HTTPException(422, f"strategy must be one of {STRATEGIES}")
    t0 = time.perf_counter()
    k = req.k or s.retrieval_top_k
    pt = ct = calls = 0
    sub_queries: list[str] = []
    try:
        if req.strategy == "decompose":
            async with st.gate:
                d = await st.llm.bind(max_tokens=200, response_format={"type": "json_object"}).ainvoke(
                    [{"role": "system", "content": DECOMPOSE_PROMPT}, {"role": "user", "content": req.question}])
            um = d.usage_metadata or {}
            pt, ct, calls = um.get("input_tokens", 0), um.get("output_tokens", 0), 1
            try:
                sub_queries = [q for q in json.loads(d.content).get("queries", []) if isinstance(q, str)][:3]
            except (ValueError, AttributeError):
                sub_queries = []
            sub_queries = sub_queries or [req.question]
            seen, chunks = set(), []
            per = max(2, k // len(sub_queries) + 1)
            for q in sub_queries:
                for c in await st.retriever.retrieve(q, k=per, filters=req.filters):
                    if c.chunk_id not in seen:
                        seen.add(c.chunk_id)
                        chunks.append(c)
        else:
            chunks = await st.retriever.retrieve(req.question, k=k, filters=req.filters)
        t_retrieve = (time.perf_counter() - t0) * 1000

        max_tokens = req.max_tokens or (768 if req.strategy == "cot" else s.llm_default_max_tokens)
        messages, used = build_messages(req.question, chunks, [t.model_dump() for t in req.history],
                                        context_tokens=s.llm_context_tokens, max_answer_tokens=max_tokens,
                                        system_prompt=system_for(req.strategy))
        t1 = time.perf_counter()
        model = st.llm.bind(max_tokens=max_tokens)
        if req.strategy == "few_shot_json":
            model = st.llm.bind(max_tokens=max_tokens, response_format={"type": "json_object"})
        async with st.gate:
            msg = await model.ainvoke(messages)
    except APIError as exc:
        raise HTTPException(status_code=502, detail=f"LLM endpoint error: {exc}") from exc
    um = msg.usage_metadata or {}
    answer = parse_strategy_output(req.strategy, str(msg.content or ""))
    return ChatResponse(
        answer=answer, strategy=req.strategy, sub_queries=sub_queries,
        citations=extract_citations(answer, [c.citation for c in used]), chunks=used,
        grounded=bool(used), retriever=st.retriever.name,
        usage=Usage(prompt_tokens=pt + um.get("input_tokens", 0), completion_tokens=ct + um.get("output_tokens", 0),
                    llm_calls=calls + 1),
        latency_ms={"retrieve": round(t_retrieve, 1), "llm": round((time.perf_counter() - t1) * 1000, 1),
                    "total": round((time.perf_counter() - t0) * 1000, 1)},
    )


@app.post("/agent", response_model=AgentResponse)
async def agent(req: AgentRequest, request: Request) -> AgentResponse:
    st = request.app.state
    s = st.settings
    session_id = req.session_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": s.agent_max_steps * 2 + 4}
    turn_input = {"messages": [HumanMessage(req.question)], "route": "hr_policy", "critic_rounds": 0,
                  "critique": [], "critic_usage": []}
    t0 = time.perf_counter()
    status = "completed"
    async with st.gate:
        try:
            await st.agent.ainvoke(turn_input, config)
        except GraphRecursionError:
            status = "step_limit"
        except APIError as exc:
            status = "llm_error"
            error = str(exc)
    snap = await st.agent.aget_state(config)
    vals = snap.values
    summary = summarize_turn(vals.get("messages", []), vals.get("critic_usage", []))
    answer = summary["answer"]
    if status == "step_limit" and not answer:
        answer = f"I could not finish within the {s.agent_max_steps}-step limit. Please narrow the question."
    elif status == "llm_error":
        answer = f"The model endpoint failed: {error}"
    citations = extract_citations(answer, _known_citations_from_trace(summary["trace"]))
    retrieved_ids: list[str] = []
    retrieved_cites: list[str] = []
    for t in summary["trace"]:
        if t["type"] == "tool_result" and t.get("tool") == "search_policy" and t["status"] != "error":
            try:
                for r in json.loads(t["full_content"]).get("results", []):
                    retrieved_ids += r.get("clause_ids", [])
                    retrieved_cites.append(r.get("citation", ""))
            except (ValueError, KeyError):
                pass
        t.pop("full_content", None)
    return AgentResponse(
        session_id=session_id, route=summary["route"], answer=answer, citations=citations,
        retrieved_clause_ids=list(dict.fromkeys(retrieved_ids)),
        retrieved_citations=[c for c in dict.fromkeys(retrieved_cites) if c],
        status=status, critique=vals.get("critique", []), tool_calls=summary["tool_calls"],
        tool_errors=summary["tool_errors"], trace=summary["trace"], usage=Usage(**summary["usage"]),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
