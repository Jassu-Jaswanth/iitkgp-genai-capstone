"""HR Policy Navigator API.

Run from backend/:  uv run uvicorn app.main:app --reload --port 8100

/chat   single-prompt RAG baseline (one model call), kept for the baseline-vs-agent comparison
/agent  LangGraph agent: planner -> tools -> critic, session memory, step trace
"""

import asyncio
import logging
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from openai import APIError
from pydantic import BaseModel, Field

from .agent import CRITIC_PROMPT, PLANNER_PROMPT, ROUTER_PROMPT, build_agent, summarize_turn
from .cache import answer_key, cacheable, version_hash
from .config import get_settings
from .llm import build_chat_model
from .prompts import DECOMPOSE_PROMPT, STRATEGIES, build_messages, parse_strategy_output, system_for
from .retrieval import Chunk, build_retriever
from .tools import build_tools
from .store import open_store
from .trace import TraceCollector


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
    use_cache: bool = Field(default=True, description="False forces a fresh answer (the eval runner sends False)")


class AgentResponse(BaseModel):
    session_id: str
    turn_id: str  # GET /turns/{turn_id}/trace returns the LLM trace of this answer
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
    cached_from: str | None = None  # turn_id of the original answer when served from the answer cache


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


def _usage(fallback: dict, steps: list[dict]) -> "Usage":
    """Token usage of the whole turn, counted from the trace (every model call, router included)."""
    llm = [x for x in steps if x["kind"] == "llm"]
    if not llm:
        return Usage(**fallback)
    return Usage(prompt_tokens=sum(x["prompt_tokens"] or 0 for x in llm),
                 completion_tokens=sum(x["completion_tokens"] or 0 for x in llm), llm_calls=len(llm))


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    st = app.state
    st.settings = s
    st.gate = asyncio.Semaphore(s.llm_max_concurrency)  # the key allows one active request
    st.llm = build_chat_model(s)
    st.retriever = build_retriever(s.rag_service_url, s.retrieval_mode, s.rag_strategy, s.rag_search_mode,
                                   cache_size=s.search_cache_size)
    st.tools = build_tools(st.retriever, s.data_path())
    st.store = await open_store(s.database_url, s.answer_cache_ttl_s)  # Postgres if DATABASE_URL, else in-process
    # Answer-cache version: any change to prompts, tools, retriever or model starts a fresh cache.
    st.cache_version = version_hash(PLANNER_PROMPT, ROUTER_PROMPT, CRITIC_PROMPT, st.retriever.name, s.llm_model,
                                    *[f"{t.name}:{t.description}" for t in st.tools])
    st.agent = build_agent(build_chat_model(s), build_chat_model(s, max_tokens=400), st.tools,
                           context_tokens=s.llm_context_tokens, max_answer_tokens=s.llm_default_max_tokens,
                           checkpointer=await st.store.checkpointer())
    yield
    await st.store.close()
    if hasattr(st.retriever, "aclose"):
        await st.retriever.aclose()


app = FastAPI(title="HR Policy Navigator", version="0.3.0", lifespan=lifespan)
log = logging.getLogger("hr_navigator")


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    st = request.app.state
    try:
        models = [m.id async for m in st.llm.root_async_client.models.list()]
        llm = {"ok": st.settings.llm_model in models, "models": models}
    except Exception as exc:  # report, don't crash the health check
        llm = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    stats = getattr(st.retriever, "cache_stats", lambda: None)()
    return {"status": "ok" if llm["ok"] else "degraded", "llm": llm, "retriever": st.retriever.name,
            "tools": [t.name for t in st.tools], "storage": "postgres" if st.settings.database_url else "memory",
            "cache": {"search": stats, "answer_ttl_s": st.settings.answer_cache_ttl_s}}


@app.delete("/cache")
async def clear_cache(request: Request) -> dict:
    """Empty both caches (e.g. after re-indexing the corpus)."""
    st = request.app.state
    cache = getattr(getattr(st.retriever, "lc", None), "cache", None)
    if cache is not None:
        cache._d.clear()
    return {"answers_cleared": await st.store.answer_cache.clear()}


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
    turn_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": session_id},
              "recursion_limit": s.agent_max_steps * 2 + 4}
    turn_input = {"messages": [HumanMessage(req.question)], "route": "hr_policy", "critic_rounds": 0,
                  "critique": [], "critic_usage": []}
    t0 = time.perf_counter()
    # Answer cache: only the first question of a conversation (follow-ups depend on what came before).
    use_cache = req.use_cache and s.answer_cache_ttl_s > 0
    key = None
    if use_cache and not (req.session_id and (await st.agent.aget_state(config)).values.get("messages")):
        key = answer_key(st.cache_version, req.question)
        if (hit := await st.store.answer_cache.get(key)) is not None:
            return await _from_cache(st, hit, session_id, turn_id, req.question, config, t0)
    status = "completed"
    collector = TraceCollector()
    async with st.gate:
        try:
            async for ev in st.agent.astream_events(turn_input, config, version="v2"):
                collector.add(ev)  # LangGraph's own events -> the LLM trace of this turn
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
    resp = AgentResponse(
        session_id=session_id, turn_id=turn_id, route=summary["route"], answer=answer, citations=citations,
        retrieved_clause_ids=list(dict.fromkeys(retrieved_ids)),
        retrieved_citations=[c for c in dict.fromkeys(retrieved_cites) if c],
        status=status, critique=vals.get("critique", []), tool_calls=summary["tool_calls"],
        tool_errors=summary["tool_errors"], trace=summary["trace"], usage=_usage(summary["usage"], collector.steps),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    try:
        await st.store.save_turn(session_id, req.question, resp.model_dump(), collector.steps)
        if key and cacheable(resp.model_dump()):
            await st.store.answer_cache.put(key, {"question": req.question, "response": resp.model_dump(),
                                                  "steps": collector.steps})
    except Exception:  # storage problems must not lose the answer
        log.exception("could not store turn %s", turn_id)
    return resp


async def _from_cache(st, hit: dict, session_id: str, turn_id: str, question: str, config: dict,
                      t0: float) -> AgentResponse:
    """Serve a cached answer as a new turn: same answer and trace, new ids, zero model calls."""
    orig = hit["response"]
    resp = AgentResponse(**{**orig, "session_id": session_id, "turn_id": turn_id,
                            "cached_from": orig.get("cached_from") or orig["turn_id"],
                            "usage": Usage(prompt_tokens=0, completion_tokens=0, llm_calls=0),
                            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
    # Put the exchange into the agent's memory so follow-up questions in this conversation have context.
    await st.agent.aupdate_state(config, {"messages": [
        HumanMessage(question), AIMessage(resp.answer, additional_kwargs={"route": resp.route})]}, as_node="critic")
    try:
        await st.store.save_turn(session_id, question, resp.model_dump(), hit.get("steps", []))
    except Exception:
        log.exception("could not store cached turn %s", turn_id)
    return resp


# ------------------------------------------------------------------ history and traces (for the dashboard)
@app.get("/conversations")
async def conversations(request: Request, limit: int = 50) -> list[dict]:
    return await request.app.state.store.conversations(limit)


@app.get("/conversations/{conversation_id}/messages")
async def conversation_messages(conversation_id: str, request: Request) -> list[dict]:
    return await request.app.state.store.messages(conversation_id)


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request) -> dict:
    st = request.app.state
    if not await st.store.delete(conversation_id):
        raise HTTPException(404, "no such conversation")
    await st.agent.checkpointer.adelete_thread(conversation_id)  # forget the agent memory too
    return {"deleted": conversation_id}


@app.get("/turns")
async def turns(request: Request, limit: int = 100) -> list[dict]:
    return await request.app.state.store.turns(limit)


@app.get("/turns/{turn_id}/trace")
async def turn_trace(turn_id: str, request: Request) -> list[dict]:
    return await request.app.state.store.trace(turn_id)


# ------------------------------------------------------------------ eval reports (files in eval/results/)
def _results_dir():
    from .config import REPO_ROOT
    return REPO_ROOT / "eval" / "results"


@app.get("/eval/runs")
async def eval_runs() -> list[dict]:
    """Every end-to-end eval run (eval/run_eval.py), newest first: its summary numbers."""
    out = []
    for p in sorted(_results_dir().glob("e2e_*.summary.json"), reverse=True):
        try:
            out.append({"run": p.name.removesuffix(".summary.json"), **json.loads(p.read_text())})
        except ValueError:
            continue
    return out


@app.get("/eval/runs/{run}")
async def eval_run(run: str) -> list[dict]:
    """Per-question rows of one run."""
    p = _results_dir() / f"{run}.jsonl"
    if "/" in run or ".." in run or not p.exists():
        raise HTTPException(404, "no such run")
    return [json.loads(line) for line in p.open() if line.strip()]


@app.get("/eval/retrieval")
async def eval_retrieval() -> list[dict]:
    """Retrieval experiments (rag/experiments.py): one CSV per run, newest first, rows as dicts."""
    import csv
    out = []
    for p in sorted(_results_dir().glob("retrieval_*.csv"), reverse=True):
        rows = []
        for r in csv.DictReader(p.open()):
            rows.append({k: (float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v) for k, v in r.items()})
        out.append({"run": p.stem, "rows": rows})
    return out
