# Backend: HR Policy Navigator API

FastAPI + LangChain + LangGraph. Calls the upGrad Track B model (Qwen3-8B, OpenAI-compatible vLLM) and searches
the policy corpus with `rag/retriever.py`, inside the same process.

```bash
cd backend
uv sync --extra rag
uv run uvicorn app.main:app --reload --port 8100   # reads ../.env
uv run pytest                                       # unit tests (scripted fake models, no network)
uv run pytest -m live                               # real calls on the shared key; run sparingly
```

| Endpoint | What it does |
|---|---|
| `GET /health` | Model endpoint check, active retriever, tool list, storage (postgres / memory) |
| `POST /chat` | Single-prompt RAG baseline (one model call): answer, citations, chunks, usage, latency |
| `POST /agent` | LangGraph agent `{question, session_id?}`: answer, citations, status, critique, tool calls, trace, usage, latency, `turn_id` |
| `GET /turns/{turn_id}/trace` | The LLM trace of one answer: every model call, tool call and search, with JSON input/output, tokens, times |
| `GET /turns` | Recent answers with route, latency, tokens, tools (the Traces page) |
| `GET /conversations`, `GET /conversations/{id}/messages`, `DELETE /conversations/{id}` | Chat history |
| `GET /eval/runs`, `GET /eval/runs/{run}`, `GET /eval/retrieval` | Eval reports from `eval/results/` (the Evaluation and Retrieval pages) |
| `DELETE /cache` | Empty both caches (e.g. after re-indexing the corpus) |

`/agent` also takes `use_cache` (default true) and returns `cached_from` (the original `turn_id` when the answer came
from the cache). The eval runner sends `use_cache: false`.

## Trace and storage (app/trace.py, app/store.py, app/schema.sql)
The trace comes straight from LangGraph: `/agent` runs the graph with `astream_events(version="v2")` and keeps the
start/end events of model calls, tool calls and retrievals (no tracing server). With `DATABASE_URL` set (compose
does it) everything goes to Postgres: schema `chat` (conversations, messages, trace_steps) and schema `agent`
(LangGraph's `AsyncPostgresSaver`, so conversations keep their memory across restarts). Without it, an in-process
store keeps the last 500 answers.

## Caching (app/cache.py)
On by default, nothing to set up.
- **Search cache:** identical searches (case/spacing ignored) reuse the chunks, in memory. `SEARCH_CACHE_SIZE` (512, 0 = off).
- **Answer cache:** the first question of a new conversation, asked again, is answered from Postgres
  (`chat.answer_cache`) with zero model calls, plus the trace of the original answer. `ANSWER_CACHE_TTL_S`
  (86400 = one day, 0 = off). Follow-ups, HR escalations and failed answers are never cached. The cache key includes
  a hash of the prompts, tools, retriever and model, so changing any of them starts a fresh cache.

## Agent (app/agent.py)
`planner` (ChatOpenAI + bound tools) → `tools` (LangGraph ToolNode; errors become observations) → back to
planner → `critic` (structured `issues` list; verdict derived in code) → END, or one revision round.
Session memory: LangGraph checkpointer keyed by `session_id`. Step cap: `AGENT_MAX_STEPS` (default 10) via
`recursion_limit`. Context budget: `trim_messages`.

Tools (app/tools.py): `search_policy` (calls the retriever), `leave_accrual_calc`, `pay_lookup` (OPM 2026 GS
table), `holiday_calendar`, `escalate_to_hr` (draft only; human approval boundary). Tool data is read from
`DATA_DIR` (default `data`); missing files come back as tool errors, not crashes.

## Retrieval (`RETRIEVAL_MODE`)
`inprocess` (compose default): `rag/retriever.py` (hybrid BM25 + bge-small + cross-encoder rerank) runs inside the
backend; needs `uv sync --extra rag`; builds `data/index/<RAG_STRATEGY>` on first start if missing.
`http`: the same search as a separate service (`rag/service.py`, `RAG_SERVICE_URL=http://localhost:8200`).
`placeholder`: no search (the agent still runs). Over HTTP the server sends `POST {url}/retrieve` with `{"query", "k", "filters"}` and
expects `{"chunks": [{"chunk_id", "text", "citation", "clause_ids", "score", "source_url"?, "metadata"?}]}`.

## Rate limits
Model endpoint: one request at a time per key (process semaphore), 60 req/min (LangChain rate limiter), SDK retries
with backoff on 429/5xx, `max_tokens` capped at 1024, Qwen3 thinking disabled.
Users: the frontend's nginx allows 10 questions a minute per IP (burst 5) and answers 429 beyond that.
