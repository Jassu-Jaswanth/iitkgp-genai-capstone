# Backend: HR Policy Navigator API

FastAPI + LangChain + LangGraph. Calls the upGrad Track B model (Qwen3-8B, OpenAI-compatible vLLM) and the
RAG service built by the RAG teammate. The server never builds the index; it only calls `retrieve()`.

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8100   # reads ../.env
uv run pytest                                       # unit tests (scripted fake models, no network)
uv run pytest -m live                               # real calls on the shared key; run sparingly
```

| Endpoint | What it does |
|---|---|
| `GET /health` | Model endpoint check, active retriever, tool list |
| `POST /chat` | Single-prompt RAG baseline (one model call): answer, citations, chunks, usage, latency |
| `POST /agent` | LangGraph agent `{question, session_id?}`: answer, citations, status, critique, tool calls, trace, usage, latency |

## Agent (app/agent.py)
`planner` (ChatOpenAI + bound tools) → `tools` (LangGraph ToolNode; errors become observations) → back to
planner → `critic` (structured `issues` list; verdict derived in code) → END, or one revision round.
Session memory: LangGraph checkpointer keyed by `session_id`. Step cap: `AGENT_MAX_STEPS` (default 10) via
`recursion_limit`. Context budget: `trim_messages`.

Tools (app/tools.py): `search_policy` (calls the retriever), `leave_accrual_calc`, `pay_lookup` (OPM 2026 GS
table), `holiday_calendar`, `escalate_to_hr` (draft only; human approval boundary). Tool data is read from
`DATA_DIR` (default `data/raw`); missing files come back as tool errors, not crashes.

## Integration point for the RAG service
Set `RAG_SERVICE_URL` in `.env`. The server sends `POST {url}/retrieve` with `{"query", "k", "filters"}` and
expects `{"chunks": [{"chunk_id", "text", "citation", "clause_ids", "score", "source_url"?, "metadata"?}]}`.

## Endpoint rules enforced
One request at a time per key (process semaphore), 60 req/min (LangChain rate limiter), SDK retries with
backoff on 429/5xx, `max_tokens` capped at 1024, Qwen3 thinking disabled.
