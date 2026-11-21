# backend/: the agent

| File | Job |
|---|---|
| `app/agent.py` | the LangGraph loop: router → planner ⇄ tools → critic |
| `app/tools.py` | tools (1 example: `search_policy`; the rest are `TODO(tools)`) |
| `app/retrieval.py` | the plug for the RAG service (`retrieve()`); placeholder until `RAG_SERVICE_URL` is set |
| `app/llm.py` | the Qwen3-8B client (retries, rate limit) |
| `app/prompts.py` | prompts for the single-call `/chat` baseline (4 strategies) |
| `app/tracing.py` | sends traces to Phoenix when `PHOENIX_COLLECTOR_ENDPOINT` is set |
| `app/main.py` | API: `GET /health`, `POST /agent`, `POST /chat` |

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8100   # reads ../.env
uv run pytest                                       # offline tests
uv run pytest -m live                               # 1 real call to the shared model
```
`POST /agent {"question": "...", "session_id": "optional, to continue a chat"}` returns: the answer, citations,
route, critic notes, tool calls, a step trace, tokens and latency.
