# Backend: HR Policy Navigator API

FastAPI service that calls the upGrad Track B model endpoint (Qwen3-8B, OpenAI-compatible) and the RAG
service built by the RAG teammate. The server never builds the index; it only calls `retrieve()`.

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8100   # reads ../.env
uv run pytest                                       # unit tests (no network)
uv run pytest -m live                               # one real call to the shared endpoint
```

| Endpoint | What it does |
|---|---|
| `GET /health` | Checks the model endpoint and reports which retriever is active |
| `POST /chat` | `{question, history?, k?, filters?, max_tokens?}` → answer, citations, chunks used, usage, latency |

## Integration point for the RAG service
Set `RAG_SERVICE_URL` in `.env`. The server sends `POST {url}/retrieve` with `{"query", "k", "filters"}` and
expects `{"chunks": [{"chunk_id", "text", "citation", "clause_ids", "score", "source_url"?, "metadata"?}]}`.
With no URL the placeholder retriever returns nothing and the model answers
"The provided policy documents do not cover this."

## Endpoint rules this client enforces
- One request at a time per key (a second concurrent call gets HTTP 429), so calls are queued in-process.
- 429/503 retried with 2, 4, 8 s backoff; `max_tokens` capped at 1024; Qwen3 thinking disabled.
