# HR Policy Navigator (IIT KGP Gen AI capstone, Track B)

**What it is:** a chat assistant that answers US federal HR questions (leave, pay, holidays) from the real rules,
cites the rule it used, and says "not covered" instead of guessing. It runs on the course's hosted
Qwen3-8B model, so all the cleverness is in the agent, the tools and the retrieval.

**Key dates:** submit on the portal 25–27 Nov 2026 (no extensions) · presentation 5 Dec 2026.

## How it works (one picture)
```
question → router → planner (LLM) ⇄ tools → critic → answer + [citations]
                                   │
                                   └─ search_policy → retrieve() → RAG service (TODO)
```
- **router:** blocks private or off-topic questions early.
- **planner:** decides which tool to use next.
- **tools:** do the lookups and maths.
- **critic:** checks the answer against what the tools returned, and sends it back once if wrong.
- **Phoenix:** shows every step as a trace.

**Today on main:** the loop works end to end with one example tool. With no RAG service it answers "not covered".
What's left: [docs/TODO.md](docs/TODO.md).

## Run it (5 minutes)
1. **Get the secrets** from the team rep: the two API keys, plus two CA certificates.
   ```bash
   cp .env.example .env                 # paste LLM_API_KEY and EMBEDDING_API_KEY
   mkdir -p .context/certs              # put llm-vm04-ca.crt and embedding-vm02-ca.crt here
   python scripts/check_endpoints.py    # should print "LLM OK" and "Embeddings OK"
   ```
2. **Start everything** (Docker):
   ```bash
   docker compose up --build
   ```
   - Agent API: http://localhost:8100/docs
   - Trace dashboard: http://localhost:6006
3. **Ask something:**
   ```bash
   curl -s localhost:8100/agent -H 'content-type: application/json' \
        -d '{"question": "How much annual leave can I carry over?"}'
   ```
   Then open Phoenix to watch what the agent did.

**Without Docker:** `cd backend && uv sync && uv run uvicorn app.main:app --port 8100`.
**Tests:** `cd backend && uv run pytest` (no network needed).

## Where things are
| Folder | What |
|---|---|
| `backend/` | the agent (LangGraph), tools, API, tracing |
| `eval/` | the golden set (50 verified Q&A), held-out template, grading script: [eval/README.md](eval/README.md) |
| `data/processed/` | the curated HR rules for RAG + tables for tools: [docs/DATASET.md](docs/DATASET.md) |
| `docs/` | **TODO tracker**, dataset guide, course brief and upGrad notes |
| `scripts/` | endpoint check |

## Two rules
- **The model key allows one request at a time for the whole team.** Don't run big evals at the same time as someone else.
- **Never commit** `.env`, keys or certificates. `main` takes pull requests with one approval.
