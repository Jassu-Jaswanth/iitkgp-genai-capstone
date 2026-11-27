# HR Policy Navigator (IIT KGP Gen AI capstone, Track B)

**What it is:** a chat assistant that answers US federal HR questions (leave, pay, holidays) from the real rules,
cites the rule it used, and says "not covered" instead of guessing. It runs on the course's hosted
Qwen3-8B model, so all the cleverness is in the agent, the tools and the retrieval.

**Key dates:** submit on the portal 25–27 Nov 2026 (no extensions) · presentation 5 Dec 2026.

## How it works (one picture)
```
question → router → planner (LLM) ⇄ tools → critic → answer + [citations]
                                   │
                                   ├─ search_policy → hybrid search (BM25 + embeddings + rerank) over the corpus
                                   └─ leave_accrual_calc · pay_lookup · holiday_calendar · escalate_to_hr
```
- **router:** blocks private or off-topic questions early.
- **planner:** decides which tool to use next.
- **tools:** do the lookups and maths (the 8B model is bad at arithmetic, so it never does it itself).
- **critic:** checks the answer against what the tools returned, and sends it back once if wrong.
- **trace:** every step (model calls, tool calls, searches) is saved in Postgres and shown next to the chat.
- **cache:** a question asked before is answered instantly; repeated searches are reused. Nothing to set up.

Project status and future work: [docs/STATUS.md](docs/STATUS.md). Final report and demo video: [report/](report/).

## Run it (5 minutes)
1. **Get the secrets** from the team rep: the two API keys, plus two CA certificates.
   ```bash
   cp .env.example .env                 # paste LLM_API_KEY and EMBEDDING_API_KEY
   mkdir -p .context/certs              # put llm-vm04-ca.crt and embedding-vm02-ca.crt here
   python scripts/check_endpoints.py    # should print "LLM OK" and "Embeddings OK"
   ```
2. **Start everything** (Docker). Three containers: `postgres`, `backend`, `frontend`.
   ```bash
   docker compose up --build
   ```
   The first start builds the search index (~2 minutes); later starts reuse it.
   - **The app:** http://localhost:3000 (chat with the trace beside it, all traces, eval results)
   - Agent API: http://localhost:8100/docs
3. **Ask something** in the app, then click **Trace ▸** under the answer to see every step the agent took.

**Without Docker:** backend `cd backend && uv sync --extra rag && uv run uvicorn app.main:app --port 8100`;
UI `cd frontend && npm install && npm run dev` (http://localhost:5173).
**Tests:** `cd backend && uv run pytest` (no network needed) · `cd frontend && npm run build` (type check + build).

## Where things are
| Folder | What |
|---|---|
| `backend/` | the agent (LangGraph), tools, API, trace + cache (Postgres) |
| `frontend/` | the React + TypeScript app: [frontend/README.md](frontend/README.md) |
| `rag/` | the search: chunking, hybrid retrieval, reranking, experiments: [rag/README.md](rag/README.md) |
| `eval/` | the grading script + validator: [eval/README.md](eval/README.md) |
| `data/` | the dataset: corpus for RAG, tables for tools, golden queries: [docs/DATASET.md](docs/DATASET.md) |
| `docs/` | **project status**, dataset guide, course brief and upGrad notes |
| `scripts/` | endpoint check |

## Two rules
- **The model key allows one request at a time for the whole team.** Don't run big evals at the same time as someone else.
- **Never commit** `.env`, keys or certificates. `main` takes pull requests with one approval.
