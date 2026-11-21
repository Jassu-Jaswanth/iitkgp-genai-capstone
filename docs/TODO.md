# What's left to build (TODO tracker)

Search the code for `TODO(` to find every spot. Each one is listed here once.

## The picture

```
user question
     │
     ▼
 ┌────────┐   not HR / private?  → polite fixed reply (done)
 │ router │──────────────────────────────────────────────►
 └───┬────┘
     ▼
 ┌─────────┐  calls tools  ┌──────────────────────────────────┐
 │ planner │──────────────►│ search_policy ✅ (example)        │──► retrieve() ──► RAG service ⬜
 │  (LLM)  │◄──────────────│ leave_accrual_calc ⬜  pay_lookup ⬜│
 └───┬─────┘   results     │ holiday_calendar ⬜ escalate_to_hr ⬜│
     ▼                     └──────────────────────────────────┘
 ┌────────┐  wrong? send back once
 │ critic │──────────────────────► answer + citations
 └────────┘
 everything above shows up in Phoenix (traces) ✅
```
✅ done on main · ⬜ to build

## 1. RAG service ⬜ (owner: RAG teammate)
**Think of it as:** a librarian. You hand it a question; it hands back the 5 most useful paragraphs.

- **Input data:** `data/corpus/clauses.jsonl` (dataset card: `docs/DATASET.md`).
- **Build:** load → chunk → embed → index → search → rerank. Any stack.
- **Must serve:** `POST /retrieve` with `{"query", "k", "filters"}` → `{"chunks": [{"chunk_id", "text", "citation", "clause_ids", "score"}]}`.
- **Then:** set `RAG_SERVICE_URL=http://...` in `.env`. No agent code changes.
- **Plug point:** `backend/app/retrieval.py` → `TODO(rag)`; `compose.yaml` → `TODO(rag)`.
- **Tips we measured:** 256-token chunks with a `[citation] heading` line on top + hybrid (BM25 + embeddings) +
  cross-encoder rerank worked best. Reranking alone gave +5–10 points.

## 2. Agent tools ⬜ (owner: agent)
**Think of it as:** giving the model a calculator so it stops doing maths in its head.

| Tool | What it does | Data |
|---|---|---|
| `leave_accrual_calc` | leave hours per pay period, carryover cap | rules in the code comment |
| `pay_lookup` | 2026 GS salary / hourly / overtime | `data/tables/gs_pay_2026.csv` |
| `holiday_calendar` | federal holidays between two dates | `data/tables/holidays.csv` |
| `escalate_to_hr` | drafts an HR case for a human; never submits | none |

- **Where:** `backend/app/tools.py` → `TODO(tools)` (each has args, rules, data).
- **For each tool:** write it → add it in `build_tools()` → add one "question type → tool" line in
  `PLANNER_RULES` (`agent.py`) → add a test → add eval questions that expect it.
- **Gotchas we hit:** always return a `citation` field; don't make optional facts required (the model will invent
  them); say "never guess" in the description.

## 3. Evaluation ⬜ (owner: everyone)
**Think of it as:** an exam the agent must pass, written before we peek at its answers.

- **Write the held-out set:** fill `data/queries/heldout_v1.jsonl` (40 items, `<FILL>` placeholders), following
  `eval/HELDOUT.md`. Someone who didn't tune the agent should write it.
- **Check it:** `python eval/validate_gold.py data/queries/heldout_v1.jsonl --strict`.
- **Run it:** `python eval/run_eval.py --mode agent --gold data/queries/heldout_v1.jsonl`.
- **Golden set (done):** `data/queries/gold_v1.jsonl`, 50 verified questions. 13 of them expect the TODO tools,
  so they double as tests for the tool builders. It's a dev set (we tuned on it), so the held-out set is what we report.
- **Code TODOs:** `eval/run_eval.py` → `TODO(eval)` (LLM-judge faithfulness score, links to traces).

## 4. Trace dashboard ✅ (small TODOs)
**Think of it as:** a flight recorder. Every question → one trace: router, planner, tool calls, retrieved chunks,
critic, with prompts, tokens and time.

- **Already works:** `docker compose up` starts Phoenix at http://localhost:6006.
- **TODO:** auth + API key if Phoenix is shared (`backend/app/tracing.py` → `TODO(tracing)`).

## 5. Chat / eval dashboard ⬜
- **TODO:** a Streamlit dashboard in `frontend/`: chat with the agent (show route, citations, trace) and an eval
  results page (accuracy by category, failures). Compose slot: `TODO(frontend)`.

## 6. Before submission (25–27 Nov)
- [ ] RAG service live, `RAG_SERVICE_URL` set
- [ ] 4 tools in, tests green
- [ ] held-out set frozen and run once; numbers into the report
- [ ] report: what failed, what worked, trade-offs
