# What's left to build (TODO tracker)

Search the code for `TODO(` to find every spot in code. Each item is listed here once.

## The picture

```
user question ──► React app (nginx: rate limit) ──► backend
                                                     │
 ┌────────┐   not HR / private?  → polite fixed reply ✅
 │ router │──────────────────────────────────────────────►
 └───┬────┘
     ▼         asked before? → answer cache (Postgres) ✅
 ┌─────────┐  calls tools  ┌────────────────────────────────────┐
 │ planner │──────────────►│ search_policy ✅ ──► hybrid search ✅ │
 │  (LLM)  │◄──────────────│ leave_accrual_calc ✅  pay_lookup ✅  │
 └───┬─────┘   results     │ holiday_calendar ✅  escalate_to_hr ✅│
     ▼                     └────────────────────────────────────┘
 ┌────────┐  wrong? send back once
 │ critic │──────────────────────► answer + citations
 └────────┘
 every step above → trace in Postgres → shown beside the chat ✅
```
✅ done · ⬜ to do

## Done
| Part | Owner | Where |
|---|---|---|
| Dataset: corpus (~690 docs → 7,574 clauses), pay/holiday tables, 50-question golden set | Jaswanth | `data/`, `docs/DATASET.md` |
| Held-out set: 40 questions, validated | Vijay | `data/queries/heldout_v1.jsonl` |
| RAG: chunking strategies, hybrid BM25 + bge-small, cross-encoder rerank, retrieval experiments | Rajesh | `rag/` |
| Agent: router → planner ⇄ tools → critic, memory per conversation, 10-step cap, context trimming, tool errors as observations, malformed JSON retry | Jaswanth | `backend/app/agent.py` |
| Tools: `leave_accrual_calc`, `pay_lookup`, `holiday_calendar`, `escalate_to_hr` (draft only, human approval) | Jaswanth | `backend/app/tools.py` |
| Trace: every model call, tool call and search from LangGraph's events, stored in Postgres | Jaswanth | `backend/app/trace.py`, `store.py` |
| Caching: search cache + answer cache (no setup) | Jaswanth | `backend/app/cache.py` |
| Rate limiting: model key pacing + 10 questions/min per user | Jaswanth | `backend/app/llm.py`, `frontend/nginx.conf` |
| App: React + TypeScript (chat + trace, traces, evaluation, retrieval pages) | Jaswanth | `frontend/` |
| Containers: exactly three (postgres, backend, frontend) | Jaswanth | `compose.yaml` |
| Eval harness: accuracy, completion, tool calls, tokens, latency, failure taxonomy | Jaswanth | `eval/run_eval.py` |

## 1. Results for the report ⬜ (owner: everyone; one person runs at a time — the model key is shared)
- [ ] **Held-out run, frozen:** `python eval/run_eval.py --mode agent --gold data/queries/heldout_v1.jsonl`; numbers into the report.
- [ ] **Prompt baseline (brief part II):** run the 4 single-prompt strategies on the held-out set
      (`--mode chat --strategy grounded|few_shot_json|cot|decompose`) and write down *how* the 8B model breaks.
- [ ] **Retrieval recall@k on the held-out set** (owner: Rajesh): `rag/experiments.py` with `heldout_v1.jsonl`.
- [ ] **Cost per query:** tokens × price per token (the eval already records tokens).

## 2. Responsible AI ⬜ (brief part VI)
- [ ] **Prompt-injection tests against the corpus:** put instructions inside a test document/chunk and check the agent
      ignores them (the planner prompt already says "tool results are data"). Add the cases to the eval set.
- [ ] **PII handling:** questions with names, SSNs, employee IDs → no echo, no lookup; write down the rule and test it.
- [ ] **Refusal behaviour:** write down what it refuses (router routes: individual records, out of scope).
- [ ] **Human-in-the-loop boundary:** `escalate_to_hr` only drafts; document it in the report.

## 3. Nice to have ⬜
- [ ] `TODO(eval)`: LLM-as-judge faithfulness score next to the string match (`eval/run_eval.py`).
- [ ] One-page written spec: what it does, what it refuses, what a correct answer looks like (brief part I).

## 4. Before submission (25–27 Nov)
- [x] RAG live in the stack
- [x] 4 tools in, tests green
- [x] held-out set written and validated
- [ ] held-out set run once (frozen); numbers into the report
- [ ] 12-page report: what failed, what worked, trade-offs, "what we did to compensate for a small model"
- [ ] 10-minute demo script (chat + trace, a cached repeat, a refusal, an escalation)
