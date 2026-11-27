# Project status

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
✅ done

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

## Results
- Held-out run (agent mode) and the four single-prompt baselines (`grounded`, `few_shot_json`, `cot`, `decompose`):
  `eval/results/`.
- Final report and demo video: `report/`.

## Future work
- LLM-as-judge faithfulness score next to the string match in `eval/run_eval.py`.
- Larger prompt-injection and PII test sets in the evaluation data.
