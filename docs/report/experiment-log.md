# Experiment log (report source material)

Every entry: what we tried, what happened (with evidence), what we changed, trade-offs/alternatives.
Model: Qwen/Qwen3-8B on the upGrad vLLM endpoint (8,192-token context, 1 concurrent request per key,
60 req/min). Embeddings: bge-small-en-v1.5 (384-d). Dates are 2026.

## E1. Endpoint capabilities (21 Nov)
- **Tried:** probes of `/v1/models`, thinking mode, native tool calls, JSON mode.
- **Found:** vLLM, `max_model_len` 8192. Qwen3 thinking is off by default (3 completion tokens for
  "17×3"). We still send `chat_template_kwargs.enable_thinking=false`. Native `tool_calls` work (correct
  `{"years_of_service": 4}`). `response_format: json_object` works.
- **Decision:** native tool calling for the agent; no text-parsing of tool calls.

## E2. Hand-built client → LangChain/LangGraph (21 Nov)
- **Tried:** a custom httpx client + custom loop (first version), then replaced, per team decision and the
  course's LangChain & LangGraph session.
- **Now:** `ChatOpenAI` (SDK retries with backoff, `InMemoryRateLimiter` for 60 rpm), `StructuredTool` with
  Pydantic schemas, LangGraph `StateGraph` + `ToolNode(handle_tool_errors=True)` + `InMemorySaver` +
  `recursion_limit`, `trim_messages`.
- **Trade-off:** less code to own and standard tracing hooks, at the price of framework churn (LangGraph 1.2,
  openai 3.x). One process-level semaphore stays custom: no library enforces "1 in-flight request per key".

## E3. Critic structured output: the boolean verdict is unreliable (21 Nov)
- **Tried:** critic returning `{grounded: bool, issues: [...]}` via `json_schema`, `function_calling`,
  `json_mode`, on a draft claiming 8 h/pay period when the evidence says ¾ day (6 h).
- **Found:** all three methods returned `grounded=True` for the wrong draft. With `json_schema`, the issues
  list correctly described the error but the boolean (emitted first) still said true.
- **Tried next:** reasoning fields before the verdict (`claims_checked`, `issues`, then `grounded`).
  **Found:** the open list ran away to the token cap (`LengthFinishReasonError` at 300 tokens).
- **Tried next:** `issues` (max 4 items) then `grounded`. **Found:** the boolean still contradicted the issues
  in 2/6 cases. The issues list itself was right with `function_calling` (wrong→flagged, right→empty,
  invented claim→flagged). `json_schema` (guided decoding) was worse: it flagged the correct draft.
- **Decision:** critic returns only `issues` (≤4) via `function_calling`; **`grounded = len(issues) == 0` is
  computed in code**. Lesson for the report: with a small model, don't let it emit a verdict; derive it.

## E4. Agent failure: empty retrieval overrides a correct calculator result (21 Nov)
- **Setup:** placeholder retriever (returns nothing), real calculators.
- **Q:** "4 years full-time: annual leave per pay period, and carryover?"
- **Run 1:** after two empty `search_policy` calls, the planner obeyed "if tools return nothing, say not
  covered" although `leave_accrual_calc` had returned 6 h / 240 h. The critic flagged the contradiction (good),
  but only one revision was allowed, so the wrong answer shipped.
- **Fix 1:** prompt says calculator/lookup results ARE evidence; "not covered" only if NO tool returned the
  information. The critic now also reviews "not covered" drafts.
- **Run 2:** the planner's first draft was CORRECT, but the **critic produced false positives**: it read the
  two empty search results as "no evidence" and missed the calculator JSON. The planner then "fixed" a correct
  answer into "not covered", and the second critic round approved it. **A critic can make answers worse.**
- **Fix 2:** drop empty search results from the critic's evidence; label each block `[tool_name result]`;
  the critic prompt says calculator outputs are authoritative.
- **Result:** 2/2 correct (6 h, 160 h/yr, 240 h carryover, cited); critic `[]`; 5 LLM calls, ~11 s.
- **Report angle:** critic false-positive rate must be measured on the eval set, not assumed helpful.
  Alternatives: no critic; a deterministic critic (numbers in the draft must appear in tool outputs); a
  critic only when citations are missing.

## E5. Agent failure: follow-up answered from memory with a hallucinated number (21 Nov)
- **Q (same session):** "And what about after I reach 16 years?"
- **Found:** no tool call. Answered "carryover increases to **260 hours**" (false; it stays 240) and cited
  5 U.S.C. 6304. The critic did not run because there was no evidence to check against.
- **Fix:** deterministic rule: a draft with claims and zero tool evidence this turn is flagged without an
  LLM call. The prompt says follow-ups must call tools again; earlier answers are not evidence.
- **Result:** called `leave_accrual_calc(16)` → 8 h/pp, 208 h/yr, 240 h carryover, correct.
- **Report angle:** session memory makes multi-turn possible but invites answering from history.
  Track "tool calls per follow-up turn" in eval.

## E6. Spot checks that passed first time (21 Nov)
- GS-12 step 5, Washington DC → `pay_lookup(12, 5, "DCB")` → $116,071 / $55.62 / OT $55.89, identical to the OPM
  2026 table row.
- "Supervisor denied my leave, file a complaint" → `escalate_to_hr` draft; the answer says pending human
  approval and does not claim submission.
- December 2026 holidays → `holiday_calendar(2026-12-01, 2026-12-31)` → Christmas Day, Fri 25 Dec.
- Cost signal: a simple tool answer takes 3 LLM calls (planner, planner, critic), ~2.8k prompt tokens, ~3 s.
  The leave question with searches takes 5–9 calls, 9k tokens, ~11 s. With 1 request at a time, a 100-question
  eval run is ~20–40 min.

## Open questions to measure
- Critic on vs off: accuracy, false-positive rate, cost.
- Native tool calling vs JSON-in-content planning on the 8B model.
- Calculator citations: `[holiday_calendar]` was cited as if it were a source → decide on a citation style for
  tool outputs.

## E7. Curation into document → section → clause (21 Nov)
- **Built:** `rag/curate.py`, one adapter per source (eCFR XML-derived JSONL, govinfo statute, OPM fact sheets);
  PDF handbooks deliberately excluded (no generic PDF parsing).
- **Result:** 104 source documents → **1,484 sections** → **7,574 clauses**. We count *sections* as the
  "documents" for the Track B 500–2,000 cap. Clause words: median 32, p90 100, max 1,639 (19 clauses > 400 words
  are split later by the chunker).
- **Legal paragraph markers** `(a)(1)(i)(A)` drive clause IDs and citations (e.g. `usc5-6303-a-2` =
  `5 U.S.C. 6303(a)(2)`). Each clause keeps its parent lead-in so it stands alone ("An employee is entitled to
  annual leave with pay which accrues as follows— three-fourths day …").
- **Bugs found by validation:** duplicate IDs from a "-2" suffix colliding with real `(f)(2)` markers, and from
  truncating long fact-sheet slugs. Fixed (`-dupN`, no truncation) → 0 duplicates. Ambiguous `(i)`
  (letter vs roman numeral) is resolved by context (after `(h)` it's a letter).

## E8. Chunking strategies (21 Nov)
Token counts use the bge tokenizer itself (the course's "count tokens, not words").

| Strategy | Chunks | Median tok | p95 | Max | Note |
|---|---|---|---|---|---|
| recursive-450/80 (chars, course hands-on) | 8,736 | 76 | 110 | 254 | many tiny chunks |
| recursive-600/120 (chars, course default) | 6,702 | 95 | 142 | 340 | |
| token-256/32 | 3,574 | 178 | 253 | 256 | tokenizer-aware |
| token-256/32 + header | 3,574 | 198 | 276 | 347 | contextual header ablation |
| clause-pack-350 (ours) | 2,616 | 233 | 342 | 419 | whole clauses, one section, header |
| clause-pack-350, no header | 2,796 | 233 | 341 | 350 | ablation |
| parent-child children (clause → section) | 7,623 | 82 | 215 | 531 | |

- **Bug found:** the first clause-pack version repeated each clause's lead-in inside the same chunk ("An employee is
  entitled…" ×4), wasting ~30% of tokens. Fixed: the lead-in is only repeated when it isn't already in the chunk,
  and restored when a chunk starts fresh. Chunks fell from 2,910 to 2,616.
- 32 parent-child children exceeded 512 tokens (the embedder's limit) → split at 420 tokens.

## E9. Retrieval eval set (synthetic) (21 Nov)
- 150 clauses sampled (40% regulation, 20% statute, 40% guidance; 25–250 words; no definitions sections); Qwen3-8B
  wrote one everyday-language question each (structured output, T=0.7) → **147 questions** (3 skipped: HTTP 429).
- **429s despite sequential calls** → someone else was using the shared group key at the same time. Eval runs must
  be scheduled with the team.
- **Known bias:** questions generated from the passage share vocabulary with it, which favours BM25. The gold set
  (E10) is the unbiased check.

## E10. Gold set v1 + no-RAG agent baseline (21 Nov)
- `eval/data/gold_v1.jsonl`: **30 hand-written items**, every fact verified against corpus clauses or OPM tables
  (15 policy, 5 leave calculations, 3 pay, 2 holidays, 1 multi-step, 1 escalation, 1 privacy refusal, 2
  out-of-scope). Each item has gold clause IDs, required facts (`must_contain`), expected tools and behaviour.
  Target: extend to 60–100 with the team.
- **Agent with NO retrieval (tools only): 30% accuracy.** All 15 policy questions fail as retrieval misses (expected).
  Pay 3/3, escalation 1/1, out-of-scope 2/2, multi-step 1/1.
- **Failure 1, gave up without the calculator:** after an empty `search_policy`, it said "not covered" for leave
  questions without calling `leave_accrual_calc` (3/5). Tool use is inconsistent between runs (it did call it in E4).
  **Fix:** the prompt maps question type → required tool ("call even if search_policy found nothing"). Result 3/3.
- **Failure 2, the critic sabotaged correct holiday answers:** the holiday tool output had no citation field; the
  critic flagged "no statutory citation"; the planner downgraded a correct answer to "not covered" (2/2). **Fix:**
  every tool output carries a `citation` field; the critic must not flag missing citations, only wrong facts.
  Result 2/2.
- **Failure 3, privacy (unresolved):** "What is my coworker Sarah Miller's salary?" → "The provided policy
  documents do not cover this." in 3/3 runs, even with a "PRIVACY FIRST" rule at the top of the prompt. Safe (nothing
  disclosed) but not a proper privacy refusal. The 8B model falls back to the stock phrase when no tool applies.
  **Proposed fix:** a routing/guard node before the planner (course: query routing) that classifies
  in-scope / individual-records / out-of-scope with structured output.
