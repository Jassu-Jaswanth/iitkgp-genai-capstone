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

## E11. Router / guard node fixes privacy and scope (21 Nov)
- **Design (course: query routing):** a LangGraph `router` node before the planner classifies the latest request
  (with the previous exchange as context) into `hr_policy | individual_records | out_of_scope | chitchat`, using
  `with_structured_output(method="function_calling")` with a short `reason` field first. Non-HR routes get a fixed
  reply with no tools. A router error falls back to `hr_policy`, so the user is never blocked.
- **Offline test:** 10/10 correctly routed, including tricky in-scope cases ("salary for GS-12 step 5" → hr_policy;
  "can my manager see my leave balance" → hr_policy; "supervisor John's GS grade" → individual_records).
- **Live:** g028 (coworker's salary) → proper privacy refusal, 0 tools, 0.7 s (was "not covered" 3/3).
  Out-of-scope 2/2, escalation and pay unaffected. Cost: +1 short LLM call per turn (~0.3–0.7 s).
- **Also found:** (a) revisions after critic feedback leaked meta-text ("…based on the corrected citation…") →
  the feedback now demands "ONLY the corrected final answer". (b) Our own tool bug: the part-time branch
  returned no `sick_citation`, so the model invented "5 U.S.C. 6303(b)" → fixed to 6307(a).

## E12. Operational issues (21 Nov)
- **Hung retrieval grid:** `OpenAIEmbeddings` has no request timeout by default. One stalled request to the shared
  embedding server blocked the run for minutes at 0% CPU while the endpoint answered direct probes in 0.27 s.
  Fix: `request_timeout=60` + `max_retries=5`.
- First-time Hugging Face model download (cross-encoder) can stall in WSL. Pre-download models in the container build.
- Shared key contention again: 429s during sequential generation (see E9).

## E13. Containerisation (21 Nov)
- `backend/Dockerfile`, `frontend/Dockerfile` (python:3.12-slim + uv, locked deps), `compose.yaml`. `.env`, CA certs
  and tool data are mounted read-only at runtime; nothing secret is baked into images (`.dockerignore` excludes
  `.env`, `.context`, `data`). Backend healthcheck on `/health`; the dashboard waits for a healthy backend.
- **Verified:** `docker compose up` → backend healthy; agent in a container answered "GS-12 step 5 DC overtime"
  correctly ($55.89, cited); dashboard HTTP 200.

## E14. Observability with Arize Phoenix (21 Nov)
- **Choice:** Arize Phoenix (one container, OpenTelemetry/OpenInference) instead of self-hosted Langfuse (v3 needs ~5
  services). `openinference-instrumentation-langchain` auto-traces every LangGraph node, LLM call and tool call; we
  added no custom span code (`backend/app/tracing.py`, enabled by `PHOENIX_COLLECTOR_ENDPOINT`).
- **Gotcha:** Phoenix 22 enables auth by default; span export failed with HTTP 401. For local dev it runs with
  `PHOENIX_ENABLE_AUTH=false`, **bound to 127.0.0.1 only**. A shared deployment must use auth + an API key header.
- **One agent request** = 1 trace: router → planner ×2 → tools (leave_accrual_calc) → critic, 4 LLM spans with
  prompts, tokens and latency.
- **First RAG-backed answers:** the carryover question was right (240 h, cited 5 U.S.C. 6304), but the agent appended
  a false qualifier ("applies to … employees with less than 3 years"). It called the calculator with a default
  years value and narrated the band. Candidate for the eval failure taxonomy (hallucination via tool defaults).

## E15. LLM-level tracing: retrieval, tools, model calls in one trace (21 Nov)
- **Requirement:** the trace must show what the agent does (chunks retrieved, tool calls, model calls), not HTTP plumbing.
- **API spans removed:** FastAPI 0.16x emits its own OpenTelemetry spans whenever a *global* tracer provider exists.
  The Phoenix provider is now passed only to the LangChain instrumentor (`set_global_tracer_provider=False`).
  The root span is now `LangGraph`.
- **Retrieval as a LangChain `BaseRetriever`:** `PolicyRetriever` → OpenInference RETRIEVER spans with
  `retrieval.documents` (content, citation, clause ids, score) instead of a JSON blob inside the tool output.
- **One trace across two services:** the backend forwards a W3C `traceparent` to the RAG service, which attaches
  it before invoking its retriever. Gotcha: OpenInference spans are not the OTel "current" span, so a plain
  `inject()` sent nothing (RAG spans became separate root traces). Fix: `openinference...get_current_span()` →
  `trace.set_span_in_context(span)` → `inject`.
- **Result (bone-marrow question):** LangGraph → router (LLM) → planner (LLM) → tools → `search_policy` →
  PolicyRetriever (5 chunks the agent saw) + `rag_service.hybrid_rerank` (10 after rerank) → EnsembleRetriever →
  BM25Retriever (30 candidates) + VectorStoreRetriever (30) → planner (LLM) → critic (LLM). This shows the query the
  agent chose ("leave for bone marrow donation", rewritten from the user's wording) and what each retrieval stage
  returned, which is the evidence needed to classify retrieval misses vs generation errors.

## E16. Retrieval grid results (21 Nov): 7 chunkings × 5 retrievers × 2 query variants, 147 synthetic questions
Files: `eval/results/retrieval_20261121-083407.csv`, `retrieval_20261121-084011.csv` (dense+rerank).
Metrics at depth 10: sec@k = a chunk from the gold section in the top k; cl@k = a chunk containing the gold clause.

| Config (plain query) | sec@1 | sec@5 | cl@5 | MRR | ms/q |
|---|---|---|---|---|---|
| parent-child (clause → section), dense | **0.571** | **0.857** | 0.857* | **0.693** | 35 |
| token-256/32 **+header**, hybrid+rerank | **0.571** | 0.837 | 0.782 | 0.688 | 738 |
| token-256/32 +header, dense+rerank | 0.565 | 0.823 | 0.762 | 0.680 | 500 |
| clause-pack-350, hybrid+rerank | 0.517 | 0.816 | **0.789** | 0.639 | 890 |
| recursive-450/80 (course hands-on), hybrid+rerank | 0.531 | 0.810 | 0.701 | 0.634 | 378 |
| token-256/32 (no header), hybrid+rerank | 0.517 | 0.789 | 0.728 | 0.630 | 676 |
| clause-pack-350, dense | 0.510 | 0.769 | 0.701 | 0.620 | 34 |
| token-256/32 +header, dense | 0.503 | 0.762 | 0.667 | 0.608 | 27 |
| token-256/32 +header, hybrid-rrf | 0.483 | 0.721 | 0.660 | 0.590 | 37 |
| recursive-450/80, dense | 0.429 | 0.701 | 0.558 | 0.549 | 35 |
| best BM25 (token+header) | 0.354 | 0.619 | 0.531 | 0.458 | 7 |

\* parent-child cl@5 is inflated: a parent contains all its clauses.

**Findings**
1. **The reranker is the biggest lever:** +5–10 points of sec@5 for every chunking (e.g. recursive-450: 0.701 → 0.810).
   Cost: 0.4–0.9 s/query on CPU (ms-marco-MiniLM-L-6-v2, 30 candidates).
2. **Contextual headers help everywhere:** token-256 0.789 → 0.837 (rerank), clause-pack 0.789 → 0.816 (rerank),
   dense 0.728 → 0.769. Cheap: about +20 tokens per chunk.
3. **Equal-weight hybrid RRF is often WORSE than dense alone** (token+header: 0.721 vs 0.762; recursive-450: 0.680 vs
   0.701). BM25 alone is weak (0.52–0.62) and drags the fused ranking down. Yet hybrid+rerank (0.837) beats
   dense+rerank (0.823): BM25 helps as a **candidate generator** for the reranker, not as an equal voter.
   Alternatives: weighted RRF (e.g. 0.3/0.7), or BM25 only as a reranker feeder.
   This contradicts the expectation that synthetic questions favour BM25 (E9). The paraphrase prompt worked.
4. **BGE query instruction: no effect** (±0.01 everywhere) on this endpoint. The course's "#1 silent bug" didn't
   materialise for bge-small-en-v1.5 v1.5 (its model card says the instruction is optional for v1.5).
5. **Parent-child gives the best section-level scores at dense cost (35 ms)** but returns whole sections (truncated
   to ~700 tokens). Five results take ~3.5k of the 8k context and may cut off the needed clause.
6. **Chunk size:** the course default recursive-600/120 is worse than 450/80 (sec@5 0.789 vs 0.810 with rerank);
   tokenizer-aware 256-token chunks + header is best among chunk strategies.

**Decision for the running service:** token-256/32 + header + hybrid+rerank (best chunk-level config, ~1k tokens of
context for 5 chunks). Parent-child is the runner-up to test end-to-end. Caveat: synthetic questions; confirm on the
gold set.

## E17. Reproducible hallucination via tool defaults (21 Nov)
- "Maximum annual leave I can carry over?" → correct 240 h [5 U.S.C. 6304], but in 2/2 runs (two RAG configs) the
  agent appended "applies to … employees with less than 3 years of creditable service". The user gave no years.
  The agent called `leave_accrual_calc` with a made-up `years_of_service` and narrated the service band.
- **Root cause:** the tool requires `years_of_service` even for a carryover-only question, so the model invents one.
- **Candidate fixes:** make `years_of_service` optional (carryover doesn't depend on it); tell the planner to ask for
  missing facts instead of inventing them; have the critic flag claims about user attributes not stated by the user.

## E18. Gold-set evaluation: single-prompt strategies vs agent (21 Nov)
Setup: gold set v1 (50 items), RAG = token-256/32+header, hybrid+rerank (E16), Qwen3-8B, one request at a time.
Results: `eval/results/e2e_*_rag-v1_*`, `e2e_agent_rag-v2_*`, `e2e_agent_rag-v3_*`.

| Setup | Accuracy | Policy (31) | Tools/guardrails (19) | Cited | LLM calls | Prompt tok | p50 |
|---|---|---|---|---|---|---|---|
| chat: grounded | 0.76 | 30 | 8 | 0.87 | 1.0 | 1,302 | 2.5 s |
| chat: few-shot JSON | 0.74 | 29 | 8 | 0.87 | 1.0 | 1,510 | 2.5 s |
| chat: chain-of-thought | 0.72 | 29 | 7 | 0.92 | 1.0 | 1,344 | 4.3 s |
| chat: decompose | 0.66 | 24 | 9 | 0.79 | 2.0 | 1,573 | 4.1 s |
| agent v1 | 0.82 | 22 | **19** | 0.53 | 2.6 | 4,280 | 4.5 s |
| agent v2 (forced first tool call) | 0.90 | 26 | 19 | 0.68 | 3.1 | 5,461 | 5.1 s |
| **agent v3 (+ user question in search)** | **1.00** | **31** | **19** | 0.71 | 2.8 | 5,510 | 5.7 s |

**Prompting strategies (single prompt):** plain grounded was best (0.76). Few-shot JSON and CoT gave no gain
(CoT: more "not covered" on answerable questions, 4 vs 3). Decompose was worst (0.66): its sub-queries retrieved
worse chunks than the original question (9 retrieval misses vs 3). No single-prompt strategy can do pay (0/3),
escalation or privacy refusals: these need tools or routing.

**Agent v1 → v2: the agent "gave up" without searching (6/50).** The planner answered "The provided policy documents
do not cover this." with ZERO tool calls on answerable questions (FMLA eligibility, within-grade waits, retention
incentive…). Our own critic rule accepted "not covered + no evidence". Plus 1 router misroute ("waiting period
between step 5 and step 6" → out_of_scope).
Fixes: `bind_tools(..., tool_choice="required")` on the first planner step of each turn (vLLM supports it); the critic
sends back "not covered" if `search_policy` didn't run; the router prompt lists steps/WGIs/incentives/FMLA; tool
descriptions say pay_lookup is "dollar amounts only, never guess a grade" (with forced tool use it had called
`pay_lookup(grade=1)` for a waiting-period question).

**Agent v2 → v3: the agent's own query rewriting hurt retrieval (5/50).** All 5 remaining failures were answered
correctly by the single-prompt baseline, which searches with the user's wording. The agent's "name of the rule"
queries pulled neighbouring rules (administrative leave → 45-day carryover text; retention incentive 25% → 50% waiver
clause; restored leave → carryover). Fix: `search_policy` reads the user's question via LangGraph `InjectedState`
(hidden from the model's schema), searches with BOTH queries and interleaves results. → 31/31 policy.
Lesson for the report: the course's "query with the rule name" advice was harmful with our reranked retriever;
keeping the user's wording as one query is a cheap, robust hedge.

**Measurement fixes along the way**
- Retrieval recall for the agent was under-measured (0.14) because trace content is truncated at 1,500 chars and clause
  ids come after the chunk text. `/agent` now returns `retrieved_clause_ids` from the untruncated results: true value 0.49.
- Recall (0.49) < accuracy (1.00): the same fact often appears in an OPM fact sheet as well as in the statute, and the
  gold set lists one source per question. Gold items should list all equivalent clauses.
- g009 grading false negative ("there is no general cap" was correct). Accepted phrasings widened, and noted in the
  item (`grading_note`).

**Caveats:** (1) the gold set was written by us and the agent was then fixed against its failures, so 1.00 is
optimistic. A held-out set written by teammates is required before claiming accuracy in the report. (2) n=50,
single run, temperature 0. (3) Cost: the agent uses ~4× the prompt tokens and ~2× the latency of a single prompt.
