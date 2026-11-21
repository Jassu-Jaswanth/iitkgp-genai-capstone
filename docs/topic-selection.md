# Topic selection: Track B "Multi-Agent Operations Copilot"

Source: `resources/Capstone-Project-Templates.pdf` (Track B section) and the API limits in
`upgrad-communications.md`. **Status: proposal, team to decide.** First confirm what
domain (if any) was entered on the Track Selection form; the LMS shows it as submitted.

## What Track B grades us on

The system is fixed (Qwen3-8B + bge-small embeddings, no weights or logits). The *domain* is ours to pick.
Every domain must deliver the same six parts:

| # | Part | Hard requirement |
|---|---|---|
| I | Task spec + eval set | What it does/refuses, correct-answer definition, **60–100 queries with gold answers, frozen before code** |
| II | Prompt baseline | Single prompt; compare ≥3 strategies (few-shot structured output, CoT/self-consistency, decomposition); document how the 8B model breaks |
| III | RAG | **500–2,000 docs**, chunking experiments, hybrid BM25 + dense, reranker, citations, recall@k measured separately |
| IV | Agents | ≥3 tools, planner → executor → critic, state across turns, handles tool failure / bad JSON / loops / token cap (≤10 steps) |
| V | Eval + observability | Accuracy, completion rate, tool calls/task, cost and latency, failure taxonomy (retrieval / reasoning / tool / hallucination), tracing (Langfuse etc.) |
| VI | Ship | Streamlit/React, Docker, rate limiting, caching; prompt-injection tests on the corpus, PII, refusals, human-in-the-loop boundary |

Grading: technical depth 30%, consistent progress 25%, working deployed system 20%, report + demo 25%.
"What did we do to compensate for a small model?" is a graded section.

## What makes a domain a good fit *for us, now*

1. **Corpus is public and downloadable today**, and splits naturally into 500–2,000 documents.
2. **Gold answers are objective** (numbers, rule citations, yes/no + clause), so 60–100 eval items can be written fast and scored automatically.
3. **Tools do the maths/lookup, not the model.** Deterministic tools are how an 8B model wins.
4. **A natural refusal and human-approval boundary** for the responsible-AI section.
5. Works within **8k context and 1 concurrent request per key**.

## Shortlist

### 1. Travel & Expense Policy Agent (recommended)
- **Corpus:** US Federal Travel Regulation (41 CFR 300–304, section-level, public on eCFR) + GSA per-diem
  pages/FAQs + a few public university/agency T&E policies. Splits easily into ~600–1,200 docs.
- **Tools:** per-diem rate lookup (GSA public data), currency conversion (static rate table),
  expense-claim validator (rule checks + totals), policy search (RAG), "submit for approval" (HITL stub).
- **Eval:** "Can I claim X?", "What is the M&IE rate for city Y in month Z?", "Is this claim compliant?" Answers are numeric or rule-cited, so scoring is objective.
- **Why it wins:** heavy on deterministic tools and calculations (where small models fail and tools rescue them),
  a clean refusal/approval story, and prompt-injection tests embed easily in policy text.
- **Risk:** US-centric rules (fine for an academic project).

### 2. Financial-Report Q&A Analyst
- **Corpus:** SEC EDGAR 10-K/10-Q filings for ~15–25 companies, chunked by Item/section.
- **Tools:** XBRL company-facts lookup (SEC API, no key), ratio calculator, filing-section retriever, year-over-year comparer.
- **Eval:** numeric gold answers from XBRL, so it's very objective.
- **Risk:** long tables blow the 8k context; table parsing eats time. A strong option if someone has finance background.

### 3. HR Policy Navigator
- **Corpus:** public government HR rules (e.g. US OPM handbooks or India CCS leave/conduct rules) + public company handbooks.
- **Tools:** leave-balance calculator, policy search, holiday calendar, HR-ticket creation (HITL).
- **Eval:** moderately objective; more interpretation, so it's harder to auto-grade.
- **Risk:** gold answers are fuzzier; corpus assembly is slower.

Not shortlisted: incident-postmortem drafter (generative output is hard to score),
competitive-research analyst (needs live web, less reproducible), procurement (vendor APIs need registration).

## Proposed plan with ~6 days left (submit by 27 Nov)

| Day | Focus (roles from the template) |
|---|---|
| 21–22 Nov | Lock topic; write spec + refusal policy; build corpus; **freeze 60–100 eval queries**; prompt baselines |
| 22–23 Nov | RAG: chunking, BM25 + dense, reranker, citations, recall@k |
| 23–24 Nov | Agent loop (planner/executor/critic), 3+ tools, JSON validation + retries, tracing |
| 25 Nov | Full eval run (queued, 1 request at a time), failure taxonomy, prompt-injection tests |
| 26 Nov | Streamlit UI, Docker, caching/rate limit; report draft |
| 27 Nov | Submit. Then prepare the 10-min demo for 5 Dec |

Plan for slow evals: one key, one request at a time. 100 queries × ~5 LLM calls × ~5 s ≈ 40+ min per full run.
Cache aggressively and don't let teammates run evals simultaneously on the same key.
