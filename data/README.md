# Data: curated federal HR policy corpus (hand-off to the RAG build)

Everything here is **public domain** (US government works). Curated 21 Nov 2026.

| File | What |
|---|---|
| `processed/clauses.jsonl` | **7,574 clauses**, one per line, flat: the main input for chunking/indexing |
| `processed/documents.jsonl` | Same content nested: 104 documents → 1,484 sections → clauses |
| `processed/opm/gs_pay_2026.csv` | OPM 2026 GS pay table (59 localities × 15 grades; annual/hourly/overtime per step). Used by the agent's `pay_lookup` tool |
| `processed/opm/holidays.csv` | Federal holidays 2021–2030 (OPM). Used by `holiday_calendar` |

## Sources
| Source | Documents | Sections | How it was obtained |
|---|---|---|---|
| 5 CFR parts 531, 550, 551, 575, 610, 630, 752 (regulations) | 7 | 534 | eCFR API, as of 2026-11-19 |
| 5 U.S.C. chapter 63 (leave statute) | 1 | 55 | govinfo, 2024 edition |
| OPM leave & pay fact sheets (guidance) | 96 | ~895 | opm.gov HTML |

No PDFs: each source has its own adapter that maps its native structure to one model
**document → section → clause**. A clause is one self-contained rule (with its parent's lead-in when the
paragraph is a sub-item), so chunks can be built from whole rules and cited precisely.
For the Track B "500–2,000 documents" requirement we count **sections (1,484)**.

## `clauses.jsonl` schema
```jsonc
{
  "clause_id": "usc5-6303-a-2",            // unique, stable
  "citation": "5 U.S.C. 6303(a)(2)",        // cite this in answers
  "marker": "(a)(2)",                      // paragraph marker (legal sources) or ordinal (fact sheets)
  "text": "three-fourths day for each full biweekly pay period, except ...",
  "lead_in": "An employee is entitled to annual leave with pay which accrues as follows—",  // or null
  "section_id": "usc5-6303", "section_citation": "5 U.S.C. 6303", "heading": "Annual leave; accrual",
  "path": ["SUBCHAPTER I-ANNUAL AND SICK LEAVE"],
  "doc_id": "usc5-ch63", "doc_title": "5 U.S.C. Chapter 63: Leave",
  "source_type": "statute",                // statute | regulation | guidance
  "as_of": "2024", "url": "https://www.govinfo.gov/link/uscode/5/6303?link-type=html&year=2024"
}
```
Word counts: median 32, p90 100; 19 clauses exceed 400 words (split them when chunking).
The text of a clause for embedding is usually `marker + lead_in + text` (drop the lead-in if the parent clause is
already in the same chunk).

## The interface the server needs from the RAG build
The agent only calls **`retrieve(query, k, filters)`**, over HTTP:

```
POST {RAG_SERVICE_URL}/retrieve
{"query": "how much annual leave can I carry over", "k": 5, "filters": {"source_type": "statute"}}   // filters optional

200 → {"chunks": [
  {"chunk_id": "usc5-6304#0",                 // unique
   "text": "[5 U.S.C. 6304] Annual leave; accumulation\n(a) Except as provided ...",
   "citation": "5 U.S.C. 6304",               // section- or clause-level citation
   "clause_ids": ["usc5-6304-a", "..."],      // clause_ids from clauses.jsonl covered by this chunk
   "score": 0.83, "source_url": "https://...", "metadata": {...}}   // score/source_url/metadata optional
]}
```
- Ranked best-first; return at most `k`. Stateless and fast: the agent may call it **twice per search** (its own
  rule-name query + the user's original question) and merges results.
- `citation` is what the model cites; `clause_ids` let the evaluation measure retrieval recall.
- A placeholder is used until `RAG_SERVICE_URL` is set (the agent then says the documents don't cover it).

## What our experiments found (a starting point, not a requirement)
147 synthetic questions, metrics at depth 10 (see `docs/report/experiment-log.md`, E16):
- **Best:** tokenizer-aware chunks of 256 bge tokens (32 overlap) **with a `[citation] heading` header on every chunk**,
  hybrid BM25 + dense (RRF) → **cross-encoder rerank** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, 30 candidates):
  section hit@5 0.837, MRR 0.688, ~0.7 s/query on CPU.
- Reranking added +5–10 points everywhere; headers +2–5; equal-weight RRF *without* rerank was worse than dense alone.
- The bge query instruction prefix made no difference on the upGrad bge-small endpoint.
- Parent-child (embed clauses, return sections) scored well (hit@5 0.857) but returns ~700-token sections; mind the
  8k context.
- Embedding endpoint: max 16 texts/request, ≤512 tokens each; set a request timeout (a hung request stalled a run).

## Rebuilding from source
The fetch and curation scripts are on the `jaswanth/hr-policy-navigator` branch (stdlib only, a few minutes):
```bash
python scripts/fetch_ecfr.py && python scripts/fetch_uscode.py && python scripts/fetch_opm.py   # -> data/raw/
python scripts/curate.py                                                                       # -> data/processed/
```
