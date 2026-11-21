# FedHR-Policy: US federal HR rules + golden questions

**FedHR-Policy** is a small, citable corpus of real US federal HR rules on leave, pay, working hours and
discipline. It is split into self-contained **clauses**, and comes with **golden questions** that have verified
answers for evaluating RAG systems and agents.

| | |
|---|---|
| Language | English |
| Domain | US federal employment law & HR guidance |
| Tasks | retrieval-augmented QA, agent tool use, citation grounding |
| Size | 104 documents · 1,484 sections · 7,574 clauses · 50 golden questions (+ 40 held-out slots) |
| Snapshot | November 2026 |
| Licence | Public domain (US government works) |

## Layout
```
data/
├── corpus/
│   ├── clauses.jsonl       7,574 clauses, one per line (flat)   ← index this
│   └── documents.jsonl     104 documents → sections → clauses (nested)
├── tables/
│   ├── gs_pay_2026.csv     2026 General Schedule pay: 59 localities × 15 grades × 10 steps
│   └── holidays.csv        federal holidays 2021–2030 (112 rows)
└── queries/
    ├── gold_v1.jsonl       50 questions with verified answers (dev set)
    ├── gold_example.jsonl  5 questions (format example)
    └── heldout_v1.jsonl    40-slot held-out template (to be written)
```

## How the corpus is organised
Like a book: **document → section → clause**.
```
document   5 U.S.C. Chapter 63: Leave
└─ section  § 6303 Annual leave; accrual
   └─ clause (a)(2)  three-fourths day for each full biweekly pay period … 3 but less than 15 years
```
A **clause** is one rule that can be quoted on its own. When a clause is a sub-item, it carries the
**lead-in** of its parent paragraph so it still reads correctly in isolation (2,920 clauses have one).

## Subsets

### `corpus/clauses.jsonl`
```json
{"clause_id": "usc5-6303-a-2", "citation": "5 U.S.C. 6303(a)(2)", "marker": "(a)(2)",
 "lead_in": "An employee is entitled to annual leave with pay which accrues as follows—",
 "text": "three-fourths day for each full biweekly pay period, except that the accrual for the last full biweekly pay period in the year is one and one-fourth days, for an employee with 3 but less than 15 years of service; and",
 "section_id": "usc5-6303", "section_citation": "5 U.S.C. 6303", "heading": "Annual leave; accrual",
 "path": ["SUBCHAPTER I-ANNUAL AND SICK LEAVE"], "doc_id": "usc5-ch63", "doc_title": "5 U.S.C. Chapter 63: Leave",
 "source_type": "statute", "as_of": "2024", "url": "https://www.govinfo.gov/link/uscode/5/6303?link-type=html&year=2024"}
```
| Field | Type | Description |
|---|---|---|
| `clause_id` | str | Unique, stable ID |
| `citation` | str | Official citation of this clause, e.g. `5 U.S.C. 6303(a)(2)`, `5 CFR 630.401(b)` |
| `marker` | str | Paragraph marker (`(a)(2)`) for legal text; ordinal for guidance |
| `text` | str | The rule |
| `lead_in` | str \| null | Parent paragraph's opening, for sub-items |
| `section_id`, `section_citation`, `heading`, `path` | | The section the clause belongs to |
| `doc_id`, `doc_title` | str | The document |
| `source_type` | str | `statute` · `regulation` · `guidance` |
| `as_of`, `url` | str | Version and link to the original |

### `corpus/documents.jsonl`
The same text, nested: `{doc_id, title, source_type, authority, version, as_of, url, sections: [{section_id,
citation, heading, path, url, clauses: [{clause_id, citation, marker, text, lead_in}]}]}`.

### `tables/`
| File | Columns |
|---|---|
| `gs_pay_2026.csv` | `LOCNAME` (locality code, e.g. `RUS`, `DCB`, `NY`), `GRADE` (1–15), `ANNUAL1..10`, `HOURLY1..10`, `OVERTIME1..10` |
| `holidays.csv` | `date` (YYYY-MM-DD), `holiday` |

### `queries/gold_v1.jsonl`
```json
{"qid": "g005", "category": "policy",
 "question": "What is the maximum amount of annual leave most federal employees can carry over into the next leave year?",
 "gold_answer": "240 hours (30 days) [5 U.S.C. 6304(a)]", "must_contain": [["240"], ["hour"]],
 "gold_citations": ["5 U.S.C. 6304(a)"], "gold_clause_ids": ["usc5-6304-a"],
 "expected_tools": ["search_policy"], "expected_behavior": "answer", "verified_against": "corpus/OPM tables, 21 Nov 2026"}
```
| Field | Description |
|---|---|
| `question` | As a user would ask it |
| `gold_answer` | Verified answer with its citation |
| `must_contain` | Groups of required facts; an answer is correct if every group matches one alternative |
| `gold_citations` / `gold_clause_ids` | The clause(s) that hold the answer: links each question to the corpus, for retrieval recall |
| `expected_tools` | Tools a correct agent should call (`[]` for guardrail questions) |
| `expected_behavior` | `answer` · `refuse` (another person's records) · `escalate` (needs a human) · `not_covered` (out of scope) |

`heldout_v1.jsonl` has the same fields, with `<FILL>` placeholders to be completed by people who did not tune
the system.

## Statistics
| | Statute | Regulation | Guidance | Total |
|---|---|---|---|---|
| Documents | 1 | 7 | 96 | **104** |
| Sections | 55 | 534 | 895 | **1,484** |
| Clauses | 628 | 3,847 | 3,099 | **7,574** |

- Clause length (words): median 32, 90th percentile 100, max 1,639; 19 clauses exceed 400 words.
- Golden questions by category: policy 31, leave calculation 5, pay 3, holiday 2, multi-step 2, refuse 2,
  out-of-scope 3, escalate 1, safety 1.
- Golden questions by expected behaviour: answer 43, not covered 4, refuse 2, escalate 1.

## Sources
| Source | Content | Version |
|---|---|---|
| US Code, Title 5, Chapter 63 (govinfo) | Leave law | 2024 edition |
| Code of Federal Regulations, Title 5, Parts 531, 550, 551, 575, 610, 630, 752 (eCFR) | Pay, premium pay, overtime, incentives, hours of duty, leave, adverse actions | as of 2026-11-19 |
| US Office of Personnel Management | 96 leave & pay fact sheets; 2026 GS pay tables; federal holidays | retrieved 2026-11-21 |

## Processing
- **Parsing:** each source was parsed from its native structure (eCFR XML, govinfo HTML, OPM HTML); no PDF text
  extraction.
- **Clause splitting:** legal text was split on paragraph markers `(a)(1)(i)(A)`; guidance was split on headings,
  paragraphs, bullet lists and table rows.
- **Cleanup:** editorial notes and "[Reserved]" sections were dropped. IDs were checked to be unique.
- **Golden answers:** each one was checked by hand against the clause text or the OPM table it cites.

## Intended use
- Build and evaluate retrieval (chunking, embedding, reranking) with clause-level relevance labels.
- Evaluate grounded QA and agents: answer correctness, citations, tool use, refusals and escalation.

## Limitations
- A snapshot: rules and pay tables change. Check `as_of` before relying on any answer.
- The same fact often appears in both the law and an OPM fact sheet; only one source is labelled per question.
- `gold_v1` was used during system development, so scores on it are optimistic; report the held-out set.
- Not legal advice. Coverage is limited to the parts listed above (no retirement, health or life insurance rules).

## Citation
```
@misc{fedhr_policy_2026,
  title  = {FedHR-Policy: US federal HR rules and golden questions for grounded QA},
  author = {IIT KGP Gen AI Capstone Team (Track B)},
  year   = {2026},
  note   = {Public-domain sources: US Code, eCFR, OPM. Snapshot November 2026.}
}
```
