# The dataset, explained simply

## In one sentence
Real US federal HR rules (leave, pay, working hours), cut into small self-contained rules ("clauses") that
each know exactly where they came from.

## Where it comes from
| Source | What it is | Size | Licence |
|---|---|---|---|
| **5 U.S.C. chapter 63** | the law on leave | 55 sections | public domain |
| **5 CFR** parts 531, 550, 551, 575, 610, 630, 752 | regulations: pay, overtime, hours, leave, discipline | 534 sections | public domain |
| **OPM fact sheets** | plain-English guidance from the HR agency | 96 sheets → 895 sections | public domain |
| **OPM tables** | 2026 GS pay table, federal holidays 2021–2030 | 2 CSV files | public domain |

All from official websites (eCFR API, govinfo, opm.gov), no PDFs. Snapshot: November 2026.

## The shape: like a book
```
document   "5 U.S.C. Chapter 63: Leave"                 (104 documents)
└─ section "§ 6303 Annual leave; accrual"                (1,484 sections)
   └─ clause "(a)(2) three-fourths day for each full     (7,574 clauses)
              pay period ... 3 but less than 15 years"
```
- A **clause** is one rule you could quote on its own.
- If a clause is a sub-item ("(2) three-fourths day…"), it carries its **lead-in** ("An employee is entitled to
  annual leave … as follows—") so it still makes sense alone.
- For the project's "500–2,000 documents" rule, we count **sections**: 1,484.

## Files (`data/processed/`)
| File | Use it for |
|---|---|
| `clauses.jsonl` | **Building the RAG index.** One clause per line, flat. |
| `documents.jsonl` | Browsing the structure: same text, nested document → section → clause. |
| `opm/gs_pay_2026.csv` | The `pay_lookup` tool. |
| `opm/holidays.csv` | The `holiday_calendar` tool. |

## One clause (`clauses.jsonl` line)
```json
{
  "clause_id": "usc5-6303-a-2",
  "citation": "5 U.S.C. 6303(a)(2)",
  "marker": "(a)(2)",
  "lead_in": "An employee is entitled to annual leave with pay which accrues as follows—",
  "text": "three-fourths day for each full biweekly pay period, except that the accrual for the last full biweekly pay period in the year is one and one-fourth days, for an employee with 3 but less than 15 years of service; and",
  "section_id": "usc5-6303", "section_citation": "5 U.S.C. 6303", "heading": "Annual leave; accrual",
  "doc_id": "usc5-ch63", "doc_title": "5 U.S.C. Chapter 63: Leave",
  "source_type": "statute",
  "as_of": "2024", "url": "https://www.govinfo.gov/link/uscode/5/6303?link-type=html&year=2024"
}
```
| Field | Meaning |
|---|---|
| `clause_id` | unique ID; RAG chunks list the clause IDs they contain, so we can score retrieval |
| `citation` | what the agent writes in brackets, e.g. `[5 U.S.C. 6303(a)(2)]` |
| `text` + `lead_in` | what to embed (put `lead_in` first if present) |
| `section_*`, `heading` | group clauses back into their section; good chunk header: `[citation] heading` |
| `source_type` | `statute`, `regulation` or `guidance`, usable as a search filter |
| `url`, `as_of` | link to the original, and which version |

## Good to know
- Typical clause: ~32 words; 1 in 10 is over 100 words; **19 are over 400 words** (split those).
- The same fact often appears twice: in the law **and** in an OPM fact sheet. Both are correct to cite.
- The embedding model (bge-small) reads at most 512 tokens per chunk.

## The golden set (the exam), not the corpus
`data/processed/` is what the RAG system **reads**. The questions that **grade** it are in `eval/data/gold_v1.jsonl`:
50 Q&A pairs, each with the citation of the clause that answers it (e.g. `5 U.S.C. 6304(a)`) and its `clause_id`
in this corpus. That link is how we measure whether retrieval found the right rule. Details: `eval/README.md`.

## Rebuild it
Scripts are on the `jaswanth/hr-policy-navigator` branch (no keys needed, a few minutes):
```bash
python scripts/fetch_ecfr.py && python scripts/fetch_uscode.py && python scripts/fetch_opm.py
python scripts/curate.py
```
