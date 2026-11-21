# Datasets: HR Policy Navigator (Track B)

**Status:** **Decided 21 Nov 2026: Option A (US federal HR: eCFR + OPM).** Raw files go in `data/raw/` (git-ignored).

Ready-made "HR policy" datasets on Hugging Face and Kaggle are tiny or synthetic (a few hundred generated
Q&A rows, or tabular employee records), so the corpus has to come from **real published policy documents**.
Two viable routes:

## Option A (chosen): US federal employee HR policy

One coherent employer (the US federal workforce), public domain, short self-contained documents, and
rules with **exact numeric answers**, so gold answers and calculators are easy.

| Source | What it gives us | Size | License |
|---|---|---|---|
| [eCFR Title 5](https://www.ecfr.gov/current/title-5) via the [eCFR API](https://www.ecfr.gov/developers/documentation/api/v1) | Regulation **sections** (one doc each): GS pay (531), pay administration (550), overtime/FLSA (551), incentives (575), hours of duty (610), leave (630), adverse actions (752). Optional: health/life insurance (890/870) | **Downloaded: 534 sections** (as of 2026-11-19) via `scripts/fetch_ecfr.py`. 144 exceed 512 tokens → need chunking | Public domain (US govt work) |
| [5 U.S.C. chapter 63](https://www.govinfo.gov/content/pkg/USCODE-2024-title5/html/USCODE-2024-title5-partIII-subpartE-chap63.htm) (govinfo, 2024 edition) | The **statute** behind the leave rules: accrual rates (§ 6303), carryover cap of 30 days = 240 hours (§ 6304), sick leave (§ 6307), leave transfer/banks, FMLA (§§ 6381–6387) | **Downloaded: 55 sections** via `scripts/fetch_uscode.py` | Public domain |
| OPM [leave](https://www.opm.gov/policy-data-oversight/pay-leave/leave-administration/fact-sheets/) and [pay](https://www.opm.gov/policy-data-oversight/pay-leave/pay-administration/fact-sheets/) fact sheets | Plain-language guidance: annual/sick leave, FMLA, paid parental leave, LWOP, leave banks, bereavement, court/military leave, overtime, holiday pay, premium pay | **Downloaded: 96 fact sheets** (36 leave, 60 pay); median 1,356 words → chunk by heading | Public domain |
| OPM [handbooks](https://www.opm.gov/policy-data-oversight/pay-leave/reference-materials/handbooks/) (PDF) | Childbirth/adoption/foster care, elder care, emergencies/dismissal, recruitment flexibilities | **Downloaded: 5 PDFs** + extracted text (needs `pdftotext`) | Public domain |
| OPM [2026 GS pay table](https://www.opm.gov/policy-data-oversight/pay-leave/salaries-wages/) | Annual, hourly and overtime rate per locality × grade × step | **885 rows** (59 localities × 15 grades), CSV | Public domain |
| OPM [federal holidays](https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/) (iCalendar feed) | Observed holiday dates | **112 holidays, 2021–2030**, CSV | Public domain |

**Corpus (downloaded 21 Nov):** 534 regulation sections + 55 statute sections + 96 fact sheets + 5 handbooks
≈ **690 documents** (within 500–2,000). Add parts 890/870 if we want benefits coverage.

### Rebuild the data
```bash
python scripts/fetch_ecfr.py      # data/raw/ecfr/title5_<date>.jsonl (+ raw XML)
python scripts/fetch_uscode.py    # data/raw/uscode/usc5_chap63_2024.jsonl (+ raw HTML)
python scripts/fetch_opm.py       # data/raw/opm/{factsheets,handbooks}.jsonl, gs_pay_2026.csv, holidays.csv
```
Stdlib only; handbook text extraction needs `pdftotext` (`sudo apt install poppler-utils`).
A few links on OPM's own index pages are dead (404) and are skipped with a log line.

Regulations alone are not enough for the common questions: e.g. the 4/6/8-hour accrual rates and the
240-hour carryover cap live in the **statute and OPM fact sheets**, not in 5 CFR 630.

**Tools (≥3, deterministic where the 8B model is weak):**
- `search_policy`: hybrid BM25 + dense over statute, regs and fact sheets, with citations (e.g. "5 U.S.C. 6303", "5 CFR 630.202")
- `leave_accrual_calc`: annual/sick leave hours by years of service and pay periods; carryover caps
- `pay_lookup`: GS grade/step/locality salary, overtime rate
- `holiday_calendar`: federal holidays and pay periods for a date range
- `escalate_to_hr`: human-in-the-loop stub for consequential actions (adverse actions, disputes)

**Eval examples (objective, checked against the OPM annual leave fact sheet):**
"How much annual leave does a full-time employee with 4 years of service earn per pay period?"
(6 hours; 10 hours in the last pay period of the year; 5 U.S.C. 6303) · "Under 3 years?" (4 hours) ·
"What is the maximum annual leave carryover for a regular employee?" (240 hours; 5 U.S.C. 6304) ·
"Can I use sick leave to care for a family member, and how much?" · "What is my overtime rate at GS-12 step 5?"

**Refusals:** individual legal advice, personnel decisions about named people, other employees' records.

**Changing-documents stretch fits naturally:** the eCFR API serves every section **as of any date**,
so "what was the rule in 2020 vs now?" needs no extra data.

## Option B: company employee handbooks

Closer to a typical corporate HR bot, but mixing companies gives conflicting answers, so pick **one** as
"our company" and use others only for extra coverage.

| Source | Size | License |
|---|---|---|
| [GitLab handbook](https://gitlab.com/gitlab-com/content-sites/handbook) (people-group 67, total-rewards 25, hiring 14 pages; long pages, split by heading) | ~106 pages → several hundred sections | CC BY-SA 4.0 per handbook site (repo LICENSE covers software only; confirm) |
| [CivicActions guidebook](https://github.com/CivicActions/guidebook) | 151 md files (whole-company, part HR) | CC BY 4.0 |
| [Obvious playbook](https://github.com/obvious/playbook), [Clef handbook](https://github.com/clef/handbook) | 104 / 39 md files | CC0 |
| [Basecamp handbook](https://github.com/basecamp/handbook) | 16 md files | No license file in repo (CC BY announced in 2017; confirm) |
| [Sparksuite employee handbook](https://github.com/sparksuite/employee-handbook) | 1 file | CC0 |
| Kaggle [US Employee Handbook](https://www.kaggle.com/datasets/abhilashjash/us-employee-handbook-dataset) | 1 PDF, 450 KB | Apache-2.0 |

Weaker for evaluation: fewer numeric rules, more prose, and GitLab moved many people policies to an internal
(non-public) handbook.

## Considered, not used

| Dataset | Why not |
|---|---|
| HF `strova-ai/hr-policies-qa-dataset`, `adarsh-08/hr-policy-dataset`, `xwjzds/extractive_qa_question_answering_hr` etc. | <1K synthetic rows; no real source documents |
| ViDoRe v3 HR | EU HR *reports* (visual documents), not employee policies |
| Kaggle HR "employee datasets" | Tabular attrition/employee records, not policies |

## Open items
- Next: chunking (144 regulation sections and most fact sheets/handbooks exceed 512 tokens), then indexing.
