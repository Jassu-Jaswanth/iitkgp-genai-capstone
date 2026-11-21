# Datasets: HR Policy Navigator (Track B)

**Status:** shortlist, 21 Nov 2026 (branch `jaswanth/hr-policy-navigator`). Raw files go in `data/raw/` (git-ignored).

Ready-made "HR policy" datasets on Hugging Face and Kaggle are tiny or synthetic (a few hundred generated
Q&A rows, or tabular employee records), so the corpus has to come from **real published policy documents**.
Two viable routes:

## Option A (recommended): US federal employee HR policy

One coherent employer (the US federal workforce), public domain, short self-contained documents, and
rules with **exact numeric answers**, so gold answers and calculators are easy.

| Source | What it gives us | Size | License |
|---|---|---|---|
| [eCFR Title 5](https://www.ecfr.gov/current/title-5) via the [eCFR API](https://www.ecfr.gov/developers/documentation/api/v1) | Regulation **sections** (one doc each): leave (part 630), pay administration (550), overtime/FLSA (551), hours of duty (610), GS pay (531), incentives (575), adverse actions (752), health/life insurance (890/870) | **1,106 sections** in those parts (counted 21 Nov). Core subset 630+550+551+610+531+575 = 518 | Public domain (US govt work) |
| [OPM pay & leave handbooks and fact sheets](https://www.opm.gov/policy-data-oversight/pay-leave/reference-materials/handbooks) | Plain-language explanations of the same rules (e.g. [childbirth/adoption flexibilities, Apr 2025](https://www.opm.gov/chcoc/transmittals/2025/Rev%20CPM%202025-04%20Handbook%20on%20Flexibilities%20for%20Childbirth%20Adoption%20and%20Foster%20Care.pdf), [leave administration](https://www.opm.gov/oca/leave/)) | Tens of PDFs/pages, split by section | Public domain |
| OPM salary tables + federal holiday list | Structured data for tools (GS grade/step pay by locality; holidays) | Small CSV/XLSX | Public domain |

**Corpus plan:** ~518–800 regulation sections + ~100–300 OPM handbook/fact-sheet sections → within 500–2,000.

**Tools (≥3, deterministic where the 8B model is weak):**
- `search_policy`: hybrid BM25 + dense over regs and handbooks, with citations (e.g. "5 CFR 630.202")
- `leave_accrual_calc`: annual/sick leave hours by years of service and pay periods; carryover caps
- `pay_lookup`: GS grade/step/locality salary, overtime rate
- `holiday_calendar`: federal holidays and pay periods for a date range
- `escalate_to_hr`: human-in-the-loop stub for consequential actions (adverse actions, disputes)

**Eval examples (objective):** "How much annual leave does an employee with 4 years of service earn per
pay period?" (4 hrs, 5 CFR 630.202) · "What is the maximum annual leave carryover?" (240 hrs) ·
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
- Team to confirm Option A vs B.
- eCFR/OPM need no keys; downloads are scriptable into `data/raw/`.
