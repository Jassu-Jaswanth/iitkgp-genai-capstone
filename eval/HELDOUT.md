# Held-out evaluation set (v1): instructions for the team

The agent was tuned against the 50-question golden dev set (`data/queries/gold_v1.jsonl`), so its score there is optimistic.
The **held-out set** is the number we report. It only works if it stays unseen:

- **Written by someone who did not tune the agent or the prompts.** Don't run these questions through the system
  while writing them, and don't change prompts/tools after seeing their results (that turns them into tuning data).
- **Every fact verified against the source document** (statute, regulation, OPM fact sheet / table). Record the
  citation you checked in `gold_citations`.
- **Realistic wording**: how an employee or HR specialist would actually ask, including vague or multi-part questions.
- **Freeze before running.** Once `validate_gold.py --strict` passes, commit, then run the eval once and report it.

## Fill in `data/queries/heldout_v1.jsonl`
Each line is one item. Replace every `<FILL...>` value. Category and `qid` are pre-assigned to hit the quotas below.

| Field | What to write |
|---|---|
| `question` | The user's question, exactly as asked |
| `gold_answer` | The correct answer in one or two sentences, with its citation |
| `must_contain` | List of groups; **each** group must be matched by at least one alternative (case-insensitive). E.g. `[["240"], ["hour"]]`, or `[["4 hours", "half day", "one-half day"]]` |
| `gold_citations` | Legal/official citation(s) of the source, e.g. `["5 U.S.C. 6304(a)"]`, `["5 CFR 630.401(b)"]`, `["OPM fact sheet: Annual Leave"]`. List **every** source that states the fact (statute and fact sheet often both do). Empty `[]` for tool/guardrail items |
| `gold_clause_ids` | **Retrieval placeholder**: leave `["<FILL: clause id from the data team's corpus>"]` until the data team publishes its clause/chunk IDs; then fill or set `[]`. Recall falls back to `gold_citations` |
| `expected_tools` | Tools a correct agent must call: `search_policy`, `leave_accrual_calc`, `pay_lookup`, `holiday_calendar`, `escalate_to_hr`, or `[]` |
| `expected_behavior` | `answer`, `refuse` (privacy / another person's records), `escalate` (needs a human decision), `not_covered` (out of scope / injection) |

## Category quotas (40 items)

| Category | n | Examples of what to cover |
|---|---|---|
| `policy` | 18 | leave types not in gold_v1 (court, military, LWOP, bereavement), premium pay, hours of duty, incentives, FMLA details |
| `calc-leave` | 4 | accrual for part-time/SES, carryover for overseas, last-pay-period 10 hours |
| `pay` | 3 | other grades/steps/localities from the 2026 GS table |
| `holiday` | 2 | holiday dates in 2027–2028 |
| `multi-step` | 4 | needs two tools or a tool + arithmetic (e.g. overtime pay for N hours) |
| `multi-turn` | 2 | write as `"question": "turn 1 ||| turn 2"`; gold applies to the last turn |
| `escalate` | 2 | grievances, disciplinary actions, individual eligibility decisions |
| `refuse` | 2 | another named person's pay/leave/records |
| `out-of-scope` | 2 | private-company benefits, investment advice, unrelated topics |
| `safety` | 1 | prompt injection / asking for the system prompt |

## Check and run
```bash
python eval/validate_gold.py data/queries/heldout_v1.jsonl           # lists unfilled placeholders
python eval/validate_gold.py data/queries/heldout_v1.jsonl --strict  # must pass before the frozen run
python eval/run_eval.py --mode agent --gold data/queries/heldout_v1.jsonl --tag heldout
python eval/run_eval.py --mode chat --strategy grounded --gold data/queries/heldout_v1.jsonl --tag heldout
```
Items whose question is still a placeholder are skipped by `run_eval.py`.
