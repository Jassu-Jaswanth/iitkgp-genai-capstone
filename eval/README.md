# eval/: grading the agent

| File | Job |
|---|---|
| `run_eval.py` | asks every question, grades the answers, writes `results/` |
| `validate_gold.py` | checks a question file; `--strict` fails while `<FILL>` is left |
| `data/gold_v1.jsonl` | **the golden set: 50 questions with verified answers** (dev set, see below) |
| `data/gold_example.jsonl` | 5 example questions (the format) |
| `data/heldout_v1.jsonl` | **40-question template for the team to fill**: read `HELDOUT.md` |

## The golden set (`gold_v1.jsonl`)
50 real questions, each answer checked by hand against the law, regulation, OPM fact sheet or pay table.
Each item says what a correct answer must contain, which source holds it, which tools should be used, and how the
agent should behave (answer / refuse / escalate / not covered).

| Kind | Items | Passes on main today? |
|---|---|---|
| needs only `search_policy` | 31 | after the RAG service is plugged in |
| guardrails (privacy, off-topic, injection) | 6 | yes |
| needs a TODO tool (leave calc, pay, holidays, escalation) | 13 | after that tool is built |

It is a **dev set**: we tuned the agent on it (full build scored 50/50), so report the **held-out** score instead.

```bash
python eval/run_eval.py --mode agent --gold eval/data/gold_v1.jsonl
python eval/run_eval.py --mode agent --gold eval/data/gold_example.jsonl
python eval/run_eval.py --mode chat --strategy grounded --gold eval/data/gold_example.jsonl
```
Grades per question: correct facts, right behaviour (answer / refuse / escalate / not covered), citation present,
right source retrieved, tools used, tokens, time, and why it failed.
