# eval/: grading the agent

| File | Job |
|---|---|
| `run_eval.py` | asks every question, grades the answers, writes `results/` |
| `validate_gold.py` | checks a question file; `--strict` fails while `<FILL>` is left |
| `data/gold_example.jsonl` | 5 example questions (the format) |
| `data/heldout_v1.jsonl` | **40-question template for the team to fill**: read `HELDOUT.md` |

```bash
python eval/run_eval.py --mode agent --gold eval/data/gold_example.jsonl
python eval/run_eval.py --mode chat --strategy grounded --gold eval/data/gold_example.jsonl
```
Grades per question: correct facts, right behaviour (answer / refuse / escalate / not covered), citation present,
right source retrieved, tools used, tokens, time, and why it failed.
