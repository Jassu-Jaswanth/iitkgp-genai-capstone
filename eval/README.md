# eval/: grading the agent

| File | Job |
|---|---|
| `run_eval.py` | asks every question, grades the answers, writes `results/` |
| `validate_gold.py` | checks a question file; `--strict` fails while `<FILL>` is left |

The questions live with the dataset in `data/queries/` (see `docs/DATASET.md`): `gold_v1.jsonl` (50 verified,
dev set), `gold_example.jsonl` (5, the format), `heldout_v1.jsonl` (40-item template; read `HELDOUT.md`).

```bash
python eval/run_eval.py --mode agent                                  # golden set (default)
python eval/run_eval.py --mode chat --strategy grounded               # grounded | few_shot_json | cot | decompose
python eval/run_eval.py --mode agent --gold data/queries/heldout_v1.jsonl --tag heldout
```
**Grades per question:** correct facts, right behaviour (answer / refuse / escalate / not covered), citation present,
right source retrieved, tools used, tokens, time, and why it failed. The golden set is a dev set (we tuned on it):
report the **held-out** score.
