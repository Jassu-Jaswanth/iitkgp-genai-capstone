"""Validate a gold/held-out eval file.

    python eval/validate_gold.py eval/data/heldout_v1.jsonl            # report problems + unfilled placeholders
    python eval/validate_gold.py eval/data/heldout_v1.jsonl --strict   # also fail on any remaining <FILL ...>

Stdlib only.
"""

import argparse
import json
import sys
from collections import Counter

CATEGORIES = {"policy", "calc-leave", "pay", "holiday", "multi-step", "multi-turn", "escalate", "refuse",
              "out-of-scope", "safety"}
BEHAVIORS = {"answer", "refuse", "escalate", "not_covered"}
TOOLS = {"search_policy", "leave_accrual_calc", "pay_lookup", "holiday_calendar", "escalate_to_hr"}
REQUIRED = ["qid", "category", "question", "gold_answer", "must_contain", "gold_citations", "expected_tools",
            "expected_behavior"]
PH = "<FILL"


def placeholders(obj) -> int:
    if isinstance(obj, str):
        return int(obj.startswith(PH) or f"{PH}" in obj)
    if isinstance(obj, list):
        return sum(placeholders(x) for x in obj)
    if isinstance(obj, dict):
        return sum(placeholders(v) for v in obj.values())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--strict", action="store_true", help="fail while any <FILL ...> placeholder remains")
    args = ap.parse_args()

    errors, unfilled = [], Counter()
    qids, questions = Counter(), Counter()
    rows = []
    for n, line in enumerate(open(args.path), 1):
        try:
            r = json.loads(line)
        except ValueError as exc:
            errors.append(f"line {n}: invalid JSON ({exc})")
            continue
        rows.append(r)
        qid = r.get("qid", f"line{n}")
        qids[qid] += 1
        questions[r.get("question", "").strip().lower()] += 1
        for k in REQUIRED:
            if k not in r:
                errors.append(f"{qid}: missing field '{k}'")
        if r.get("category") not in CATEGORIES:
            errors.append(f"{qid}: unknown category {r.get('category')!r}")
        if r.get("expected_behavior") not in BEHAVIORS:
            errors.append(f"{qid}: expected_behavior must be one of {sorted(BEHAVIORS)}")
        mc = r.get("must_contain")
        if not (isinstance(mc, list) and mc and all(isinstance(g, list) and g for g in mc)):
            errors.append(f"{qid}: must_contain must be a non-empty list of non-empty lists")
        for t in r.get("expected_tools", []):
            if not t.startswith(PH) and t not in TOOLS:
                errors.append(f"{qid}: unknown tool {t!r}")
        if r.get("category") in ("policy", "calc-leave") and not r.get("gold_citations"):
            errors.append(f"{qid}: policy items need gold_citations (the source you verified)")
        if r.get("category") == "multi-turn" and "|||" not in r.get("question", ""):
            errors.append(f"{qid}: multi-turn questions use 'turn 1 ||| turn 2'")
        p = placeholders({k: v for k, v in r.items() if k != "gold_clause_ids"})
        if p:
            unfilled[qid] = p
    for q, c in qids.items():
        if c > 1:
            errors.append(f"duplicate qid {q}")
    for q, c in questions.items():
        if c > 1 and not q.startswith(PH.lower()):
            errors.append(f"duplicate question: {q[:60]!r}")

    cats = Counter(r.get("category") for r in rows)
    filled = len(rows) - len(unfilled)
    print(f"{args.path}: {len(rows)} items, {filled} filled, {len(unfilled)} with placeholders")
    print("categories:", dict(cats))
    retrieval_ph = sum(1 for r in rows if any(str(x).startswith(PH) for x in r.get("gold_clause_ids", [])))
    if retrieval_ph:
        print(f"note: {retrieval_ph} items keep the gold_clause_ids retrieval placeholder (OK until the data team's "
              "corpus IDs exist; recall falls back to gold_citations)")
    for e in errors:
        print("ERROR", e)
    if unfilled:
        print("unfilled:", ", ".join(sorted(unfilled)))
    bad = bool(errors) or (args.strict and bool(unfilled))
    print("FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
