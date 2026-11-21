"""End-to-end evaluation against the running backend (Track B part V).

    python eval/run_eval.py --mode agent --gold eval/data/gold_v1.jsonl
    python eval/run_eval.py --mode chat  --gold eval/data/gold_v1.jsonl

Calls POST /chat (single-prompt baseline) or POST /agent, one question at a time (the model key allows one
active request). Writes eval/results/e2e_<mode>_<timestamp>.jsonl (per question) and prints a summary.

Per-question scores:
  correct        every must_contain group matched (case-insensitive, any alternative in the group)
  behavior_ok    answer / refuse / escalate / not_covered as expected
  cited          answer carries at least one verified citation (policy questions only)
  retrieved_gold a gold clause id appeared in retrieved chunks / search_policy results (recall@k)
  tools_ok       every expected tool was called (agent only)
Failure taxonomy (first matching rule), for incorrect answers:
  tool_error        a tool returned an error
  retrieval_miss    gold clauses exist but none were retrieved
  not_covered_wrong answered "not covered" although evidence was retrieved
  hallucination     wrong answer although the gold clause was retrieved / the right tool ran
  reasoning_error   everything else
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
NOT_COVERED = re.compile(r"do(?:es)? not cover|don't cover|not covered|outside (?:the )?scope", re.I)
REFUSAL = re.compile(r"\b(cannot|can't|unable|not able|do not have access|don't have access|privacy|not permitted)\b", re.I)
ESCALATED = re.compile(r"\b(draft|pending|approval|HR specialist|ticket)\b", re.I)


def matched(answer: str, groups: list[list[str]]) -> bool:
    a = answer.lower()
    return all(any(alt.lower() in a for alt in group) for group in groups)


def behavior_of(answer: str, tools_called: list[str]) -> str:
    if "escalate_to_hr" in tools_called or (ESCALATED.search(answer) and "HR" in answer and "draft" in answer.lower()):
        return "escalate"
    if NOT_COVERED.search(answer):
        return "not_covered"
    if REFUSAL.search(answer) and len(answer) < 400:
        return "refuse"
    return "answer"


def retrieved_clause_ids(mode: str, resp: dict) -> set[str]:
    ids: set[str] = set()
    if mode == "chat":
        for c in resp.get("chunks", []):
            ids |= set(c.get("clause_ids", []))
    else:  # search_policy results in the trace (content may be truncated, so regex rather than json.loads)
        for t in resp.get("trace", []):
            if t.get("type") == "tool_result" and t.get("tool") == "search_policy":
                for arr in re.findall(r'"clause_ids":\s*\[([^\]]*)\]', t.get("content", "")):
                    ids |= set(re.findall(r'"([^"]+)"', arr))
    return ids


def evaluate(item: dict, mode: str, resp: dict) -> dict:
    answer = resp.get("answer", "")
    tools_called = [t["tool"] for t in resp.get("trace", []) if t.get("type") == "tool_call"]
    tool_errors = resp.get("tool_errors", 0)
    got_ids = retrieved_clause_ids(mode, resp)
    gold_ids = set(item["gold_clause_ids"])
    retrieved_gold = bool(gold_ids & got_ids) if gold_ids else None
    behavior = behavior_of(answer, tools_called)
    behavior_ok = behavior == item["expected_behavior"] or (item["expected_behavior"] == "not_covered" and behavior == "refuse")
    correct = behavior_ok and (matched(answer, item["must_contain"]) if item["expected_behavior"] == "answer" else True)
    if item["expected_behavior"] in ("refuse", "not_covered", "escalate"):
        correct = behavior_ok and matched(answer, item["must_contain"])
    tools_ok = all(t in tools_called for t in item["expected_tools"] if t != "search_policy") if mode == "agent" else None

    failure = None
    if not correct:
        if tool_errors:
            failure = "tool_error"
        elif gold_ids and not retrieved_gold:
            failure = "retrieval_miss"
        elif behavior == "not_covered" and item["expected_behavior"] == "answer":
            failure = "not_covered_wrong"
        elif retrieved_gold or (mode == "agent" and tools_ok and item["expected_tools"]):
            failure = "hallucination"
        else:
            failure = "reasoning_error"
    return {"qid": item["qid"], "category": item["category"], "question": item["question"], "answer": answer,
            "gold_answer": item["gold_answer"], "correct": correct, "behavior": behavior, "behavior_ok": behavior_ok,
            "cited": bool(resp.get("citations")), "retrieved_gold": retrieved_gold, "tools_called": tools_called,
            "tools_ok": tools_ok, "tool_errors": tool_errors, "failure": failure,
            "prompt_tokens": resp.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": resp.get("usage", {}).get("completion_tokens", 0),
            "llm_calls": resp.get("usage", {}).get("llm_calls", 0),
            "latency_ms": resp.get("latency_ms") if isinstance(resp.get("latency_ms"), (int, float)) else resp.get("latency_ms", {}).get("total"),
            "status": resp.get("status", "completed")}


def summarize(rows: list[dict], mode: str) -> dict:
    n = len(rows)
    pol = [r for r in rows if r["category"] in ("policy", "calc-leave", "multi-step")]
    rg = [r for r in rows if r["retrieved_gold"] is not None]
    fail = {}
    for r in rows:
        if r["failure"]:
            fail[r["failure"]] = fail.get(r["failure"], 0) + 1
    by_cat = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[0] += r["correct"]
        c[1] += 1
    return {"mode": mode, "n": n, "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
            "behavior_ok": round(sum(r["behavior_ok"] for r in rows) / n, 3),
            "completion_rate": round(sum(r["status"] == "completed" for r in rows) / n, 3),
            "cited_rate_policy": round(sum(r["cited"] for r in pol) / max(len(pol), 1), 3),
            "retrieval_recall": round(sum(bool(r["retrieved_gold"]) for r in rg) / max(len(rg), 1), 3),
            "avg_tool_calls": round(statistics.mean(len(r["tools_called"]) for r in rows), 2),
            "avg_llm_calls": round(statistics.mean(r["llm_calls"] for r in rows), 2),
            "avg_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in rows)),
            "avg_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in rows)),
            "p50_latency_ms": round(statistics.median(r["latency_ms"] or 0 for r in rows)),
            "failures": fail, "by_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["chat", "agent"], default="agent")
    ap.add_argument("--gold", default="eval/data/gold_v1.jsonl")
    ap.add_argument("--api", default="http://localhost:8100")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--qids", nargs="*", help="only these question ids")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    items = [json.loads(l) for l in (ROOT / args.gold).open()]
    items = [i for i in items if not args.qids or i["qid"] in args.qids][: args.limit]
    out = ROOT / "eval" / "results" / f"e2e_{args.mode}{('_' + args.tag) if args.tag else ''}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(base_url=args.api, timeout=600) as client, out.open("w") as fh:
        for i, item in enumerate(items, 1):
            r = client.post(f"/{args.mode}", json={"question": item["question"]})
            resp = r.json() if r.status_code == 200 else {"answer": f"HTTP {r.status_code}: {r.text[:200]}", "status": "http_error"}
            row = evaluate(item, args.mode, resp)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            print(f"[{i:02d}/{len(items)}] {'OK ' if row['correct'] else 'BAD'} {item['qid']} {item['category']:12s} "
                  f"{row['failure'] or '':18s} tools={row['tools_called']} {row['latency_ms']}ms", flush=True)
    s = summarize(rows, args.mode)
    (out.with_suffix(".summary.json")).write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    print(f"-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
