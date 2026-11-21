"""Generate a synthetic RETRIEVAL eval set: one question per sampled clause, gold = that clause/section.

Run from the repo root:  rag/.venv/bin/python rag/make_retrieval_eval.py --n 150
Writes data/queries/retrieval_synthetic_v1.jsonl. Uses the upGrad Qwen3-8B endpoint (1 request at a time).

Known bias (report): questions are generated FROM the clause, so they share vocabulary with it, which favours
BM25/lexical retrieval. The prompt asks for everyday wording to reduce this; the hand-written gold set
(data/queries/gold_*.jsonl) is the unbiased check.
"""

import argparse
import json
import random
import ssl
import sys
import time
from pathlib import Path

import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import ROOT, _env, clause_text, load_sections  # noqa: E402

PROMPT = """You write test questions for an HR policy search engine for US federal employees.
Passage (from {citation}, "{heading}"):
\"\"\"{text}\"\"\"

Write ONE question that a federal employee or HR specialist might realistically ask, which this passage answers.
Rules: use everyday wording; do not copy phrases of more than 3 words from the passage; no section numbers or
citations; one sentence; it must be answerable from this passage alone."""


class Q(BaseModel):
    question: str = Field(description="The test question")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/queries/retrieval_synthetic_v1.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pools: dict[str, list] = {"regulation": [], "statute": [], "guidance": []}
    for s in load_sections():
        for c in s.clauses:
            t = clause_text(c)
            if 25 <= len(t.split()) <= 250 and "definitions" not in s.heading.lower():
                pools[s.source_type].append((s, c, t))
    quota = {"regulation": round(args.n * 0.4), "statute": round(args.n * 0.2)}
    quota["guidance"] = args.n - sum(quota.values())
    picks = [x for kind, q in quota.items() for x in rng.sample(pools[kind], min(q, len(pools[kind])))]
    rng.shuffle(picks)
    print({k: len(v) for k, v in pools.items()}, "->", quota)

    env = _env()
    ctx = ssl.create_default_context(cafile=str((ROOT / env["LLM_CA_CERT_PATH"]).resolve()))
    llm = ChatOpenAI(model=env.get("LLM_MODEL", "Qwen/Qwen3-8B"), base_url=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"],
                     temperature=0.7, max_tokens=120, max_retries=5, http_client=httpx.Client(verify=ctx),
                     extra_body={"chat_template_kwargs": {"enable_thinking": False}}).with_structured_output(Q, method="json_schema")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n = time.time(), 0
    with out.open("w") as fh:
        for i, (s, c, t) in enumerate(picks, 1):
            try:
                q = llm.invoke(PROMPT.format(citation=c["citation"], heading=s.heading, text=t[:1500])).question.strip()
            except Exception as exc:
                print(f"  skip {c['clause_id']}: {type(exc).__name__}")
                continue
            fh.write(json.dumps({"qid": f"rs-{i:03d}", "question": q, "gold_clause_ids": [c["clause_id"]],
                                 "gold_section_ids": [s.section_id], "source_type": s.source_type,
                                 "citation": c["citation"], "heading": s.heading, "passage": t}) + "\n")
            n += 1
            if i % 25 == 0:
                print(f"  {i}/{len(picks)} ({time.time() - t0:.0f}s)")
    print(f"{n} questions -> {out.relative_to(ROOT)} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
