"""Retrieval experiments: recall@k measured separately from end-to-end answer quality.

    rag/.venv/Scripts/python rag/experiments.py                                  # all built indexes, all modes
    rag/.venv/Scripts/python rag/experiments.py --indexes clause-pack-256 --modes bm25 hybrid_rerank

Query sets (data/queries/, see docs/DATASET.md): gold_v1 (dev set) and heldout_v1 once its <FILL> slots are written.
Only questions with gold_clause_ids are scored (guardrail questions have none).

Metrics per (index, retriever, query set):
  sec@k   share of questions with a gold SECTION in the top k chunks
  cl@k    share of gold CLAUSES found in the top k chunks
  fact@k  share of questions whose top-k chunk text contains every must_contain fact group. Label-free: credits
          the same rule found in another source (statute vs regulation vs OPM fact sheet), which the dataset card
          lists as a limitation of clause labels ("only one source is labelled per question").
  MRR     mean reciprocal rank of the first chunk holding a gold clause or section
Writes eval/results/retrieval_<timestamp>.csv and a per-query JSONL for error analysis.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from retriever import INDEX, HybridRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "data" / "queries"
RESULTS = ROOT / "eval" / "results"
KS = (1, 3, 5, 10)


def load_set(path: Path, clause_to_section: dict[str, str]) -> list[dict]:
    items = []
    for r in map(json.loads, path.open(encoding="utf-8")):
        if "<FILL" in json.dumps(r) or not r.get("gold_clause_ids"):  # skip unwritten template slots
            continue
        cl = set(r["gold_clause_ids"])
        items.append(dict(qid=r["qid"], question=r["question"], clauses=cl,
                          sections={clause_to_section.get(c, c.rsplit("-", 1)[0]) for c in cl},
                          must=r.get("must_contain") or []))
    return items


def has_facts(text: str, groups: list[list[str]]) -> bool:
    t = text.lower()
    return bool(groups) and all(any(alt.lower() in t for alt in g) for g in groups)


def score(results: list[dict], q: dict) -> dict:
    m, first = {}, None
    for k in KS:
        top = results[:k]
        found_cl = set().union(*[set(c["clause_ids"]) for c in top]) if top else set()
        m[f"sec@{k}"] = float(bool({c["metadata"]["section_id"] for c in top} & q["sections"]))
        m[f"cl@{k}"] = len(found_cl & q["clauses"]) / len(q["clauses"])
        m[f"fact@{k}"] = float(has_facts("\n".join(c["text"] for c in top), q["must"]))
    for rank, c in enumerate(results, 1):
        if set(c["clause_ids"]) & q["clauses"] or c["metadata"]["section_id"] in q["sections"]:
            first = rank
            break
    m["MRR"] = 1.0 / first if first else 0.0
    m["first_hit_rank"] = first
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indexes", nargs="*", default=None, help="folders under data/index (default: all built)")
    ap.add_argument("--modes", nargs="*", default=["bm25", "dense", "hybrid", "hybrid_rerank"])
    ap.add_argument("--sets", nargs="*", default=["gold_v1", "heldout_v1"])
    a = ap.parse_args()

    indexes = a.indexes or sorted(p.name for p in INDEX.iterdir() if (p / "meta.json").exists())
    clause_to_section = {}
    with (ROOT / "data" / "corpus" / "clauses.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            clause_to_section[c["clause_id"]] = c["section_id"]
    sets = {}
    for name in a.sets:
        f = QUERIES / f"{name}.jsonl"
        if f.exists() and (items := load_set(f, clause_to_section)):
            sets[name] = items
    if not sets:
        sys.exit("No scorable query set found in data/queries/")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows, per_query = [], []
    keys = [f"{m}@{k}" for m in ("sec", "cl", "fact") for k in KS] + ["MRR"]
    for idx in indexes:
        r = HybridRetriever(idx)
        for mode in a.modes:
            for set_name, qs in sets.items():
                agg = dict.fromkeys(keys, 0.0)
                t0 = time.perf_counter()
                for q in qs:
                    res = r.search(q["question"], k=max(KS), mode=mode)
                    m = score(res, q)
                    for key in keys:
                        agg[key] += m[key]
                    per_query.append(dict(index=idx, retriever=mode, query=set_name, qid=q["qid"],
                                          question=q["question"], first_hit_rank=m["first_hit_rank"],
                                          fact_at_5=m["fact@5"], gold_clauses=sorted(q["clauses"]),
                                          top5=[c["citation"] for c in res[:5]]))
                ms = (time.perf_counter() - t0) * 1000 / len(qs)
                row = dict(chunking=r.meta.get("chunking", idx), index=idx, embedder=r.meta.get("embedder"),
                           retriever=mode, query=set_name, n=len(qs),
                           **{key: round(v / len(qs), 3) for key, v in agg.items()},
                           **{"ms/query": round(ms, 1), "chunks": len(r.chunks)})
                rows.append(row)
                print(f"{idx:<24} {mode:<14} {set_name:<10} n={len(qs):<3} sec@5={row['sec@5']:.3f} "
                      f"cl@5={row['cl@5']:.3f} fact@5={row['fact@5']:.3f} MRR={row['MRR']:.3f} {ms:7.1f} ms/q",
                      flush=True)

    out = RESULTS / f"retrieval_{stamp}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (RESULTS / f"retrieval_{stamp}.per_query.jsonl").open("w", encoding="utf-8") as fh:
        for p in per_query:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
