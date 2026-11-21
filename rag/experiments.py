"""Retrieval experiments: chunking strategy x retriever x embedding/query variant, scored on an eval set.

Run from the repo root:
    rag/.venv/bin/python rag/experiments.py                       # main grid
    rag/.venv/bin/python rag/experiments.py --only clause-pack-350 --reranker BAAI/bge-reranker-base

Metrics (computed at k=10 retrieval depth):
    sec@k   any of the top-k chunks comes from the gold SECTION       (chunking-independent)
    cl@k    any of the top-k chunks contains the gold CLAUSE           (stricter)
    MRR     1/rank of the first chunk from the gold section (0 if not in top 10)
Writes eval/results/retrieval_<timestamp>.csv and prints a markdown table.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (ROOT, QueryInstructionEmbeddings, build_faiss, chunk_strategies, clause_children,  # noqa: E402
                      dedupe_by_chunk, load_sections, parent_child_retriever, retrievers_for, upgrad_embeddings)

K = 10


def score(results: list[list[dict]], evalset: list[dict]) -> dict:
    agg = {f"sec@{k}": 0.0 for k in (1, 3, 5, 10)} | {f"cl@{k}": 0.0 for k in (1, 5, 10)} | {"MRR": 0.0}
    for hits, q in zip(results, evalset):
        gs, gc = set(q["gold_section_ids"]), set(q["gold_clause_ids"])
        sec_flags = [h["section_id"] in gs for h in hits]
        cl_flags = [bool(gc & set(h["clause_ids"])) for h in hits]
        for k in (1, 3, 5, 10):
            agg[f"sec@{k}"] += any(sec_flags[:k])
        for k in (1, 5, 10):
            agg[f"cl@{k}"] += any(cl_flags[:k])
        agg["MRR"] += next((1 / r for r, f in enumerate(sec_flags, 1) if f), 0.0)
    return {m: round(v / len(evalset), 3) for m, v in agg.items()}


def run(retriever, evalset) -> tuple[list[list[dict]], float]:
    out, t0 = [], time.perf_counter()
    for q in evalset:
        docs = dedupe_by_chunk(retriever.invoke(q["question"]), K)
        out.append([{"section_id": d.metadata["section_id"], "clause_ids": d.metadata.get("clause_ids", [])} for d in docs])
    return out, (time.perf_counter() - t0) * 1000 / len(evalset)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval/data/retrieval_synthetic_v1.jsonl")
    ap.add_argument("--only", nargs="*", help="chunking strategies to run (default: all)")
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--skip-parent-child", action="store_true")
    ap.add_argument("--retrievers", nargs="*", default=["dense", "bm25", "hybrid-rrf", "hybrid+rerank"])
    args = ap.parse_args()

    evalset = [json.loads(l) for l in (ROOT / args.eval).open()]
    sections = load_sections()
    base = upgrad_embeddings()
    variants = {"plain": base, "bge-instr": QueryInstructionEmbeddings(base)}
    rows = []
    print(f"eval: {args.eval} ({len(evalset)} questions)\n")

    for name, fn in chunk_strategies().items():
        if args.only and name not in args.only:
            continue
        docs = fn(sections)
        t0 = time.perf_counter()
        idx_name = "upgrad-bge-small__" + name.replace("/", "_")
        vs = build_faiss(docs, base, idx_name)
        build_s = time.perf_counter() - t0
        for qv, emb in variants.items():
            vs.embedding_function = emb  # same document vectors; only the query embedding changes
            rets = retrievers_for(docs, vs, K, reranker_model=args.reranker)
            for rname in args.retrievers:
                if qv == "bge-instr" and rname == "bm25":
                    continue  # BM25 has no embeddings
                res, ms = run(rets[rname], evalset)
                row = {"chunking": name, "retriever": rname, "query": qv, "chunks": len(docs), **score(res, evalset),
                       "ms/query": round(ms, 1), "index_build_s": round(build_s, 1)}
                rows.append(row)
                print(f"{name:26s} {rname:14s} {qv:9s} sec@1={row['sec@1']:.3f} sec@5={row['sec@5']:.3f} "
                      f"cl@5={row['cl@5']:.3f} MRR={row['MRR']:.3f} {ms:7.1f} ms/q", flush=True)

    if not args.skip_parent_child and (not args.only or "parent-child" in args.only):
        children, parents = clause_children(sections)
        for qv, emb in (("plain", base), ("bge-instr", QueryInstructionEmbeddings(base))):
            pc = parent_child_retriever(children, parents, emb, K, name="upgrad-bge-small__parent-child-children")
            res, ms = run(pc, evalset)
            row = {"chunking": "parent-child (clause->section)", "retriever": "dense", "query": qv,
                   "chunks": len(children), **score(res, evalset), "ms/query": round(ms, 1), "index_build_s": 0}
            rows.append(row)
            print(f"{'parent-child':26s} {'dense':14s} {qv:9s} sec@1={row['sec@1']:.3f} sec@5={row['sec@5']:.3f} "
                  f"cl@5={row['cl@5']:.3f} MRR={row['MRR']:.3f} {ms:7.1f} ms/q", flush=True)

    out = ROOT / "eval" / "results" / f"retrieval_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
