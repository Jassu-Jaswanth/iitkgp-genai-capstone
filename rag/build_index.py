"""Build chunk + dense index over data/corpus/clauses.jsonl for one or more chunking strategies.

    rag/.venv/Scripts/python rag/build_index.py                       # default: clause-pack-350
    rag/.venv/Scripts/python rag/build_index.py --strategy all
    rag/.venv/Scripts/python rag/build_index.py --embedder api        # use the upGrad endpoint (needs .env)
Writes data/index/<strategy>/{chunks.jsonl, dense.npy, meta.json} and prints chunk statistics.
"""

import argparse
import json
import sys
import time

import numpy as np

from chunking import EMBED_MAX_TOKENS, STRATEGIES, load
from retriever import INDEX, get_embedder, safe_name


def stats(chunks):
    t = sorted(c["tokens"] for c in chunks)
    return dict(chunks=len(chunks), median_tok=t[len(t) // 2], p95_tok=t[int(len(t) * 0.95)], max_tok=t[-1],
                over_512=sum(x > EMBED_MAX_TOKENS for x in t))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="clause-pack-350", help=f"one of {list(STRATEGIES)} or 'all'")
    ap.add_argument("--embedder", default="api", choices=["local", "api"],
                    help="api = upGrad endpoint (default, needs .env); local = same model via fastembed on CPU")
    ap.add_argument("--index-name", default=None, help="folder name under data/index (default: strategy name)")
    a = ap.parse_args()
    if a.index_name and a.strategy == "all":
        ap.error("--index-name needs a single --strategy")
    clauses = load()  # data/corpus/clauses.jsonl
    names = list(STRATEGIES) if a.strategy == "all" else [a.strategy]
    emb = get_embedder(a.embedder)
    report = {}
    for name in names:
        t0 = time.time()
        chunks = STRATEGIES[name](clauses)
        st = stats(chunks)
        vecs = emb.passages([c["text"] for c in chunks])
        d = INDEX / safe_name(a.index_name or name)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        np.save(d / "dense.npy", vecs)
        meta = dict(strategy=a.index_name or name, chunking=name, embedder=emb.name, embedder_kind=a.embedder, dim=int(vecs.shape[1]),
                    built=time.strftime("%Y-%m-%d %H:%M"), build_s=round(time.time() - t0, 1), **st)
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        report[name] = meta
        print(f"{name:<24} {st}  ({meta['build_s']} s)", flush=True)
    rep_file = INDEX / "chunking_report.json"
    old = json.loads(rep_file.read_text(encoding="utf-8")) if rep_file.exists() else {}
    rep_file.write_text(json.dumps({**old, **{m["strategy"]: m for m in report.values()}}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
