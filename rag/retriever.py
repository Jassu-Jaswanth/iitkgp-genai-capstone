"""Hybrid retriever: BM25 + dense (bge-small-en-v1.5), fused with Reciprocal Rank Fusion, then cross-encoder rerank.

Index files (built by rag/build_index.py): data/index/<strategy>/{chunks.jsonl, dense.npy, meta.json}
Dense embedder:
  EMBEDDER=local (default) -> fastembed ONNX bge-small-en-v1.5 on CPU (same weights as the upGrad endpoint)
  EMBEDDER=api             -> upGrad embedding endpoint (EMBEDDING_BASE_URL / EMBEDDING_API_KEY / CA cert)
"""

import json
import os
import re
import time

# Public models only: don't send a locally stored (possibly expired) HF token, which turns public downloads into 401s
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "index"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
STOP = set("a an the of to in for on and or is are be by with as at this that from it its any an i my me "
           "do does can how what when which who whom whose will would should may".split())


def safe_name(strategy: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", strategy)


def bm25_tokens(text: str) -> list[str]:
    # keep section numbers like 630.202 and 6303 as tokens
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in STOP]


# ------------------------------------------------------------------ embedders

class LocalEmbedder:
    name = "local:fastembed/bge-small-en-v1.5"

    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def passages(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.embed(texts, batch_size=32)), dtype=np.float32)

    def query(self, text: str) -> np.ndarray:
        return np.array(list(self.model.embed([BGE_QUERY_PREFIX + text])), dtype=np.float32)[0]


class ApiEmbedder:
    """upGrad endpoint: <=16 texts per request, <=512 tokens per text."""
    name = "api:bge-small-en-v1.5"

    def __init__(self):
        import httpx
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        ca = os.environ.get("EMBEDDING_CA_CERT_PATH", "")
        self.url = os.environ["EMBEDDING_BASE_URL"].rstrip("/") + "/embeddings"
        self.model = os.environ.get("EMBEDDING_MODEL", "bge-small-en-v1.5")
        self.client = httpx.Client(timeout=60, verify=str(ROOT / ca) if ca else True,
                                   headers={"Authorization": f"Bearer {os.environ['EMBEDDING_API_KEY']}"})

    def _embed(self, texts):
        for attempt in range(5):
            r = self.client.post(self.url, json={"model": self.model, "input": texts})
            if r.status_code in (429, 502, 503):
                time.sleep(2 ** (attempt + 1))
                continue
            r.raise_for_status()
            return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        r.raise_for_status()

    def passages(self, texts):
        from chunking import cap_tokens  # the endpoint returns 422 above 512 tokens instead of truncating
        texts = [cap_tokens(t, 510) for t in texts]
        out = []
        for i in range(0, len(texts), 16):
            out += self._embed(texts[i:i + 16])
        v = np.array(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def query(self, text):
        try:
            v = np.array(self._embed([BGE_QUERY_PREFIX + text])[0], dtype=np.float32)
            self.last_source = "api"
        except Exception as exc:  # endpoint down/busy -> same model locally (verified cosine 1.0 vs the API)
            print(f"embedding API failed ({type(exc).__name__}: {exc}); using local bge-small", flush=True)
            v = get_embedder("local").query(text)
            self.last_source = "local-fallback"
        return v / np.linalg.norm(v)


@lru_cache(maxsize=2)
def get_embedder(kind: str | None = None):
    """Default: the upGrad embedding endpoint (EMBEDDER=api). EMBEDDER=local runs the same model on CPU."""
    kind = kind or os.environ.get("EMBEDDER", "api")
    return ApiEmbedder() if kind == "api" else LocalEmbedder()


@lru_cache(maxsize=2)
def get_reranker(model: str | None = None):
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model or os.environ.get("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))


# ------------------------------------------------------------------ retriever

class HybridRetriever:
    def __init__(self, strategy: str = "clause-pack-350"):
        d = INDEX / safe_name(strategy)
        if not d.exists():
            raise FileNotFoundError(f"No index at {d}. Run: rag/build_index.py --strategy {strategy}")
        with (d / "chunks.jsonl").open(encoding="utf-8") as fh:
            self.chunks = [json.loads(line) for line in fh]
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.dense = np.load(d / "dense.npy")
        self.bm25 = BM25Okapi([bm25_tokens(c["text"]) for c in self.chunks])
        self.strategy = strategy

    # individual retrievers -------------------------------------------------
    def bm25_search(self, query: str, n: int) -> list[tuple[int, float]]:
        s = self.bm25.get_scores(bm25_tokens(query))
        idx = np.argsort(-s)[:n]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]

    def dense_search(self, query: str, n: int) -> list[tuple[int, float]]:
        q = get_embedder(self.meta.get("embedder_kind")).query(query)
        s = self.dense @ q
        idx = np.argsort(-s)[:n]
        return [(int(i), float(s[i])) for i in idx]

    @staticmethod
    def rrf(lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
        score: dict[int, float] = {}
        for lst in lists:
            for rank, (i, _) in enumerate(lst):
                score[i] = score.get(i, 0.0) + 1.0 / (k + rank + 1)
        return sorted(score.items(), key=lambda x: -x[1])

    def rerank(self, query: str, cands: list[tuple[int, float]], n: int) -> list[tuple[int, float]]:
        if not cands:
            return []
        scores = list(get_reranker().rerank(query, [self.chunks[i]["text"] for i, _ in cands]))
        order = np.argsort(-np.array(scores))[:n]
        return [(cands[j][0], float(scores[j])) for j in order]

    # main entry -------------------------------------------------------------
    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank", n_candidates: int = 30,
               rerank_pool: int = 20, filters: dict | None = None, timings: dict | None = None) -> list[dict]:
        t = timings if timings is not None else {}
        t0 = time.perf_counter()
        if mode == "bm25":
            hits = self.bm25_search(query, max(k, n_candidates))
        elif mode == "dense":
            hits = self.dense_search(query, max(k, n_candidates))
        else:
            b = self.bm25_search(query, n_candidates)
            t["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            t1 = time.perf_counter()
            d = self.dense_search(query, n_candidates)
            t["dense_ms"] = round((time.perf_counter() - t1) * 1000, 1)
            hits = self.rrf([b, d])
            t["bm25_candidates"], t["dense_candidates"], t["fused"] = len(b), len(d), len(hits)
            if mode == "hybrid_rerank":
                t2 = time.perf_counter()
                hits = self.rerank(query, hits[:rerank_pool], rerank_pool)
                t["rerank_ms"] = round((time.perf_counter() - t2) * 1000, 1)
        if filters:
            hits = [(i, s) for i, s in hits if all(
                (self.chunks[i].get(f) in v if isinstance(v, list) else self.chunks[i].get(f) == v)
                for f, v in filters.items())]
        t["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        out = []
        for i, s in hits[:k]:
            c = self.chunks[i]
            out.append(dict(chunk_id=c["chunk_id"], text=c["text"], citation=c["citation"],
                            clause_ids=c["clause_ids"], score=round(s, 4), source_url=c.get("source_url"),
                            metadata=dict(section_id=c["section_id"], source_type=c["source_type"],
                                          heading=c.get("heading"), strategy=c["strategy"], tokens=c["tokens"])))
        return out
