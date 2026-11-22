"""Caching. Nothing to set up: both caches are on by default and live where the app already keeps its data.

1. Search cache (in memory, per backend process): the same search query returns the same chunks, so the
   BM25 + embedding + rerank work (and the embedding API call) is done once. LRU, `SEARCH_CACHE_SIZE` entries.
2. Answer cache (Postgres table chat.answer_cache, or in memory without DATABASE_URL): the FIRST question of a
   new conversation, asked again word for word (case/spacing ignored), is answered from the cache instead of
   ~4 calls to the shared 1-request-at-a-time model. Expires after `ANSWER_CACHE_TTL_S` (0 turns it off).
   Never cached: follow-ups (they depend on the conversation), answers that drafted an HR case (consequential
   action), and anything that did not complete. The key includes a version hash of the prompts, tools,
   retriever and model, so changing any of them starts a fresh cache automatically.
"""

import hashlib
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Hashable

NO_CACHE_TOOLS = {"escalate_to_hr"}


class LRUCache:
    """Small thread-safe LRU (the search runs in a worker thread)."""

    def __init__(self, size: int = 512):
        self.size = size
        self._d: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            self.misses += 1
            return None

    def put(self, key: Hashable, value: Any) -> None:
        if self.size <= 0:
            return
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._d), "hits": self.hits, "misses": self.misses}


def search_key(query: str, k: int | None, filters: dict | None) -> tuple:
    return (" ".join(query.lower().split()), k, tuple(sorted((filters or {}).items(), key=str)))


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower()).rstrip("?!. ")


def version_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def answer_key(version: str, question: str) -> str:
    return hashlib.sha256(f"{version}\x1f{normalize_question(question)}".encode()).hexdigest()


def cacheable(resp: dict[str, Any]) -> bool:
    if resp.get("status") != "completed" or not resp.get("answer"):
        return False
    return not any(t.get("tool") in NO_CACHE_TOOLS for t in resp.get("trace", []) if t.get("type") == "tool_call")


class MemoryAnswerCache:
    def __init__(self, ttl_s: int, size: int = 1000):
        self.ttl_s = ttl_s
        self._lru = LRUCache(size)

    async def get(self, key: str) -> dict | None:
        hit = self._lru.get(key)
        if hit is None or time.time() - hit[0] > self.ttl_s:
            return None
        return hit[1]

    async def put(self, key: str, entry: dict) -> None:
        self._lru.put(key, (time.time(), entry))

    async def clear(self) -> int:
        n = len(self._lru._d)
        self._lru._d.clear()
        return n
