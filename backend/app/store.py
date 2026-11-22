"""Where conversations, answers and their LLM traces are kept.

- PostgresStore (DATABASE_URL set): schema `chat` (schema.sql) + LangGraph's AsyncPostgresSaver in schema `agent`,
  so conversations and agent memory survive restarts.
- MemoryStore (no DATABASE_URL): same interface, kept in this process (last 500 turns), for tests and quick runs.
"""

import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cache import MemoryAnswerCache

SCHEMA_SQL = Path(__file__).with_name("schema.sql").read_text()
TURN_FIELDS = ("turn_id", "route", "status", "citations", "critique", "tool_calls", "usage", "latency_ms",
               "cached_from")


def _turn_summary(steps: list[dict]) -> dict:
    return {"llm_calls": sum(s["kind"] == "llm" for s in steps),
            "tools": [s["name"] for s in steps if s["kind"] == "tool"]}


class MemoryStore:
    def __init__(self, max_turns: int = 500, answer_cache_ttl_s: int = 0):
        self.convs: dict[str, dict] = {}
        self.msgs: dict[str, list[dict]] = {}
        self.steps: OrderedDict[str, list[dict]] = OrderedDict()
        self.max_turns = max_turns
        self.answer_cache = MemoryAnswerCache(answer_cache_ttl_s)

    async def checkpointer(self):
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()

    async def close(self) -> None:
        pass

    async def save_turn(self, conversation_id: str, question: str, r: dict, steps: list[dict]) -> None:
        now = datetime.now(timezone.utc)
        c = self.convs.setdefault(conversation_id, {"id": conversation_id, "title": question[:80], "created_at": now})
        c["updated_at"] = now
        self.msgs.setdefault(conversation_id, []).extend([
            {"role": "user", "content": question, "created_at": now},
            {"role": "assistant", "content": r["answer"], "created_at": now, "question": question,
             "conversation_id": conversation_id, **{k: r.get(k) for k in TURN_FIELDS}}])
        self.steps[r["turn_id"]] = steps
        while len(self.steps) > self.max_turns:
            self.steps.popitem(last=False)

    async def conversations(self, limit: int = 50) -> list[dict]:
        cs = sorted(self.convs.values(), key=lambda c: c["updated_at"], reverse=True)[:limit]
        return [{**c, "messages": len(self.msgs.get(c["id"], []))} for c in cs]

    async def messages(self, conversation_id: str) -> list[dict]:
        return [{k: v for k, v in m.items() if k not in ("question", "conversation_id")}
                for m in self.msgs.get(conversation_id, [])]

    async def delete(self, conversation_id: str) -> bool:
        self.msgs.pop(conversation_id, None)
        return self.convs.pop(conversation_id, None) is not None

    async def trace(self, turn_id: str) -> list[dict]:
        return self.steps.get(turn_id, [])

    async def turns(self, limit: int = 100) -> list[dict]:
        rows = [m for ms in self.msgs.values() for m in ms if m["role"] == "assistant"]
        rows.sort(key=lambda m: m["created_at"], reverse=True)
        return [{"created_at": m["created_at"], "conversation_id": m["conversation_id"], "question": m["question"],
                 **{k: m[k] for k in TURN_FIELDS}, **_turn_summary(self.steps.get(m["turn_id"], []))}
                for m in rows[:limit]]


class PostgresAnswerCache:
    """chat.answer_cache; rows older than the TTL are ignored (and cleared on startup)."""

    def __init__(self, store: "PostgresStore", ttl_s: int):
        self.store, self.ttl_s = store, ttl_s

    async def get(self, key: str) -> dict | None:
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute(
                """UPDATE chat.answer_cache SET hits = hits + 1
                   WHERE key = %s AND created_at > now() - make_interval(secs => %s) RETURNING entry""",
                (key, self.ttl_s))).fetchone()
        return row["entry"] if row else None

    async def put(self, key: str, entry: dict) -> None:
        async with self.store.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO chat.answer_cache (key, question, entry) VALUES (%s, %s, %s)
                   ON CONFLICT (key) DO UPDATE SET entry = EXCLUDED.entry, created_at = now(), hits = 0""",
                (key, entry["question"], json.dumps(entry, default=str)))

    async def clear(self) -> int:
        async with self.store.pool.connection() as conn:
            return (await conn.execute("DELETE FROM chat.answer_cache")).rowcount


class PostgresStore:
    def __init__(self, url: str, answer_cache_ttl_s: int = 0):
        self.url = url
        self.pool = self._cp_pool = None
        self.answer_cache = PostgresAnswerCache(self, answer_cache_ttl_s)

    @classmethod
    async def open(cls, url: str, answer_cache_ttl_s: int = 0) -> "PostgresStore":
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
        self = cls(url, answer_cache_ttl_s)
        self.pool = AsyncConnectionPool(url, min_size=1, max_size=5, open=False,
                                        kwargs={"autocommit": True, "row_factory": dict_row})
        await self.pool.open(wait=True, timeout=30)
        async with self.pool.connection() as conn:
            await conn.execute(SCHEMA_SQL)
            await conn.execute("DELETE FROM chat.answer_cache WHERE created_at < now() - make_interval(secs => %s)",
                               (answer_cache_ttl_s,))
        return self

    async def checkpointer(self):
        """LangGraph's own Postgres checkpointer (agent memory), its tables in the `agent` schema."""
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
        self._cp_pool = AsyncConnectionPool(self.url, min_size=1, max_size=5, open=False,
                                            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                                                    "options": "-c search_path=agent"})
        await self._cp_pool.open(wait=True, timeout=30)
        saver = AsyncPostgresSaver(self._cp_pool)
        await saver.setup()
        return saver

    async def close(self) -> None:
        for p in (self.pool, self._cp_pool):
            if p is not None:
                await p.close()

    async def save_turn(self, conversation_id: str, question: str, r: dict, steps: list[dict]) -> None:
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute(
                """INSERT INTO chat.conversations (id, title) VALUES (%s, %s)
                   ON CONFLICT (id) DO UPDATE SET updated_at = now()""", (conversation_id, question[:80]))
            await conn.execute("INSERT INTO chat.messages (conversation_id, role, content) VALUES (%s, 'user', %s)",
                               (conversation_id, question))
            await conn.execute(
                """INSERT INTO chat.messages (conversation_id, role, content, turn_id, route, status, citations,
                                              critique, tool_calls, usage, latency_ms, cached_from)
                   VALUES (%s, 'assistant', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (conversation_id, r["answer"], r["turn_id"], r["route"], r["status"], json.dumps(r["citations"]),
                 json.dumps(r["critique"]), r["tool_calls"], json.dumps(r["usage"]), r["latency_ms"],
                 r.get("cached_from")))
            if steps:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """INSERT INTO chat.trace_steps (turn_id, seq, parent_seq, node, kind, name, input, output,
                                                         prompt_tokens, completion_tokens, started_at, ended_at, error)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        [(r["turn_id"], s["seq"], s["parent_seq"], s["node"], s["kind"], s["name"],
                          json.dumps(s["input"], default=str), json.dumps(s["output"], default=str),
                          s["prompt_tokens"], s["completion_tokens"], s["started_at"], s["ended_at"], s["error"])
                         for s in steps])

    async def _all(self, sql: str, args: tuple) -> list[dict]:
        async with self.pool.connection() as conn:
            return await (await conn.execute(sql, args)).fetchall()

    async def conversations(self, limit: int = 50) -> list[dict]:
        return await self._all(
            """SELECT c.id, c.title, c.created_at, c.updated_at, count(m.id) AS messages
               FROM chat.conversations c LEFT JOIN chat.messages m ON m.conversation_id = c.id
               GROUP BY c.id ORDER BY c.updated_at DESC LIMIT %s""", (limit,))

    async def messages(self, conversation_id: str) -> list[dict]:
        return await self._all(
            f"""SELECT role, content, {", ".join(TURN_FIELDS)}, created_at
                FROM chat.messages WHERE conversation_id = %s ORDER BY id""", (conversation_id,))

    async def delete(self, conversation_id: str) -> bool:
        async with self.pool.connection() as conn:
            return (await conn.execute("DELETE FROM chat.conversations WHERE id = %s", (conversation_id,))).rowcount > 0

    async def trace(self, turn_id: str) -> list[dict]:
        return await self._all(
            """SELECT seq, parent_seq, node, kind, name, input, output, prompt_tokens, completion_tokens,
                      started_at, ended_at, error
               FROM chat.trace_steps WHERE turn_id = %s ORDER BY seq""", (turn_id,))

    async def turns(self, limit: int = 100) -> list[dict]:
        """One row per answer, newest first: the numbers the dashboard shows."""
        return await self._all(
            f"""SELECT a.created_at, a.conversation_id,
                       (SELECT q.content FROM chat.messages q WHERE q.conversation_id = a.conversation_id
                          AND q.role = 'user' AND q.id < a.id ORDER BY q.id DESC LIMIT 1) AS question,
                       {", ".join("a." + f for f in TURN_FIELDS)},
                       (SELECT count(*) FROM chat.trace_steps t WHERE t.turn_id = a.turn_id AND t.kind = 'llm')
                         AS llm_calls,
                       (SELECT coalesce(array_agg(t.name ORDER BY t.seq), '{{}}') FROM chat.trace_steps t
                          WHERE t.turn_id = a.turn_id AND t.kind = 'tool') AS tools
                FROM chat.messages a WHERE a.role = 'assistant' ORDER BY a.id DESC LIMIT %s""", (limit,))


async def open_store(database_url: str, answer_cache_ttl_s: int = 0):
    if database_url:
        return await PostgresStore.open(database_url, answer_cache_ttl_s)
    return MemoryStore(answer_cache_ttl_s=answer_cache_ttl_s)
