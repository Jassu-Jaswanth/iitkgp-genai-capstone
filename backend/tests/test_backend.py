import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import llm as llm_mod
from app.config import Settings, get_settings
from app.llm import LLMClient, LLMError
from app.main import app, extract_citations
from app.prompts import NO_CONTEXT_NOTE, build_messages
from app.retrieval import Chunk


def settings(**kw) -> Settings:
    base = dict(llm_base_url="https://llm.test/v1", llm_api_key="test-key")
    return Settings(_env_file=None, **{**base, **kw})


def ok_body(content="Hello", pt=10, ct=2):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct}}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def fast(_):
        return None
    monkeypatch.setattr(llm_mod.asyncio, "sleep", fast)


# ---------- LLM client ----------

async def test_retries_429_then_succeeds():
    calls = []

    def handler(req: httpx.Request):
        calls.append(req)
        return httpx.Response(429) if len(calls) < 3 else httpx.Response(200, json=ok_body("OK"))

    c = LLMClient(settings(), transport=httpx.MockTransport(handler))
    r = await c.chat([{"role": "user", "content": "hi"}])
    assert r.content == "OK" and r.attempts == 3 and len(calls) == 3
    body = json.loads(calls[0].content)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls[0].headers["authorization"] == "Bearer test-key"


async def test_gives_up_after_retries():
    c = LLMClient(settings(), transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(LLMError, match="4 attempts"):
        await c.chat([{"role": "user", "content": "hi"}])


async def test_non_retryable_error_raises_immediately():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(400, text="bad request")

    c = LLMClient(settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="HTTP 400"):
        await c.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


async def test_strips_think_block_and_caps_max_tokens():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=ok_body("<think>hmm</think>\n42"))

    c = LLMClient(settings(), transport=httpx.MockTransport(handler))
    r = await c.chat([{"role": "user", "content": "q"}], max_tokens=5000)
    assert r.content == "42" and seen["max_tokens"] == 1024


async def test_one_request_at_a_time():
    active = peak = 0

    async def handler(req):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)  # yield to other tasks while "in flight"
        active -= 1
        return httpx.Response(200, json=ok_body())

    c = LLMClient(settings(), transport=httpx.MockTransport(handler))
    await asyncio.gather(*(c.chat([{"role": "user", "content": str(i)}]) for i in range(5)))
    assert peak == 1 and c.calls == 5


# ---------- prompts ----------

def chunk(i, text="Annual leave accrues at 6 hours per pay period.", cite=None):
    return Chunk(chunk_id=f"c{i}", text=text, citation=cite or f"5 U.S.C. 630{i}", clause_ids=[f"cl{i}"])


def test_build_messages_includes_citations_and_question():
    msgs, used = build_messages("How much leave?", [chunk(1), chunk(2)], None, context_tokens=8192, max_answer_tokens=512)
    user = msgs[-1]["content"]
    assert "[5 U.S.C. 6301]" in user and "Question: How much leave?" in user and len(used) == 2


def test_build_messages_respects_budget():
    big = [chunk(i, text="x" * 4000) for i in range(10)]  # ~1000 tokens each
    msgs, used = build_messages("q", big, None, context_tokens=8192, max_answer_tokens=1024)
    assert 0 < len(used) < 10


def test_build_messages_without_chunks():
    msgs, used = build_messages("q", [], None, context_tokens=8192, max_answer_tokens=512)
    assert NO_CONTEXT_NOTE in msgs[-1]["content"] and used == []


def test_extract_citations_drops_invented():
    used = [chunk(3, cite="5 U.S.C. 6303(a)")]
    ans = "Six hours [5 U.S.C. 6303(a)], see also [5 CFR 999.1] and [5 U.S.C. 6303(a)]."
    assert extract_citations(ans, used) == ["5 U.S.C. 6303(a)"]


def test_extract_citations_accepts_finer_clause_only():
    used = [chunk(3, cite="5 U.S.C. 6303(a)")]
    ans = "Three-fourths day [5 U.S.C. 6303(a)(2)]; not [5 U.S.C. 6303(ab)] or [5 U.S.C. 6303(b)]."
    assert extract_citations(ans, used) == ["5 U.S.C. 6303(a)(2)"]


# ---------- API ----------

class FakeLLM:
    def __init__(self):
        self.messages = None

    async def models(self):
        return ["Qwen/Qwen3-8B"]

    async def chat(self, messages, **kw):
        self.messages = messages
        return llm_mod.LLMResult(content="Six hours per pay period [5 U.S.C. 6303(a)].", tool_calls=[],
                                 finish_reason="stop", prompt_tokens=100, completion_tokens=12,
                                 latency_ms=5.0, attempts=1)

    async def aclose(self):
        pass


class FakeRetriever:
    name = "fake"

    async def retrieve(self, query, k=5, filters=None):
        return [chunk(3, cite="5 U.S.C. 6303(a)")]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.main.get_settings", lambda: settings())
    with TestClient(app) as c:
        c.app.state.llm = FakeLLM()
        c.app.state.retriever = FakeRetriever()
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["retriever"] == "fake"


def test_chat_returns_grounded_answer(client):
    r = client.post("/chat", json={"question": "How much annual leave with 4 years of service?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] == ["5 U.S.C. 6303(a)"] and body["grounded"] is True
    assert body["usage"]["completion_tokens"] == 12 and "total" in body["latency_ms"]


def test_chat_validates_input(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422


# ---------- live (opt-in: pytest -m live) ----------

@pytest.mark.live
async def test_live_endpoint_roundtrip():
    get_settings.cache_clear()
    c = LLMClient(get_settings())
    try:
        r = await c.chat([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=10)
        assert "OK" in (r.content or "") and r.prompt_tokens > 0
    finally:
        await c.aclose()
