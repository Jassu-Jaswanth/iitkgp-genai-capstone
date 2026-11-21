"""Client for the upGrad OpenAI-compatible chat endpoint.

- One request at a time per process (the group key allows 1 active request; a second gets HTTP 429).
- Retries 429/503 and network errors with 2, 4, 8 s backoff, as the upGrad learner guide asks.
- Records token usage and latency for every call (needed for cost/latency per query in evaluation).
"""

import asyncio
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    content: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    attempts: int
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


class LLMClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.s = settings
        verify: ssl.SSLContext | bool = True
        if transport is None:
            verify = ssl.create_default_context(cafile=settings.ca_path(settings.llm_ca_cert_path))
        self._http = httpx.AsyncClient(
            base_url=settings.llm_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=settings.llm_timeout_s,
            verify=verify,
            transport=transport,
        )
        self._gate = asyncio.Semaphore(settings.llm_max_concurrency)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def aclose(self) -> None:
        await self._http.aclose()

    async def models(self) -> list[str]:
        r = await self._http.get("/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        body: dict[str, Any] = {
            "model": self.s.llm_model,
            "messages": messages,
            "max_tokens": min(max_tokens or self.s.llm_default_max_tokens, 1024),  # endpoint cap: 1024
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},  # Qwen3: no <think> block
        }
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        backoff = [2, 4, 8][: self.s.llm_max_retries]
        last_error = ""
        async with self._gate:
            start = time.perf_counter()
            for attempt in range(len(backoff) + 1):
                try:
                    r = await self._http.post("/chat/completions", json=body)
                except httpx.TransportError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                else:
                    if r.status_code == 200:
                        return self._parse(r.json(), start, attempt + 1)
                    if r.status_code not in (429, 503):
                        raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                    last_error = f"HTTP {r.status_code}"
                if attempt < len(backoff):
                    await asyncio.sleep(backoff[attempt])
        raise LLMError(f"gave up after {len(backoff) + 1} attempts ({last_error})")

    def _parse(self, data: dict[str, Any], start: float, attempts: int) -> LLMResult:
        choice = data["choices"][0]
        msg = choice.get("message", {})
        content = msg.get("content")
        if content:
            content = _THINK.sub("", content).strip()
        usage = data.get("usage") or {}
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct
        return LLMResult(
            content=content,
            tool_calls=msg.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            attempts=attempts,
            raw=data,
        )
