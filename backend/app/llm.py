"""Chat model for the upGrad endpoint, via LangChain's ChatOpenAI (OpenAI-compatible vLLM, Qwen3-8B).

Library features used instead of custom code:
- retries with exponential backoff on 429/5xx: openai SDK `max_retries`
- request pacing (60 requests/min per key): LangChain `InMemoryRateLimiter`
- token usage: `AIMessage.usage_metadata`
The endpoint allows ONE active request per key; callers run one graph/chain at a time (see main.py gate).
"""

import ssl
from functools import lru_cache

import httpx
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from .config import Settings


@lru_cache
def _rate_limiter() -> InMemoryRateLimiter:
    # upGrad guide: 60 requests per minute per key.
    return InMemoryRateLimiter(requests_per_second=1.0, check_every_n_seconds=0.1, max_bucket_size=5)


def build_chat_model(settings: Settings, *, max_tokens: int | None = None, temperature: float = 0.0) -> ChatOpenAI:
    ctx = ssl.create_default_context(cafile=settings.ca_path(settings.llm_ca_cert_path))
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=temperature,
        max_tokens=min(max_tokens or settings.llm_default_max_tokens, 1024),  # endpoint cap: 1024
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        rate_limiter=_rate_limiter(),
        http_client=httpx.Client(verify=ctx),
        http_async_client=httpx.AsyncClient(verify=ctx),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # Qwen3: no <think> block
    )
