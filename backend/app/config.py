"""Settings loaded from the repo-root .env (see .env.example)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    # upGrad Track B LLM endpoint (OpenAI-compatible, vLLM, Qwen3-8B, 8k context)
    llm_base_url: str
    llm_api_key: str
    llm_model: str = "Qwen/Qwen3-8B"
    llm_ca_cert_path: str = "./.context/certs/llm-vm04-ca.crt"
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 3  # upGrad guide: retry 429/503 with 2, 4, 8 s backoff
    llm_max_concurrency: int = 1  # the key allows ONE active request across everyone using it
    llm_context_tokens: int = 8192
    llm_default_max_tokens: int = 512

    # Retrieval component (one retrieve() contract, three implementations):
    #   retrieval_mode = "" (auto: http if rag_service_url else placeholder) | placeholder | http | inprocess
    #   inprocess imports rag/retriever.py (Rajesh's hybrid BM25 + bge-small + rerank) into this process;
    #   needs `uv sync --extra rag`. rag_strategy / rag_search_mode mirror the RAG service's RAG_STRATEGY / RAG_MODE.
    retrieval_mode: str = ""
    rag_service_url: str = ""
    rag_strategy: str = "clause-pack-200"
    rag_search_mode: str = "hybrid_rerank"
    retrieval_top_k: int = 5

    # Storage: Postgres for chat history, LLM traces and agent memory. Empty -> kept in-process (lost on restart).
    database_url: str = ""

    # Caching (app/cache.py): on by default, nothing to set up.
    search_cache_size: int = 512  # identical searches served from memory (0 = off)
    answer_cache_ttl_s: int = 86400  # repeated first questions answered from Postgres for a day (0 = off)

    # Agent
    agent_max_steps: int = 10  # Track B guardrail: agent capped at 10 steps per query
    data_dir: str = "./data"  # the dataset: corpus/, tables/ (tool data), queries/ (eval)

    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    def ca_path(self, value: str) -> str:
        p = Path(value)
        return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


@lru_cache
def get_settings() -> Settings:
    return Settings()
