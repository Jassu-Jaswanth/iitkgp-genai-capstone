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

    # Retrieval (built by the RAG teammate). Empty URL -> placeholder retriever.
    rag_service_url: str = ""
    retrieval_top_k: int = 5

    # Agent
    agent_max_steps: int = 10  # Track B guardrail: agent capped at 10 steps per query
    data_dir: str = "./data/raw"  # pay table + holidays (owned by the data teammate)

    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    def ca_path(self, value: str) -> str:
        p = Path(value)
        return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


@lru_cache
def get_settings() -> Settings:
    return Settings()
