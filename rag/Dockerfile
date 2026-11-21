FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
WORKDIR /app/rag
COPY rag/pyproject.toml rag/uv.lock ./
RUN uv sync --frozen --no-dev
ENV PATH="/app/rag/.venv/bin:$PATH" PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    FASTEMBED_CACHE_PATH=/models
# Pre-download the embedder, reranker and tokenizer at build time (first-run downloads can stall; experiment log E12)
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
from huggingface_hub import hf_hub_download; TextEmbedding('BAAI/bge-small-en-v1.5'); \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2'); hf_hub_download('BAAI/bge-small-en-v1.5', 'tokenizer.json')"
COPY rag/*.py ./
EXPOSE 8200
# The index (data/index) and processed corpus are mounted at runtime (see compose.yaml).
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8200"]
