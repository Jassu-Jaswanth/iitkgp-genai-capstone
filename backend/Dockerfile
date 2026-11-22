FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
# RAG_EXTRA=rag bakes in the search libraries (RETRIEVAL_MODE=inprocess); empty = slim image (placeholder / http)
ARG RAG_EXTRA=""
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project ${RAG_EXTRA:+--extra $RAG_EXTRA}
ENV PATH="/app/backend/.venv/bin:$PATH" PYTHONUNBUFFERED=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 FASTEMBED_CACHE_PATH=/models
# Pre-download the embedder (query fallback), reranker and tokenizer so the first start doesn't stall on downloads
RUN if [ -n "$RAG_EXTRA" ]; then python -c "from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; from huggingface_hub import hf_hub_download; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2'); \
hf_hub_download('BAAI/bge-small-en-v1.5', 'tokenizer.json')"; fi
COPY rag/*.py /app/rag/
COPY backend/app ./app
EXPOSE 8100
# .env, CA certs and data/ (incl. the search index in data/index) are mounted at runtime; no secrets baked in.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
