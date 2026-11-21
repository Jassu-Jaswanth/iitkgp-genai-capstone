FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
# RAG_EXTRA=rag bakes the retrieval libraries in (RAG_MODE=inprocess); empty = slim image (placeholder / http)
ARG RAG_EXTRA=""
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project ${RAG_EXTRA:+--extra $RAG_EXTRA}
ENV PATH="/app/backend/.venv/bin:$PATH" PYTHONUNBUFFERED=1 HF_HOME=/models
# Pre-download the tokenizer and reranker so the first start doesn't stall on Hugging Face
RUN if [ -n "$RAG_EXTRA" ]; then python -c "from transformers import AutoTokenizer; \
from sentence_transformers import CrossEncoder; AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"; fi
COPY rag/pipeline.py /app/rag/pipeline.py
COPY backend/app ./app
EXPOSE 8100
# .env, CA certs and data/ (incl. the cached FAISS index in data/index) are mounted at runtime; no secrets baked in.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
