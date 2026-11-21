FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY backend/app ./app
ENV PATH="/app/backend/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8100
# .env, CA certs and data/ are mounted at runtime (see compose.yaml); nothing secret is baked into the image.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
