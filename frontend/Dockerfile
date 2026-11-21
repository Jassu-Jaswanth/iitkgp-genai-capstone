FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
WORKDIR /app/frontend
COPY frontend/pyproject.toml frontend/uv.lock ./
RUN uv sync --frozen --no-install-project
COPY frontend/app.py ./
ENV PATH="/app/frontend/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
