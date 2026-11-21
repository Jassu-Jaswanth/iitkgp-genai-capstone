# Frontend: HR Policy Navigator dashboard (Streamlit)

```bash
cd frontend
uv sync
API_URL=http://localhost:8100 uv run streamlit run app.py
```

Pages:
- **Chat**: the LangGraph agent (or the single-prompt `/chat` baseline), with route, citations, critic notes,
  tool trace, retrieved chunks, LLM calls and latency.
- **Evaluation**: gold-set runs from `eval/results/e2e_*.jsonl`: accuracy by category, failure taxonomy, per question.
- **Retrieval experiments**: chunking × retriever grids from `eval/results/retrieval_*.csv`.
