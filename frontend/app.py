"""HR Policy Navigator dashboard (Streamlit).

    cd frontend && uv run streamlit run app.py         # expects the backend on API_URL (default :8100)

Pages: Chat (agent or single-prompt baseline, with trace), Evaluation (gold-set runs), Retrieval experiments.
Talks to the backend over HTTP only; reads eval/results/*.json(l) and *.csv from the repo for the reports.
"""

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8100")
RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results"

st.set_page_config(page_title="HR Policy Navigator", layout="wide")
page = st.sidebar.radio("Page", ["Chat", "Evaluation", "Retrieval experiments"])


@st.cache_data(ttl=30)
def health() -> dict:
    try:
        return httpx.get(f"{API}/health", timeout=10).json()
    except Exception as exc:
        return {"status": "down", "error": str(exc)}


h = health()
st.sidebar.markdown(f"**Backend:** `{API}`  \nStatus: **{h.get('status')}**  \nRetriever: `{h.get('retriever', '-')}`")

# ------------------------------------------------------------------ chat
if page == "Chat":
    st.title("HR Policy Navigator")
    mode = st.radio("Mode", ["agent", "chat (single-prompt baseline)"], horizontal=True)
    if "messages" not in st.session_state:
        st.session_state.messages, st.session_state.session_id = [], None
    if st.sidebar.button("New conversation"):
        st.session_state.messages, st.session_state.session_id = [], None

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("details"):
                with st.expander("Details"):
                    st.json(m["details"], expanded=False)

    if q := st.chat_input("Ask about federal leave, pay, holidays…"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"), st.spinner("Thinking… (one request at a time on the shared model)"):
            try:
                if mode == "agent":
                    r = httpx.post(f"{API}/agent", json={"question": q, "session_id": st.session_state.session_id},
                                   timeout=600).json()
                    st.session_state.session_id = r.get("session_id")
                else:
                    hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]][-6:]
                    r = httpx.post(f"{API}/chat", json={"question": q, "history": hist}, timeout=600).json()
            except Exception as exc:
                r = {"answer": f"Request failed: {exc}"}
            answer = r.get("answer", str(r))
            st.markdown(answer)
            cols = st.columns(4)
            cols[0].metric("Route" if mode == "agent" else "Grounded", str(r.get("route", r.get("grounded", "-"))))
            cols[1].metric("Tool calls" if mode == "agent" else "Chunks", r.get("tool_calls", len(r.get("chunks", []))))
            cols[2].metric("LLM calls", r.get("usage", {}).get("llm_calls", "-"))
            lat = r.get("latency_ms")
            cols[3].metric("Latency (s)", round((lat if isinstance(lat, (int, float)) else (lat or {}).get("total", 0)) / 1000, 1))
            if r.get("citations"):
                st.caption("Citations: " + " · ".join(r["citations"]))
            if r.get("critique"):
                st.warning("Critic: " + " | ".join(r["critique"]))
            if r.get("trace"):
                with st.expander("Agent trace"):
                    for t in r["trace"]:
                        if t["type"] == "tool_call":
                            st.markdown(f"**→ {t['tool']}** `{json.dumps(t['args'])}`")
                        elif t["type"] == "tool_result":
                            st.code(t.get("content", "")[:800], language="json")
                        else:
                            st.markdown(f"*{t['type']}* {t.get('route', '')} {t.get('issues', '')}")
            if r.get("chunks"):
                with st.expander("Retrieved chunks"):
                    for c in r["chunks"]:
                        st.markdown(f"**{c['citation']}**  \n{c['text'][:600]}")
        st.session_state.messages.append({"role": "assistant", "content": answer,
                                          "details": {k: r.get(k) for k in ("route", "citations", "usage", "latency_ms", "status")}})

# ------------------------------------------------------------------ evaluation
elif page == "Evaluation":
    st.title("Gold-set evaluation")
    runs = sorted(RESULTS.glob("e2e_*.summary.json"), reverse=True)
    if not runs:
        st.info("No runs yet. Start the backend and run: python eval/run_eval.py --mode agent")
    else:
        summ = []
        for p in runs:
            s = json.loads(p.read_text())
            summ.append({"run": p.name.replace(".summary.json", ""), **{k: v for k, v in s.items() if not isinstance(v, dict)},
                         "failures": ", ".join(f"{k}:{v}" for k, v in s.get("failures", {}).items())})
        st.subheader("Runs")
        st.dataframe(pd.DataFrame(summ), use_container_width=True, hide_index=True)
        pick = st.selectbox("Run details", [s["run"] for s in summ])
        rows = [json.loads(l) for l in (RESULTS / f"{pick}.jsonl").open()]
        df = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        c1.subheader("Accuracy by category")
        c1.bar_chart(df.groupby("category")["correct"].mean())
        c2.subheader("Failure taxonomy")
        if df["failure"].notna().any():
            c2.bar_chart(df["failure"].value_counts())
        st.subheader("Questions")
        show = df[["qid", "category", "correct", "failure", "behavior", "tools_called", "latency_ms", "question", "answer", "gold_answer"]]
        if st.checkbox("Only failures"):
            show = show[~show["correct"]]
        st.dataframe(show, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ retrieval experiments
else:
    st.title("Retrieval experiments")
    files = sorted(RESULTS.glob("retrieval_*.csv"), reverse=True)
    if not files:
        st.info("No retrieval experiments yet. Run: rag/.venv/bin/python rag/experiments.py")
    else:
        f = st.selectbox("Experiment file", [p.name for p in files])
        df = pd.read_csv(RESULTS / f)
        metric = st.selectbox("Metric", ["sec@1", "sec@5", "cl@5", "MRR", "sec@10", "ms/query"], index=1)
        st.dataframe(df.sort_values(metric, ascending=metric == "ms/query"), use_container_width=True, hide_index=True)
        pivot = df.assign(variant=df["retriever"] + " / " + df["query"]).pivot_table(index="chunking", columns="variant", values=metric)
        st.subheader(f"{metric} by chunking × retriever")
        st.dataframe(pivot.style.background_gradient(axis=None), use_container_width=True)
