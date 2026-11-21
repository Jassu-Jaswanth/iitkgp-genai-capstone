"""HR Policy Navigator: one dashboard for everything (Streamlit).

    cd frontend && uv run streamlit run app.py     # backend on API_URL (:8100), Phoenix on PHOENIX_URL (:6006)

Pages
- Chat: talk to the agent; each answer can unfold into its full LLM trace (model calls, tool calls, retrieved chunks).
- Traces: every agent turn with latency / tokens / tools, and a drill-down into any one of them.
- Evaluation: gold-set runs.  - Retrieval experiments: chunking x retriever grid.
Phoenix only stores the traces; this app reads them through its API, so there is a single UI to open.
"""

import json
import os
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import streamlit as st

import trace_view as tv

API = os.getenv("API_URL", "http://localhost:8100")
RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results"

st.set_page_config(page_title="HR Policy Navigator", page_icon="🧭", layout="wide")
page = st.sidebar.radio("Page", ["Chat", "Traces", "Evaluation", "Retrieval experiments"])


@st.cache_data(ttl=30)
def health() -> dict:
    try:
        return httpx.get(f"{API}/health", timeout=10).json()
    except Exception as exc:
        return {"status": "down", "error": str(exc)}


@st.cache_data(ttl=30)
def tracing_up() -> bool:
    return tv.available()


h = health()
st.sidebar.markdown(f"**Backend** `{h.get('status')}`  \n**Retriever** `{h.get('retriever', '-')}`  \n"
                    f"**Traces** `{'on' if tracing_up() else 'off'}`")


def show_trace(turn_id: str, key: str) -> None:
    if not tracing_up():
        st.caption("Tracing is off (Phoenix not reachable), showing the backend's step list only.")
        return
    with st.spinner("Loading trace…"):
        tv.render(tv.spans_for_turn(turn_id), key=key)


def show_backend_steps(trace: list[dict]) -> None:
    for t in trace:
        if t["type"] == "tool_call":
            st.markdown(f"**→ {t['tool']}** `{json.dumps(t['args'])}`")
        elif t["type"] == "tool_result":
            st.code(t.get("content", "")[:800], language="json")
        else:
            st.markdown(f"*{t['type']}* {t.get('route', '')} {t.get('issues', '')}")


def answer_block(r: dict, mode: str, idx: int) -> None:
    st.markdown(r.get("answer", str(r)))
    if r.get("citations"):
        st.caption("Sources: " + " · ".join(r["citations"]))
    if r.get("critique"):
        st.warning("Critic: " + " | ".join(r["critique"]))
    lat = r.get("latency_ms")
    lat_s = round((lat if isinstance(lat, (int, float)) else (lat or {}).get("total", 0)) / 1000, 1)
    usage = r.get("usage", {})
    bits = [f"{lat_s}s", f"{usage.get('llm_calls', '-')} LLM calls",
            f"{usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)} tokens"]
    if mode == "agent":
        bits = [f"route `{r.get('route', '-')}`", f"{r.get('tool_calls', 0)} tool calls"] + bits
    st.caption(" · ".join(bits))
    if mode == "agent" and r.get("turn_id"):
        if st.toggle("Show trace", key=f"tr-{idx}"):
            show_trace(r["turn_id"], key=f"chat-{idx}")
            if not tracing_up() and r.get("trace"):
                show_backend_steps(r["trace"])
    elif r.get("chunks") and st.toggle("Show retrieved chunks", key=f"ch-{idx}"):
        for c in r["chunks"]:
            st.markdown(f"**{c['citation']}**")
            st.caption(c["text"][:600])


# ------------------------------------------------------------------ chat
if page == "Chat":
    st.title("HR Policy Navigator")
    mode = st.radio("Mode", ["agent", "chat (single-prompt baseline)"], horizontal=True,
                    help="agent = LangGraph router/planner/tools/critic; chat = one retrieve-then-answer prompt")
    if "messages" not in st.session_state:
        st.session_state.messages, st.session_state.session_id = [], None
    if st.sidebar.button("New conversation"):
        st.session_state.messages, st.session_state.session_id = [], None

    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                answer_block(m["response"], m["mode"], i)

    if q := st.chat_input("Ask about federal leave, pay, holidays…"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking… (one request at a time on the shared model)"):
                try:
                    if mode == "agent":
                        r = httpx.post(f"{API}/agent", json={"question": q, "session_id": st.session_state.session_id},
                                       timeout=600).json()
                        st.session_state.session_id = r.get("session_id")
                    else:
                        hist = [{"role": m["role"], "content": m.get("content") or m["response"].get("answer", "")}
                                for m in st.session_state.messages[:-1]][-6:]
                        r = httpx.post(f"{API}/chat", json={"question": q, "history": hist}, timeout=600).json()
                except Exception as exc:
                    r = {"answer": f"Request failed: {exc}"}
            idx = len(st.session_state.messages)
            answer_block(r, "agent" if mode == "agent" else "chat", idx)
        st.session_state.messages.append({"role": "assistant", "response": r,
                                          "mode": "agent" if mode == "agent" else "chat"})

# ------------------------------------------------------------------ traces
elif page == "Traces":
    st.title("Agent traces")
    if not tracing_up():
        st.error(f"Phoenix is not reachable at {tv.PHOENIX_URL}. Start it (docker compose up phoenix) and set "
                 "PHOENIX_COLLECTOR_ENDPOINT on the backend.")
        st.stop()
    c0, c1 = st.columns([3, 1])
    n = c1.selectbox("Show last", [25, 50, 100, 200], index=1)
    if c1.button("Refresh"):
        tv.recent_traces.clear()
    df = tv.recent_traces(n)
    if df.empty:
        st.info("No agent turns traced yet. Ask something on the Chat page.")
        st.stop()
    k = st.columns(5)
    k[0].metric("Turns", len(df))
    k[1].metric("Median latency (s)", round(df["latency_s"].median(), 1))
    k[2].metric("p90 latency (s)", round(df["latency_s"].quantile(0.9), 1))
    k[3].metric("Avg tokens / turn", int(df["tokens"].mean()))
    k[4].metric("Errors", int((df["status"] == "ERROR").sum()))
    g1, g2 = st.columns(2)
    g1.caption("Latency per turn, oldest → newest (s)")
    g1.bar_chart(df["latency_s"][::-1].reset_index(drop=True), height=180)
    g2.caption("Turns by route")
    g2.bar_chart(df["route"].value_counts(), height=180, horizontal=True)
    st.caption("Pick a row to open its trace.")
    sel = st.dataframe(df.drop(columns=["trace_id"]), width="stretch", hide_index=True,
                       on_select="rerun", selection_mode="single-row",
                       column_config={"time": st.column_config.DatetimeColumn(format="HH:mm:ss")})
    rows = sel.selection.rows if sel else []
    if rows:
        row = df.iloc[rows[0]]
        st.subheader(row["question"][:140])
        tv.render(tv.trace_spans(row["trace_id"]), key=f"tr-{row['trace_id']}")

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
        st.dataframe(pd.DataFrame(summ), width="stretch", hide_index=True)
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
        st.dataframe(show, width="stretch", hide_index=True)

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
        st.dataframe(df.sort_values(metric, ascending=metric == "ms/query"), width="stretch", hide_index=True)
        grid = df.assign(variant=df["retriever"] + " / " + df["query"])
        st.subheader(f"{metric} by chunking × retriever")
        heat = alt.Chart(grid).mark_rect().encode(
            x=alt.X("variant:N", title=None, axis=alt.Axis(labelAngle=-30, labelLimit=220)),
            y=alt.Y("chunking:N", title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color(f"mean({metric}):Q", title=metric,
                            scale=alt.Scale(scheme="blues", reverse=metric == "ms/query")),
            tooltip=["chunking", "variant", alt.Tooltip(f"mean({metric}):Q", format=".3f")])
        text = heat.mark_text(fontSize=11).encode(text=alt.Text(f"mean({metric}):Q", format=".2f"), color=alt.value("black"))
        st.altair_chart((heat + text).properties(height=28 * grid["chunking"].nunique() + 60), width="stretch")
