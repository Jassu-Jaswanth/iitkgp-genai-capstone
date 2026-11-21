"""Read agent traces from Phoenix (headless span store) and draw them inside the dashboard.

Phoenix collects the OpenInference spans the backend and RAG service emit; this module only reads them back through
the official `arize-phoenix-client` REST API, so nobody has to open Phoenix's own UI.
A trace = one agent turn: LangGraph root -> nodes (router, planner, tools, critic) -> LLM / TOOL / RETRIEVER spans.
"""

import json
import os
import time
from datetime import datetime
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006")
PROJECT = os.getenv("PHOENIX_PROJECT_NAME", "hr-policy-navigator")
SHOWN_KINDS = {"LLM", "TOOL", "RETRIEVER"}


def _client():
    from phoenix.client import Client
    return Client(base_url=PHOENIX_URL)


def available() -> bool:
    try:
        _client().spans.get_spans(project_identifier=PROJECT, limit=1, timeout=3)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ fetching
@st.cache_data(ttl=600, show_spinner=False)
def _trace_spans_cached(trace_id: str) -> list[dict]:
    return _client().spans.get_spans(project_identifier=PROJECT, trace_ids=[trace_id], limit=2000, timeout=20)


def spans_for_turn(turn_id: str, wait_s: float = 15) -> list[dict]:
    """Spans of one agent turn. The exporter batches (~5 s), so poll briefly after a fresh answer."""
    deadline = time.time() + wait_s
    while True:
        roots = _client().spans.get_spans(project_identifier=PROJECT, parent_id="null",
                                          attributes={"metadata.turn_id": turn_id}, limit=1, timeout=10)
        if roots:
            spans = _trace_spans_cached(roots[0]["context"]["trace_id"])
            # the root is exported last; once it is in, the whole turn is in
            return spans
        if time.time() > deadline:
            return []
        time.sleep(1.5)


def trace_spans(trace_id: str) -> list[dict]:
    return _trace_spans_cached(trace_id)


@st.cache_data(ttl=15, show_spinner=False)
def recent_traces(limit: int = 100) -> pd.DataFrame:
    """One row per agent turn with the numbers people ask about: question, route, latency, LLM calls, tokens, tools."""
    c = _client()
    roots = c.spans.get_spans(project_identifier=PROJECT, parent_id="null", name="LangGraph", sort="start_time",
                              order="desc", limit=limit, timeout=20)
    if not roots:
        return pd.DataFrame()
    ids = [r["context"]["trace_id"] for r in roots]
    start = min(_ts(r["start_time"]) for r in roots)
    kids = c.spans.get_spans(project_identifier=PROJECT, span_kind=["LLM", "TOOL"], start_time=start,
                             limit=limit * 20, timeout=30)
    agg: dict[str, dict[str, Any]] = {t: {"llm_calls": 0, "tokens": 0, "tools": []} for t in ids}
    for s in kids:
        a = agg.get(s["context"]["trace_id"])
        if a is None:
            continue
        if s["span_kind"] == "LLM":
            a["llm_calls"] += 1
            a["tokens"] += int(s["attributes"].get("llm.token_count.total") or 0)
        else:
            a["tools"].append(s["name"])
    rows = []
    for r in roots:
        at, tid = r["attributes"], r["context"]["trace_id"]
        out = _json(at.get("output.value")) or {}
        rows.append({"time": _ts(r["start_time"]), "question": _question(at),
                     "route": out.get("route", "-") if isinstance(out, dict) else "-",
                     "latency_s": round((_ts(r["end_time"]) - _ts(r["start_time"])).total_seconds(), 1),
                     "llm_calls": agg[tid]["llm_calls"], "tokens": agg[tid]["tokens"],
                     "tools": ", ".join(agg[tid]["tools"]), "status": r["status_code"],
                     "session": (at.get("session.id") or "")[:8], "trace_id": tid})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ helpers
def _ts(v: Any) -> datetime:
    return v if isinstance(v, datetime) else datetime.fromisoformat(str(v))


def _json(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except ValueError:
        return v


def _question(attrs: dict) -> str:
    """Latest user message of the turn (root input holds just this turn's new message)."""
    inp = _json(attrs.get("input.value"))
    if isinstance(inp, dict):
        for m in reversed(inp.get("messages", [])):
            data = m.get("data", m) if isinstance(m, dict) else {}
            if (m.get("type") if isinstance(m, dict) else None) == "human" or data.get("type") == "human":
                return str(data.get("content", ""))
    return str(attrs.get("llm.input_messages.0.message.content", ""))


def _indexed(attrs: dict, prefix: str) -> list[dict[str, Any]]:
    """Unflatten OpenInference keys like `llm.input_messages.3.message.content` into a list of dicts."""
    items: dict[int, dict[str, Any]] = {}
    for k, v in attrs.items():
        if k.startswith(prefix + "."):
            idx, _, rest = k[len(prefix) + 1:].partition(".")
            if idx.isdigit():
                items.setdefault(int(idx), {})[rest] = v
    return [items[i] for i in sorted(items)]


def _documents(span: dict) -> list[dict]:
    at = span["attributes"]
    docs = _indexed(at, "retrieval.documents")
    if docs:
        return [{"id": d.get("document.id"), "text": d.get("document.content", ""),
                 "meta": _json(d.get("document.metadata")) or {}} for d in docs]
    out = _json(at.get("output.value"))
    if isinstance(out, dict):
        return [{"id": d.get("id"), "text": d.get("page_content", ""), "meta": d.get("metadata", {})}
                for d in out.get("documents", [])]
    return []


def _tool_calls(msg: dict) -> list[str]:
    calls = _indexed(msg, "message.tool_calls")
    return [f"{c.get('tool_call.function.name')}({c.get('tool_call.function.arguments', '')})" for c in calls]


# ------------------------------------------------------------------ building the step list
def steps(spans: list[dict]) -> list[dict]:
    """Ordered, readable steps: graph nodes plus every LLM / TOOL / RETRIEVER span beneath them, with depth.
    LangChain plumbing (RunnableSequence, parsers, routing lambdas) is folded away."""
    if not spans:
        return []
    root = next((s for s in spans if s["parent_id"] is None), None) or min(spans, key=lambda s: _ts(s["start_time"]))
    t0 = _ts(root["start_time"])
    children: dict[str, list[dict]] = {}
    for s in spans:
        children.setdefault(s["parent_id"], []).append(s)
    out: list[dict] = []

    def walk(span: dict, depth: int, node: str) -> None:
        for ch in sorted(children.get(span["context"]["span_id"], []), key=lambda s: _ts(s["start_time"])):
            is_node = span is root and ch["span_kind"] == "CHAIN"
            if is_node and (ch["name"].startswith(("after_", "route_")) or not _has_shown(ch)):
                continue
            keep = is_node or ch["span_kind"] in SHOWN_KINDS
            if keep:
                out.append({"span": ch, "depth": depth, "node": ch["name"] if is_node else node,
                            "kind": "NODE" if is_node else ch["span_kind"],
                            "start_ms": (_ts(ch["start_time"]) - t0).total_seconds() * 1000,
                            "end_ms": (_ts(ch["end_time"]) - t0).total_seconds() * 1000})
            walk(ch, depth + 1 if keep else depth, ch["name"] if is_node else node)

    def _has_shown(s: dict) -> bool:
        return any(c["span_kind"] in SHOWN_KINDS or _has_shown(c) for c in children.get(s["context"]["span_id"], []))

    walk(root, 0, "")
    return out


# ------------------------------------------------------------------ drawing
def _under_retriever(span: dict, spans: list[dict]) -> bool:
    """HTTP mode records PolicyRetriever -> (service) -> ...; count each search once."""
    parent = next((p for p in spans if p["context"]["span_id"] == span["parent_id"]), None)
    return bool(parent and parent["span_kind"] == "RETRIEVER")


def summary_metrics(spans: list[dict]) -> dict[str, Any]:
    llm = [s for s in spans if s["span_kind"] == "LLM"]
    root = next((s for s in spans if s["parent_id"] is None), None)
    return {"llm_calls": len(llm),
            "prompt_tokens": sum(int(s["attributes"].get("llm.token_count.prompt") or 0) for s in llm),
            "completion_tokens": sum(int(s["attributes"].get("llm.token_count.completion") or 0) for s in llm),
            "tool_calls": sum(s["span_kind"] == "TOOL" for s in spans),
            "retrievals": sum(s["span_kind"] == "RETRIEVER" and s["name"] in ("PolicyRetriever", "PolicySearch")
                              and not _under_retriever(s, spans) for s in spans),
            "latency_s": round((_ts(root["end_time"]) - _ts(root["start_time"])).total_seconds(), 1) if root else 0}


def waterfall(st_steps: list[dict]) -> alt.Chart:
    df = pd.DataFrame([{"step": f"{i:02d} {'  ' * s['depth']}{_label(s)}", "kind": s["kind"],
                        "start": s["start_ms"], "end": s["end_ms"], "ms": round(s["end_ms"] - s["start_ms"])}
                       for i, s in enumerate(st_steps)])
    return (alt.Chart(df).mark_bar(cornerRadius=2)
            .encode(x=alt.X("start:Q", title="ms since the question arrived", axis=alt.Axis(tickCount=8)), x2="end:Q",
                    y=alt.Y("step:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
                    color=alt.Color("kind:N", legend=alt.Legend(orient="top", title=None)),
                    tooltip=["step", "kind", "ms"])
            .properties(height=max(120, 22 * len(df))))


def _label(s: dict) -> str:
    sp = s["span"]
    if s["kind"] == "NODE":
        return f"[{sp['name']}]"
    if s["kind"] == "LLM":
        return f"LLM · {s['node']}"
    if s["kind"] == "TOOL":
        return f"tool · {sp['name']}"
    return f"retriever · {sp['name']}"


def render(spans: list[dict], key: str) -> None:
    """Metrics, a timing waterfall and a step-by-step drill-down of one agent turn."""
    if not spans:
        st.info("No trace found yet for this turn (Phoenix may still be receiving it, or tracing is off).")
        return
    m = summary_metrics(spans)
    c = st.columns(5)
    c[0].metric("Latency (s)", m["latency_s"])
    c[1].metric("LLM calls", m["llm_calls"])
    c[2].metric("Tokens in / out", f"{m['prompt_tokens']} / {m['completion_tokens']}")
    c[3].metric("Tool calls", m["tool_calls"])
    c[4].metric("Searches", m["retrievals"])
    st_steps = steps(spans)
    st.altair_chart(waterfall(st_steps), width="stretch", key=f"wf-{key}")
    for i, s in enumerate(st_steps):
        sp, at = s["span"], s["span"]["attributes"]
        ms = round(s["end_ms"] - s["start_ms"])
        pad = "\u2003" * max(s["depth"] - 1, 0) + ("↳ " if s["depth"] else "")
        if s["kind"] == "NODE":
            st.markdown(f"**{pad}{sp['name']}**  ·  {ms} ms")
            continue
        err = " ⚠️" if sp["status_code"] == "ERROR" else ""
        if s["kind"] == "LLM":
            out = _indexed(at, "llm.output_messages")
            calls = _tool_calls(out[0]) if out else []
            content = (out[0].get("message.content") if out else "") or ""
            head = (f"{pad}🧠 LLM ({s['node']}) · {ms} ms · {at.get('llm.token_count.prompt', '?')}→"
                    f"{at.get('llm.token_count.completion', '?')} tok{err}")
            with st.expander(head):
                if calls:
                    st.markdown("**Asked for:** " + "  \n".join(f"`{c}`" for c in calls))
                if content:
                    st.markdown("**Output:**")
                    st.write(content)
                st.markdown("**Prompt messages:**")
                for msg in _indexed(at, "llm.input_messages"):
                    role = msg.get("message.role", "?")
                    body = msg.get("message.content") or " ".join(_tool_calls(msg)) or ""
                    st.markdown(f"`{role}` {body[:1500]}")
        elif s["kind"] == "TOOL":
            with st.expander(f"{pad}🛠 tool {sp['name']} · {ms} ms{err}"):
                st.markdown("**Arguments**")
                st.json(_json(at.get("input.value")) or {}, expanded=True)
                st.markdown("**Result**")
                res = _json(at.get("output.value"))
                if isinstance(res, (dict, list)):
                    st.json(res, expanded=False)
                else:
                    st.code(str(res)[:3000])
        else:
            docs = _documents(sp)
            with st.expander(f"{pad}🔎 {sp['name']} · {ms} ms · {len(docs)} chunks · “{str(at.get('input.value', ''))[:80]}”{err}"):
                for rank, d in enumerate(docs, 1):
                    meta = d["meta"] if isinstance(d["meta"], dict) else {}
                    st.markdown(f"**{rank}. {meta.get('citation', d.get('id'))}**  "
                                f"<span style='opacity:.6'>{meta.get('source_type', '')}</span>",
                                unsafe_allow_html=True)
                    st.caption(str(d["text"])[:700])
