"""The LLM trace of one agent turn, taken straight from LangGraph's event stream (astream_events v2).

Every model call, tool call and retrieval already emits start/end events with its inputs, outputs, run id, parent
run ids and the graph node it ran in; this module only keeps those three kinds of events and flattens them into
ordered steps that can be stored (chat.trace_steps) and drawn. No tracing server needed.

Step = {seq, parent_seq, node, kind (llm|tool|retriever), name, input, output, prompt_tokens, completion_tokens,
        started_at, ended_at, error}
"""

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

KINDS = {"chat_model": "llm", "tool": "tool", "retriever": "retriever"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(text: Any) -> Any:
    if isinstance(text, str) and text[:1] in "{[":
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


def jsonable(obj: Any) -> Any:
    """Messages, documents and nested containers -> plain JSON (tool results/arguments parsed when they are JSON)."""
    if isinstance(obj, BaseMessage):
        out: dict[str, Any] = {"role": obj.type, "content": _parse(obj.content)}
        if getattr(obj, "tool_calls", None):
            out["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in obj.tool_calls]
        if getattr(obj, "name", None):
            out["name"] = obj.name
        if getattr(obj, "status", None) == "error":
            out["status"] = "error"
        return out
    if isinstance(obj, Document):
        return {"text": obj.page_content, **{k: v for k, v in obj.metadata.items() if k != "text"}}
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items() if k not in ("callbacks", "run_manager", "state")}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "model_dump"):
        return jsonable(obj.model_dump())
    return str(obj)


class TraceCollector:
    """Feed it every event from `graph.astream_events(..., version="v2")`; read `.steps` afterwards."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self._open: dict[str, dict[str, Any]] = {}
        self._seq_of_run: dict[str, int] = {}

    def add(self, ev: dict[str, Any]) -> None:
        etype = ev["event"]  # on_<kind>_<start|end|error>
        kind = next((KINDS[k] for k in KINDS if etype.startswith(f"on_{k}_")), None)
        if kind is None:
            return
        run_id, data = ev["run_id"], ev.get("data", {})
        if etype.endswith("_start"):
            parent = next((self._seq_of_run[p] for p in reversed(ev.get("parent_ids", [])) if p in self._seq_of_run),
                          None)
            step = {"seq": len(self.steps), "parent_seq": parent, "node": ev.get("metadata", {}).get("langgraph_node"),
                    "kind": kind, "name": ev.get("name"), "input": self._input(kind, data.get("input")),
                    "output": None, "prompt_tokens": None, "completion_tokens": None,
                    "started_at": _now(), "ended_at": None, "error": None}
            self.steps.append(step)
            self._open[run_id] = step
            self._seq_of_run[run_id] = step["seq"]
            return
        step = self._open.pop(run_id, None)
        if step is None:
            return
        step["ended_at"] = _now()
        if etype.endswith("_error"):
            step["error"] = str(data.get("error"))
            return
        out = data.get("output")
        if kind == "llm" and isinstance(out, BaseMessage):
            um = getattr(out, "usage_metadata", None) or {}
            step["prompt_tokens"], step["completion_tokens"] = um.get("input_tokens"), um.get("output_tokens")
        if kind == "tool" and isinstance(out, BaseMessage) and getattr(out, "status", None) == "error":
            step["error"] = str(out.content)
        if kind == "tool":
            out = _parse(out.content if isinstance(out, BaseMessage) else out)
        step["output"] = jsonable(out)

    @staticmethod
    def _input(kind: str, inp: Any) -> Any:
        if kind == "llm" and isinstance(inp, dict) and "messages" in inp:
            msgs = inp["messages"]
            return jsonable(msgs[0] if msgs and isinstance(msgs[0], list) else msgs)  # [[...]] for one prompt
        if kind == "retriever" and isinstance(inp, dict) and "query" in inp:
            return inp["query"]
        return jsonable(inp)
