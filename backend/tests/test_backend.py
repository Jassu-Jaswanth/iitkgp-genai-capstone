"""Unit tests for the agent skeleton: real LangGraph wiring, scripted fake chat models (no network, no key)."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphRecursionError

from app.agent import build_agent, summarize_turn
from app.config import REPO_ROOT, Settings
from app.main import _known_citations_from_trace, app, extract_citations
from app.prompts import NO_CONTEXT_NOTE, build_messages, parse_strategy_output
from app.retrieval import Chunk
from app.tools import build_tools

DATA = REPO_ROOT / "data"


class ScriptedChat(BaseChatModel):
    """Returns scripted AIMessages in order."""
    script: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        msg = self.script.pop(0) if self.script else AIMessage(content="(script exhausted)")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kw):
        return self


def call(name: str, args: dict[str, Any], i: str = "1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{i}", "type": "tool_call"}])


def route(name: str = "hr_policy") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "Route", "args": {"reason": "t", "route": name}, "id": "rt", "type": "tool_call"}])


def critique(*issues: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "Critique", "args": {"issues": list(issues)}, "id": "c", "type": "tool_call"}])


class NoRAG:  # what main uses before the RAG service exists
    name = "placeholder"

    async def retrieve(self, query, k=5, filters=None):
        return []


class OneChunk:
    name = "fake"

    async def retrieve(self, query, k=5, filters=None):
        return [Chunk(chunk_id="c1", text="Annual leave accrues three-fourths day per pay period for 3 to 15 years.",
                      citation="5 U.S.C. 6303(a)", clause_ids=["usc5-6303-a-2"], score=0.9)]


NC = "The provided policy documents do not cover this."


def agent_for(planner, critic, retriever=None, routes=None, **kw):
    tools = build_tools(retriever or NoRAG(), DATA)
    return build_agent(ScriptedChat(script=planner), ScriptedChat(script=critic), tools,
                       router_llm=ScriptedChat(script=routes if routes is not None else [route() for _ in range(5)]), **kw)


async def run(graph, q, tid="t", limit=24):
    cfg = {"configurable": {"thread_id": tid}, "recursion_limit": limit}
    await graph.ainvoke({"messages": [HumanMessage(q)], "route": "hr_policy", "critic_rounds": 0, "critique": [],
                         "critic_usage": []}, cfg)
    st = (await graph.aget_state(cfg)).values
    return summarize_turn(st["messages"], st["critic_usage"]), st


# ---------- agent loop ----------

def test_only_the_example_tool_is_registered():
    assert [t.name for t in build_tools(NoRAG(), DATA)] == ["search_policy"]


async def test_works_without_rag():
    g = agent_for([call("search_policy", {"query": "annual leave carryover"}), AIMessage(content=NC)], [])
    s, _ = await run(g, "How much annual leave can I carry over?")
    assert s["answer"] == NC and s["tool_calls"] == 1
    assert '"results": []' in next(t for t in s["trace"] if t["type"] == "tool_result")["content"]


async def test_grounded_answer_with_rag_passes_critic():
    g = agent_for([call("search_policy", {"query": "annual leave accrual"}),
                   AIMessage(content="6 hours per pay period [5 U.S.C. 6303(a)].")], [critique()], retriever=OneChunk())
    s, st = await run(g, "Leave with 4 years?")
    assert s["answer"].startswith("6 hours") and st["critique"] == []


async def test_critic_feedback_triggers_one_revision():
    g = agent_for([call("search_policy", {"query": "annual leave accrual"}),
                   AIMessage(content="8 hours per pay period [5 U.S.C. 6303(a)]."),
                   AIMessage(content="6 hours per pay period [5 U.S.C. 6303(a)].")],
                  [critique("8 hours contradicts three-fourths day"), critique()], retriever=OneChunk())
    s, st = await run(g, "Leave with 4 years?")
    assert s["answer"].startswith("6 hours") and st["critic_rounds"] == 2


async def test_not_covered_without_search_is_sent_back():
    g = agent_for([AIMessage(content=NC), call("search_policy", {"query": "fmla eligibility"}, "2"),
                   AIMessage(content="12 months of service [5 U.S.C. 6303(a)].")], [critique()], retriever=OneChunk())
    s, _ = await run(g, "FMLA eligibility?")
    assert any(t["type"] == "critic_feedback" for t in s["trace"]) and s["answer"].startswith("12 months")


async def test_bad_tool_args_become_an_observation():
    g = agent_for([call("search_policy", {"query": "x"}),  # too short -> validation error, not a crash
                   call("search_policy", {"query": "annual leave"}, "2"), AIMessage(content=NC)], [])
    s, _ = await run(g, "q?")
    assert s["tool_errors"] == 1 and s["answer"] == NC


async def test_router_privacy_and_out_of_scope_skip_tools():
    s, _ = await run(agent_for([], [], routes=[route("individual_records")]), "What is Sarah's salary?")
    assert s["route"] == "individual_records" and s["tool_calls"] == 0 and "private" in s["answer"]
    s, _ = await run(agent_for([], [], routes=[route("out_of_scope")]), "Google 401k match?")
    assert s["route"] == "out_of_scope" and "do not cover" in s["answer"]


async def test_router_failure_falls_back_to_planner():
    g = agent_for([call("search_policy", {"query": "anything here"}), AIMessage(content=NC)], [],
                  routes=[AIMessage(content="not json")])
    s, _ = await run(g, "anything")
    assert s["route"] == "hr_policy"


async def test_step_limit_stops_runaway_loop():
    g = agent_for([call("search_policy", {"query": f"loop query {i}"}, str(i)) for i in range(50)], [])
    with pytest.raises(GraphRecursionError):
        await run(g, "loop", limit=6)


async def test_session_memory_keeps_history():
    g = agent_for([call("search_policy", {"query": "first rule"}, "1"), AIMessage(content=NC),
                   call("search_policy", {"query": "second rule"}, "2"), AIMessage(content=NC)], [])
    await run(g, "first question", tid="s1")
    _, st = await run(g, "second question", tid="s1")
    assert [m.content for m in st["messages"] if isinstance(m, HumanMessage)] == ["first question", "second question"]


# ---------- prompts / citations ----------

def test_build_messages_budget_and_empty():
    big = [Chunk(chunk_id=f"c{i}", text="x" * 4000, citation=f"5 U.S.C. 630{i}") for i in range(10)]
    _, used = build_messages("q", big, None, context_tokens=8192, max_answer_tokens=1024)
    assert 0 < len(used) < 10
    msgs, used = build_messages("q", [], None, context_tokens=8192, max_answer_tokens=512)
    assert NO_CONTEXT_NOTE in msgs[-1]["content"] and used == []


def test_citations_exact_finer_and_invented():
    ans = "Six hours [5 U.S.C. 6303(a)(2)] and [5 U.S.C. 6303(a)], not [5 CFR 999.1] or [5 U.S.C. 6303(ab)]."
    assert extract_citations(ans, ["5 U.S.C. 6303(a)"]) == ["5 U.S.C. 6303(a)(2)", "5 U.S.C. 6303(a)"]
    trace = [{"type": "tool_result", "status": "success", "full_content": '{"results": [{"citation": "5 U.S.C. 6304"}]}'}]
    assert _known_citations_from_trace(trace) == ["5 U.S.C. 6304"]


def test_strategy_output_parsing():
    assert parse_strategy_output("few_shot_json", '{"answer": "7 days"}') == "7 days"
    assert parse_strategy_output("cot", "Reasoning: ...\nFinal answer: 7 days.") == "7 days."


# ---------- API ----------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.main.get_settings", lambda: Settings(_env_file=None, llm_base_url="https://llm.test/v1",
                                                                  llm_api_key="k", llm_ca_cert_path=str(REPO_ROOT / ".env.example")))
    monkeypatch.setattr("app.main.build_chat_model", lambda s, **kw: ScriptedChat(script=[]))
    with TestClient(app) as c:
        yield c


def test_agent_endpoint(client):
    client.app.state.agent = agent_for([call("search_policy", {"query": "annual leave"}),
                                        AIMessage(content="6 hours [5 U.S.C. 6303(a)].")], [critique()], retriever=OneChunk())
    r = client.post("/agent", json={"question": "leave with 4 years?"}).json()
    assert r["status"] == "completed" and r["citations"] == ["5 U.S.C. 6303(a)"] and r["session_id"]


def test_chat_endpoint_validates_input(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "q", "strategy": "magic"}).status_code == 422


@pytest.mark.live
async def test_live_agent_without_rag():
    """Real model, no RAG: must run end to end and not invent an answer."""
    from app.config import get_settings
    from app.llm import build_chat_model
    s = get_settings()
    g = build_agent(build_chat_model(s), build_chat_model(s, max_tokens=400), build_tools(NoRAG(), s.data_path()))
    out, _ = await run(g, "How much annual leave can I carry over?", tid="live")
    assert out["tool_calls"] >= 1 and "do not cover" in out["answer"]
