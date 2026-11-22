"""Unit tests: real LangGraph wiring, scripted fake chat models (no network, no shared key)."""

from pathlib import Path
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
from app.prompts import NO_CONTEXT_NOTE, build_messages
from app.retrieval import Chunk
from app.tools import PayTable, build_tools, leave_accrual

DATA = REPO_ROOT / "data"
HAVE_OPM = (DATA / "tables" / "gs_pay_2026.csv").exists()


class ScriptedChat(BaseChatModel):
    """Returns scripted AIMessages in order; records what it was sent."""
    script: list[AIMessage]
    seen: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        self.seen.append(messages)
        msg = self.script.pop(0) if self.script else AIMessage(content="(script exhausted)")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kw):
        return self


def call(name, args, i="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{i}", "type": "tool_call"}])


def route(name="hr_policy"):
    return AIMessage(content="", tool_calls=[{"name": "Route", "args": {"reason": "test", "route": name}, "id": "rt", "type": "tool_call"}])


def critique(*issues):
    return AIMessage(content="", tool_calls=[{"name": "Critique", "args": {"issues": list(issues)}, "id": "crit", "type": "tool_call"}])


class EmptyRetriever:
    name = "empty"

    async def retrieve(self, query, k=5, filters=None):
        return []


class OneChunkRetriever:
    name = "one"

    async def retrieve(self, query, k=5, filters=None):
        return [Chunk(chunk_id="c1", text="Annual leave accrues three-fourths day per pay period for 3 to 15 years.",
                      citation="5 U.S.C. 6303(a)", clause_ids=["usc5-6303-a-2"], score=0.9)]


def agent_for(planner_script, critic_script, retriever=None, data_dir=DATA, routes=None, **kw):
    tools = build_tools(retriever or EmptyRetriever(), data_dir)
    router = ScriptedChat(script=routes if routes is not None else [route() for _ in range(5)])
    return build_agent(ScriptedChat(script=planner_script), ScriptedChat(script=critic_script), tools,
                       router_llm=router, **kw)


async def run(graph, q, tid="t", limit=24):
    cfg = {"configurable": {"thread_id": tid}, "recursion_limit": limit}
    await graph.ainvoke({"messages": [HumanMessage(q)], "route": "hr_policy", "critic_rounds": 0, "critique": [],
                         "critic_usage": []}, cfg)
    st = (await graph.aget_state(cfg)).values
    return summarize_turn(st["messages"], st["critic_usage"]), st


# ---------- tools (domain logic) ----------

@pytest.mark.parametrize("years,per_pp,per_year", [(1, 4, 104), (4, 6, 160), (14.9, 6, 160), (15, 8, 208), (30, 8, 208)])
def test_leave_accrual_bands(years, per_pp, per_year):
    r = leave_accrual(years_of_service=years)
    assert r["annual_hours_per_pay_period"] == per_pp and r["annual_hours_per_year"] == per_year
    assert r["max_annual_carryover_hours"] == 240 and r["sick_hours_per_pay_period"] == 4


def test_leave_accrual_caps_and_part_time():
    assert leave_accrual(years_of_service=5, overseas=True)["max_annual_carryover_hours"] == 360
    assert leave_accrual(years_of_service=5, position="ses")["max_annual_carryover_hours"] == 720
    pt = leave_accrual(years_of_service=5, schedule="part-time", hours_in_pay_status=52)
    assert pt["annual_hours_this_pay_period"] == 4.0 and pt["sick_hours_this_pay_period"] == 2.6


@pytest.mark.skipif(not HAVE_OPM, reason="OPM data not downloaded")
def test_pay_lookup_matches_opm_table():
    r = PayTable(DATA / "tables" / "gs_pay_2026.csv").lookup(grade=12, step=5, locality="Washington")
    assert (r["locality_code"], r["annual_salary"], r["overtime_hourly_rate"]) == ("DCB", 116071, 55.89)


def test_pay_lookup_without_data_is_a_tool_error(tmp_path: Path):
    from langchain_core.tools import ToolException
    with pytest.raises(ToolException, match="pay table not loaded"):
        PayTable(tmp_path / "missing.csv").lookup(grade=12, step=5)


# ---------- agent graph ----------

async def test_agent_uses_calculator_and_passes_critic():
    g = agent_for([call("leave_accrual_calc", {"years_of_service": 4}),
                   AIMessage(content="6 hours per pay period [5 U.S.C. 6303(a)]; carryover 240 hours [5 U.S.C. 6304].")],
                  [critique()])
    s, st = await run(g, "How much leave with 4 years?")
    assert s["answer"].startswith("6 hours") and s["tool_calls"] == 1 and st["critique"] == []
    result = next(t for t in s["trace"] if t["type"] == "tool_result")
    assert result["status"] == "success" and '"annual_hours_per_pay_period": 6' in result["content"]


async def test_bad_tool_args_become_observation_not_crash():
    g = agent_for([call("pay_lookup", {"grade": 99, "step": 5}),  # grade out of range -> validation error
                   call("search_policy", {"query": "GS grade range"}, "2"),
                   AIMessage(content="The provided policy documents do not cover this.")], [])
    s, _ = await run(g, "pay for GS-99?")
    err = next(t for t in s["trace"] if t["type"] == "tool_result")
    assert err["status"] == "error" and s["tool_errors"] == 1 and "cover this" in s["answer"]


async def test_critic_feedback_triggers_one_revision():
    g = agent_for([call("search_policy", {"query": "annual leave accrual"}),
                   AIMessage(content="8 hours per pay period [5 U.S.C. 6303(a)]."),
                   AIMessage(content="6 hours (three-fourths day) per pay period [5 U.S.C. 6303(a)].")],
                  [critique("8 hours contradicts three-fourths day"), critique()], retriever=OneChunkRetriever())
    s, st = await run(g, "How much leave with 4 years?")
    assert s["answer"].startswith("6 hours") and st["critic_rounds"] == 2
    assert [t["type"] for t in s["trace"]].count("critic_feedback") == 1


async def test_unsupported_answer_is_flagged_without_llm_critic():
    g = agent_for([AIMessage(content="Carryover is 260 hours."), AIMessage(content="The provided policy documents do not cover this.")], [])
    s, st = await run(g, "carryover after 16 years?")
    assert "cover this" in s["answer"] and any(t["type"] == "critic_feedback" for t in s["trace"])


async def test_step_limit_stops_runaway_loop():
    g = agent_for([call("holiday_calendar", {}, str(i)) for i in range(50)], [])
    with pytest.raises(GraphRecursionError):
        await run(g, "loop", limit=6)


async def test_session_memory_keeps_history():
    nc = "The provided policy documents do not cover this."
    g = agent_for([call("search_policy", {"query": "first rule"}, "1"), AIMessage(content=nc),
                   call("search_policy", {"query": "second rule"}, "2"), AIMessage(content=nc)], [])
    await run(g, "first question", tid="s1")
    _, st = await run(g, "second question", tid="s1")
    humans = [m.content for m in st["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["first question", "second question"]


async def test_router_privacy_short_circuits_without_tools():
    g = agent_for([call("pay_lookup", {"grade": 12, "step": 5})], [], routes=[route("individual_records")])
    s, _ = await run(g, "What is Sarah's salary?")
    assert s["route"] == "individual_records" and s["tool_calls"] == 0 and "private" in s["answer"]


async def test_router_out_of_scope_and_failure_fallback():
    g = agent_for([], [], routes=[route("out_of_scope")])
    s, _ = await run(g, "Google 401k match?")
    assert s["route"] == "out_of_scope" and "do not cover" in s["answer"]
    g = agent_for([AIMessage(content="The provided policy documents do not cover this.")], [],
                  routes=[AIMessage(content="not json")])  # router output unparseable -> falls through to planner
    s, _ = await run(g, "anything")
    assert s["route"] == "hr_policy"


async def test_not_covered_without_search_is_sent_back():
    g = agent_for([AIMessage(content="The provided policy documents do not cover this."),
                   call("search_policy", {"query": "fmla eligibility"}, "2"),
                   AIMessage(content="12 months of service [5 U.S.C. 6381(1)(B)].")], [critique()],
                  retriever=OneChunkRetriever())
    s, st = await run(g, "FMLA eligibility?")
    assert any(t["type"] == "critic_feedback" for t in s["trace"]) and s["answer"].startswith("12 months")


# ---------- prompts / citations ----------

def chunk(i, text="Annual leave accrues at 6 hours per pay period.", cite=None):
    return Chunk(chunk_id=f"c{i}", text=text, citation=cite or f"5 U.S.C. 630{i}", clause_ids=[f"cl{i}"])


def test_build_messages_budget_and_empty():
    msgs, used = build_messages("q", [chunk(i, text="x" * 4000) for i in range(10)], None, context_tokens=8192, max_answer_tokens=1024)
    assert 0 < len(used) < 10
    msgs, used = build_messages("q", [], None, context_tokens=8192, max_answer_tokens=512)
    assert NO_CONTEXT_NOTE in msgs[-1]["content"] and used == []


def test_extract_citations_exact_finer_and_invented():
    known = ["5 U.S.C. 6303(a)"]
    ans = "Six hours [5 U.S.C. 6303(a)(2)] and [5 U.S.C. 6303(a)], not [5 CFR 999.1] or [5 U.S.C. 6303(ab)]."
    assert extract_citations(ans, known) == ["5 U.S.C. 6303(a)(2)", "5 U.S.C. 6303(a)"]


def test_known_citations_read_from_tool_json():
    trace = [{"type": "tool_result", "status": "success",
              "full_content": '{"citation": "5 U.S.C. 6303(f); 5 CFR 630.301", "carryover_citation": "5 U.S.C. 6304"}'}]
    assert _known_citations_from_trace(trace) == ["5 U.S.C. 6303(f)", "5 CFR 630.301", "5 U.S.C. 6304"]


# ---------- API ----------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.main.get_settings", lambda: Settings(_env_file=None, llm_base_url="https://llm.test/v1",
                                                                  llm_api_key="k", llm_ca_cert_path=str(REPO_ROOT / ".env.example")))
    monkeypatch.setattr("app.main.build_chat_model", lambda s, **kw: ScriptedChat(script=[]))
    with TestClient(app) as c:
        yield c


def test_agent_endpoint(client):
    client.app.state.agent = agent_for([call("leave_accrual_calc", {"years_of_service": 4}),
                                        AIMessage(content="6 hours [5 U.S.C. 6303(a)].")], [critique()])
    r = client.post("/agent", json={"question": "leave with 4 years?"}).json()
    assert r["status"] == "completed" and r["citations"] == ["5 U.S.C. 6303(a)"] and r["tool_calls"] == 1
    assert r["session_id"] and all("full_content" not in t for t in r["trace"])

    # the LLM trace of that answer, from LangGraph's event stream, kept by the (in-memory) store
    steps = client.get(f"/turns/{r['turn_id']}/trace").json()
    kinds = [(s["kind"], s["node"]) for s in steps]
    assert ("tool", "tools") in kinds and ("llm", "planner") in kinds
    tool = next(s for s in steps if s["kind"] == "tool")
    assert tool["name"] == "leave_accrual_calc" and tool["input"] == {"years_of_service": 4}
    assert tool["output"]["annual_hours_per_pay_period"] == 6  # tool JSON is stored parsed, not as a string
    plan = next(s for s in steps if s["kind"] == "llm" and s["node"] == "planner")
    assert plan["output"]["tool_calls"] == [{"name": "leave_accrual_calc", "args": {"years_of_service": 4}}]

    # history
    conv = client.get("/conversations").json()
    assert conv[0]["id"] == r["session_id"] and conv[0]["messages"] == 2
    msgs = client.get(f"/conversations/{r['session_id']}/messages").json()
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["turn_id"] == r["turn_id"]
    assert client.get("/turns").json()[0]["tools"] == ["leave_accrual_calc"]
    assert client.delete(f"/conversations/{r['session_id']}").status_code == 200
    assert client.get("/conversations").json() == []


def test_chat_endpoint_validates_input(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422


# ---------- live (opt-in: pytest -m live; uses the shared key) ----------

@pytest.mark.live
async def test_live_agent_leave_question():
    from app.config import get_settings
    from app.llm import build_chat_model
    s = get_settings()
    g = build_agent(build_chat_model(s), build_chat_model(s, max_tokens=400), build_tools(EmptyRetriever(), s.data_path()))
    out, _ = await run(g, "Full-time, 4 years of service: annual leave hours per pay period?", tid="live")
    assert "6" in out["answer"] and any(t.get("tool") == "leave_accrual_calc" for t in out["trace"])


def test_strategy_output_parsing():
    from app.prompts import parse_strategy_output
    assert parse_strategy_output("few_shot_json", '{"answer": "7 days [5 U.S.C. 6327(b)(1)]", "citations": []}') == "7 days [5 U.S.C. 6327(b)(1)]"
    assert parse_strategy_output("cot", "Reasoning: excerpt says 7.\nFinal answer: 7 days.") == "7 days."
    assert parse_strategy_output("few_shot_json", "not json at all") == "not json at all"


def test_chat_rejects_unknown_strategy(client):
    assert client.post("/chat", json={"question": "q", "strategy": "magic"}).status_code == 422


def test_retriever_mode_selection():
    from app.retrieval import build_retriever
    assert build_retriever().name == "placeholder"
    assert build_retriever("http://rag:8200").name == "http"
    assert build_retriever("http://rag:8200", mode="placeholder").name == "placeholder"
    with pytest.raises(ValueError):
        build_retriever(mode="http")  # RETRIEVAL_MODE=http without RAG_SERVICE_URL


# ---------- caching ----------

def test_answer_cache_serves_repeat_first_question(client):
    client.app.state.agent = agent_for([call("leave_accrual_calc", {"years_of_service": 4}),
                                        AIMessage(content="6 hours [5 U.S.C. 6303(a)]."),
                                        AIMessage(content="(follow-up answer)")], [critique(), critique()])
    first = client.post("/agent", json={"question": "Leave with 4 years?"}).json()
    assert first["cached_from"] is None and first["usage"]["llm_calls"] > 0

    again = client.post("/agent", json={"question": "  leave with 4 YEARS "}).json()  # case/spacing ignored
    assert again["cached_from"] == first["turn_id"] and again["answer"] == first["answer"]
    assert again["usage"]["llm_calls"] == 0 and again["session_id"] != first["session_id"]
    assert client.get(f"/turns/{again['turn_id']}/trace").json()  # the original trace comes with it

    # the cached exchange is in the agent's memory, so a follow-up in that conversation has context
    follow = client.post("/agent", json={"question": "and with 20 years?", "session_id": again["session_id"]}).json()
    assert follow["cached_from"] is None
    import asyncio
    vals = asyncio.run(client.app.state.agent.aget_state({"configurable": {"thread_id": again["session_id"]}})).values
    texts = [m.content for m in vals["messages"]]
    assert "  leave with 4 YEARS " in texts and first["answer"] in texts

    # eval runs bypass the cache
    fresh = client.post("/agent", json={"question": "Leave with 4 years?", "use_cache": False}).json()
    assert fresh["cached_from"] is None


def test_escalations_are_not_cached(client):
    client.app.state.agent = agent_for([call("escalate_to_hr", {"summary": "Dispute about my performance rating"}),
                                        AIMessage(content="Drafted HR case; pending approval."),
                                        call("escalate_to_hr", {"summary": "Dispute about my performance rating"}),
                                        AIMessage(content="Drafted HR case; pending approval.")],
                                       [critique(), critique()])
    a = client.post("/agent", json={"question": "Please escalate my rating dispute"}).json()
    b = client.post("/agent", json={"question": "Please escalate my rating dispute"}).json()
    assert a["tool_calls"] == 1 and b["cached_from"] is None and b["tool_calls"] == 1


def test_search_cache_reuses_results():
    from app.cache import LRUCache
    from app.retrieval import LangChainRetrieverAdapter, LocalHybridRetriever

    class Engine:
        calls = 0

        def search(self, query, k=5, mode="", filters=None):
            Engine.calls += 1
            return [{"chunk_id": "c1", "text": "t", "citation": "5 U.S.C. 6303(a)", "clause_ids": ["x"], "score": 1.0}]

    r = LangChainRetrieverAdapter(LocalHybridRetriever(engine=Engine(), cache=LRUCache(8)), "test")
    import asyncio
    a = asyncio.run(r.retrieve("Annual leave", k=3))
    b = asyncio.run(r.retrieve("annual   LEAVE", k=3))
    assert Engine.calls == 1 and a == b and r.cache_stats()["hits"] == 1
    asyncio.run(r.retrieve("annual leave", k=5))  # different k -> new search
    assert Engine.calls == 2
