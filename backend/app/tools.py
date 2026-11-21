"""Agent tools. A tool = a Pydantic args schema + a Python function, wrapped as a LangChain StructuredTool.

How it works (ELI5): the model reads each tool's name + description + args, decides to "call" one, LangGraph's
ToolNode runs the Python function, and the result goes back to the model as an observation. If the function
raises ToolException (or the args are invalid), the model sees the error and can try again; nothing crashes.

There is ONE working example below (`search_policy`). The other tools are TODO stubs: see docs/TODO.md.
"""

from pathlib import Path
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool, ToolException  # noqa: F401  (ToolException: use in new tools)
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from .retrieval import Retriever

# Tools that need a human to approve before anything real happens (human-in-the-loop boundary).
CONSEQUENTIAL_TOOLS = {"escalate_to_hr"}


# ---------------------------------------------------------------- example tool: search_policy

class SearchArgs(BaseModel):
    query: str = Field(min_length=3, description="Name of the rule to look up, e.g. 'annual leave carryover limit'")
    k: int = Field(5, ge=1, le=10)
    # Injected by LangGraph, hidden from the model: lets the tool also search with the user's own words.
    state: Annotated[dict, InjectedState] = Field(default_factory=dict)


def latest_user_question(state: dict) -> str:
    for m in reversed(state.get("messages", []) if state else []):
        if isinstance(m, HumanMessage) and not m.additional_kwargs.get("critic"):
            return str(m.content)
    return ""


def make_search_policy(retriever: Retriever) -> BaseTool:
    async def search_policy(query: str, k: int = 5, state: dict | None = None) -> dict:
        # Search with BOTH the model's query and the user's question, then interleave. In our tests the model's
        # short "rule name" queries sometimes found the wrong rule; the user's wording is a cheap safety net.
        queries = [query]
        user_q = latest_user_question(state or {})
        if user_q and user_q.strip().lower() != query.strip().lower():
            queries.insert(0, user_q)
        lists = [await retriever.retrieve(q, k=k) for q in queries]
        seen, chunks = set(), []
        for rank in range(k):
            for lst in lists:
                if rank < len(lst) and lst[rank].chunk_id not in seen:
                    seen.add(lst[rank].chunk_id)
                    chunks.append(lst[rank])
        if not chunks:  # no RAG service yet (placeholder) or nothing relevant
            return {"results": [], "note": f"no policy excerpts found (retriever: {retriever.name})"}
        return {"results": [{"citation": c.citation, "text": c.text, "clause_ids": c.clause_ids,
                             "score": round(c.score, 4)} for c in chunks[: k + 2]]}

    return StructuredTool.from_function(
        coroutine=search_policy, name="search_policy", args_schema=SearchArgs,
        description="Search federal HR statutes, regulations and OPM guidance for RULES: eligibility, limits, waiting "
                    "periods, entitlements, premium pay rates, procedures. Query with the NAME OF THE RULE. Returns "
                    "cited excerpts.")


# ---------------------------------------------------------------- TODO tools (not registered yet)
#
# TODO(tools): leave_accrual_calc  (deterministic leave maths; the 8B model gets arithmetic wrong)
#   args:  years_of_service: float, schedule: "full-time"|"part-time", position: "general"|"ses",
#          hours_in_pay_status: float | None (part-time), overseas: bool
#   rules: 5 U.S.C. 6303(a): <3 yrs 4 h/pay period, 3-15 yrs 6 h (10 h last pay period), 15+ yrs 8 h;
#          part-time 1 h per 20/13/10 h in pay status; sick leave 4 h/pp (6307(a)), part-time 1 per 20 h;
#          carryover cap 240 h (6304(a)), 360 h overseas (6304(b)), 720 h SES (6304(f)).
#   return: numbers + a "citation" field (the critic and the citation checker read it).
#   note:  years_of_service is optional for carryover-only questions, otherwise the model invents a value.
#
# TODO(tools): pay_lookup  (2026 GS salary/hourly/overtime by grade, step, locality)
#   data:  data/processed/opm/gs_pay_2026.csv (columns LOCNAME, GRADE, ANNUAL1..10, HOURLY1..10, OVERTIME1..10)
#   args:  grade 1-15, step 1-10, locality code (RUS, DCB, NY, SF...) or area name
#   rule:  description must say "dollar amounts only, never guess a grade".
#
# TODO(tools): holiday_calendar  (federal holidays between two dates)
#   data:  data/processed/opm/holidays.csv (date, holiday), 2021-2030
#
# TODO(tools): escalate_to_hr  (human-in-the-loop: DRAFT a case, never submit; return a ticket id + "pending")
#
# After adding a tool: register it in build_tools(), add a "question type -> tool" line to PLANNER_RULES in
# agent.py, add tests (backend/tests) and gold items that expect it (eval/data).


def build_tools(retriever: Retriever, data_dir: Path) -> list[BaseTool]:
    """All tools the agent may call. `data_dir` (data/processed) holds tables for the TODO tools."""
    tools: list[BaseTool] = [make_search_policy(retriever)]
    # TODO(tools): tools += [leave_accrual_calc, pay_lookup(data_dir), holiday_calendar(data_dir), escalate_to_hr]
    return tools
