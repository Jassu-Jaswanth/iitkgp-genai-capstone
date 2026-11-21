"""Agent tools as LangChain StructuredTools with Pydantic argument schemas.

Domain logic (leave rules, pay/holiday tables) is ours; argument validation, schema generation and error
propagation are LangChain's. Errors raise ToolException, which LangGraph's ToolNode turns into a
ToolMessage the model can react to ("a failure is an observation, not a crash").
"""

import csv
import datetime as dt
import uuid
from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

from .retrieval import Retriever

# ---------------------------------------------------------------- leave accrual (5 U.S.C. 6303, 6304, 6307)


class LeaveArgs(BaseModel):
    years_of_service: float = Field(ge=0, le=60, description="Creditable years of federal service")
    schedule: Literal["full-time", "part-time"] = "full-time"
    position: Literal["general", "ses"] = Field("general", description="'ses' for SES, SL, ST and equivalent")
    hours_in_pay_status: float | None = Field(None, ge=0, le=80, description="Part-time only: hours in pay status this pay period")
    overseas: bool = Field(False, description="Stationed overseas (higher carryover cap)")


def leave_accrual(years_of_service: float, schedule: str = "full-time", position: str = "general",
                  hours_in_pay_status: float | None = None, overseas: bool = False) -> dict:
    years = years_of_service
    band = "under 3 years" if years < 3 else ("3 to under 15 years" if years < 15 else "15 or more years")
    out: dict = {"years_of_service": years, "service_band": band, "schedule": schedule}
    if position == "ses":
        out.update(annual_hours_per_pay_period=8, annual_hours_per_year=208, citation="5 U.S.C. 6303(f); 5 CFR 630.301")
    elif schedule == "full-time":
        per_pp = 4 if years < 3 else (6 if years < 15 else 8)
        out.update(annual_hours_per_pay_period=per_pp, annual_hours_per_year={4: 104, 6: 160, 8: 208}[per_pp],
                   citation="5 U.S.C. 6303(a)", sick_hours_per_pay_period=4, sick_citation="5 U.S.C. 6307(a)")
        if per_pp == 6:
            out["note"] = "10 hours in the last full biweekly pay period of the leave year"
    else:
        divisor = 20 if years < 3 else (13 if years < 15 else 10)
        out.update(annual_rule=f"1 hour for each {divisor} hours in a pay status", citation="5 U.S.C. 6303(a)",
                   sick_rule="1 hour for each 20 hours in a pay status")
        if hours_in_pay_status is not None:
            out["annual_hours_this_pay_period"] = round(hours_in_pay_status / divisor, 2)
            out["sick_hours_this_pay_period"] = round(hours_in_pay_status / 20, 2)
    out["max_annual_carryover_hours"] = 720 if position == "ses" else (360 if overseas else 240)
    out["carryover_citation"] = "5 U.S.C. 6304"
    return out


# ---------------------------------------------------------------- GS pay lookup (OPM 2026 tables)

LOCALITY_NAMES = {
    "RUS": "Rest of U.S.", "DCB": "Washington-Baltimore-Arlington", "NY": "New York-Newark",
    "SF": "San Jose-San Francisco-Oakland", "LA": "Los Angeles-Long Beach", "CHI": "Chicago-Naperville",
    "BOS": "Boston-Worcester-Providence", "HOU": "Houston-The Woodlands", "ATL": "Atlanta-Athens-Clarke County-Sandy Springs",
    "SEA": "Seattle-Tacoma", "DEN": "Denver-Aurora", "PHL": "Philadelphia-Reading-Camden", "DFW": "Dallas-Fort Worth",
    "SD": "San Diego-Carlsbad", "DET": "Detroit-Warren-Ann Arbor", "MSP": "Minneapolis-St. Paul",
    "MFL": "Miami-Fort Lauderdale-Port St. Lucie", "AK": "Alaska", "HI": "Hawaii",
}


class PayArgs(BaseModel):
    grade: int = Field(ge=1, le=15, description="GS grade 1-15")
    step: int = Field(ge=1, le=10, description="Step 1-10")
    locality: str = Field("RUS", description="Locality code (RUS, DCB, NY, SF...) or area name")


class PayTable:
    def __init__(self, csv_path: Path):
        self.path = csv_path
        self.rows: dict[tuple[str, int], dict[str, str]] = {}
        if csv_path.exists():
            for r in csv.DictReader(csv_path.open()):
                self.rows[(r["LOCNAME"].upper(), int(r["GRADE"]))] = r

    def lookup(self, grade: int, step: int, locality: str = "RUS") -> dict:
        if not self.rows:
            raise ToolException(f"pay table not loaded ({self.path.name} missing); data team must provide it")
        code = locality.strip().upper()
        if (code, grade) not in self.rows:
            matches = [c for c, n in LOCALITY_NAMES.items() if locality.lower() in n.lower()]
            if len(matches) != 1:
                raise ToolException(f"unknown locality {locality!r}; use a code like RUS (Rest of U.S.), DCB, NY, SF, CHI")
            code = matches[0]
        r = self.rows[(code, grade)]
        return {"grade": f"GS-{grade}", "step": step, "locality_code": code, "locality": LOCALITY_NAMES.get(code, code),
                "year": 2026, "annual_salary": int(float(r[f"ANNUAL{step}"])), "hourly_rate": float(r[f"HOURLY{step}"]),
                "overtime_hourly_rate": float(r[f"OVERTIME{step}"]), "source": "OPM 2026 General Schedule pay tables"}


# ---------------------------------------------------------------- federal holidays (OPM)

class HolidayArgs(BaseModel):
    start_date: dt.date | None = Field(None, description="YYYY-MM-DD; default today")
    end_date: dt.date | None = Field(None, description="YYYY-MM-DD; default start + 1 year")


class HolidayCalendar:
    def __init__(self, csv_path: Path):
        self.path = csv_path
        self.days = ([(dt.date.fromisoformat(r["date"]), r["holiday"]) for r in csv.DictReader(csv_path.open())]
                     if csv_path.exists() else [])

    def query(self, start_date: dt.date | None = None, end_date: dt.date | None = None) -> dict:
        if not self.days:
            raise ToolException(f"holiday calendar not loaded ({self.path.name} missing)")
        start = start_date or dt.date.today()
        end = end_date or start + dt.timedelta(days=365)
        if end < start:
            raise ToolException("end_date is before start_date")
        hits = [{"date": d.isoformat(), "weekday": d.strftime("%A"), "holiday": n} for d, n in self.days if start <= d <= end]
        return {"start_date": start.isoformat(), "end_date": end.isoformat(), "count": len(hits), "holidays": hits[:20]}


# ---------------------------------------------------------------- search + escalation

class SearchArgs(BaseModel):
    query: str = Field(min_length=3, description="Name of the rule to look up, e.g. 'annual leave carryover limit'")
    k: int = Field(5, ge=1, le=10)


class EscalateArgs(BaseModel):
    summary: str = Field(min_length=10, description="What the employee needs decided and why")


CONSEQUENTIAL_TOOLS = {"escalate_to_hr"}  # human-in-the-loop boundary


def build_tools(retriever: Retriever, data_dir: Path) -> list[BaseTool]:
    pay = PayTable(data_dir / "opm" / "gs_pay_2026.csv")
    cal = HolidayCalendar(data_dir / "opm" / "holidays.csv")

    async def search_policy(query: str, k: int = 5) -> dict:
        chunks = await retriever.retrieve(query, k=k)
        if not chunks:
            return {"results": [], "note": f"no policy excerpts found (retriever: {retriever.name})"}
        return {"results": [{"citation": c.citation, "text": c.text, "clause_ids": c.clause_ids,
                             "score": round(c.score, 4)} for c in chunks]}

    def escalate_to_hr(summary: str) -> dict:
        return {"status": "pending_human_approval", "ticket_id": f"HR-{uuid.uuid4().hex[:8].upper()}",
                "summary": summary, "note": "Draft only. Not submitted until an HR specialist approves."}

    return [
        StructuredTool.from_function(coroutine=search_policy, name="search_policy", args_schema=SearchArgs,
            description="Search federal HR statutes, regulations and OPM guidance. Query with the NAME OF THE RULE, "
                        "not the user's whole story. Returns cited excerpts."),
        StructuredTool.from_function(func=leave_accrual, name="leave_accrual_calc", args_schema=LeaveArgs,
            description="Compute annual/sick leave accrual and the annual leave carryover cap. Use for any leave arithmetic."),
        StructuredTool.from_function(func=pay.lookup, name="pay_lookup", args_schema=PayArgs,
            description="2026 General Schedule annual salary, hourly and overtime rate by grade, step and locality."),
        StructuredTool.from_function(func=cal.query, name="holiday_calendar", args_schema=HolidayArgs,
            description="List federal holidays between two dates."),
        StructuredTool.from_function(func=escalate_to_hr, name="escalate_to_hr", args_schema=EscalateArgs,
            description="Draft an HR case for anything needing a human decision (disputes, disciplinary actions, "
                        "individual eligibility). Never auto-submitted."),
    ]
