"""Curate raw sources into the canonical policy model: document -> section -> clause.

Run from the repo root (after scripts/fetch_*.py):
    python rag/curate.py
Writes data/processed/documents.jsonl (nested) and data/processed/clauses.jsonl (flat, one clause per line).

One adapter per source, no generic PDF parsing (see .context/analysis/data-curation-architecture.md).
Stdlib only.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

MARK = re.compile(r"^((?:\((?:[a-z]{1,2}|\d{1,3}|[ivxl]{1,6}|[A-Z]{1,2})\))+)\s*")
ROMAN = re.compile(r"^[ivxl]+$")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def level_of(token: str, prev_letter: str | None) -> int:
    """(a)=1, (1)=2, (i)=3, (A)=4. '(i)' after '(h)' is the letter i, otherwise roman numeral."""
    if token.isdigit():
        return 2
    if token.isupper():
        return 4
    if ROMAN.match(token) and not (token in ("i", "v", "x", "l") and prev_letter and ord(token) == ord(prev_letter) + 1):
        return 3
    return 1


def legal_clauses(text: str, citation: str, id_prefix: str) -> list[dict]:
    """Split statute/regulation text into clauses by paragraph markers, tracking the (a)(1)(i)(A) path."""
    clauses: list[dict] = []
    path: dict[int, str] = {}
    texts_at: dict[int, str] = {}
    prev_letter: str | None = None
    preamble: list[str] = []
    for line in (ln.strip() for ln in text.split("\n")):
        if not line:
            continue
        m = MARK.match(line)
        if not m:
            if clauses:
                clauses[-1]["text"] += "\n" + line  # continuation of the previous paragraph
            else:
                preamble.append(line)
            continue
        tokens = re.findall(r"\(([^)]+)\)", m.group(1))
        body = line[m.end():].strip()
        for j, tok in enumerate(tokens):
            lvl = level_of(tok, prev_letter)
            if lvl == 1:
                prev_letter = tok
            path = {k: v for k, v in path.items() if k < lvl}
            texts_at = {k: v for k, v in texts_at.items() if k < lvl}
            path[lvl] = tok
            if j < len(tokens) - 1:
                texts_at[lvl] = ""
        lvl_last = max(path)
        marker = "".join(f"({path[k]})" for k in sorted(path))
        lead_in = " ".join(texts_at[k] for k in sorted(texts_at) if k < lvl_last and texts_at[k]).strip()
        clauses.append({"clause_id": f"{id_prefix}-{slug(marker) or 'p'}", "citation": f"{citation}{marker}",
                        "marker": marker, "text": body, "lead_in": lead_in or None})
        texts_at[lvl_last] = body[:300]
    if preamble:
        clauses.insert(0, {"clause_id": f"{id_prefix}-intro", "citation": citation, "marker": "",
                           "text": " ".join(preamble), "lead_in": None})
    if not clauses and text.strip():
        clauses = [{"clause_id": f"{id_prefix}-p", "citation": citation, "marker": "", "text": text.strip(), "lead_in": None}]
    # ids must be unique within a section (repeated markers happen in tables/notes)
    seen = Counter()
    for c in clauses:
        seen[c["clause_id"]] += 1
        if seen[c["clause_id"]] > 1:
            c["clause_id"] += f"-dup{seen[c['clause_id']]}"
    return clauses


# ---------------------------------------------------------------- adapters

def adapt_ecfr(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.open()]
    docs: dict[str, dict] = {}
    for r in rows:
        doc_id = f"cfr5-part-{r['part']}"
        d = docs.setdefault(doc_id, {"doc_id": doc_id, "title": f"5 CFR Part {r['part']}: {r['part_title'].split(' ', 2)[-1]}",
                                     "source_type": "regulation", "authority": "eCFR", "version": r["as_of"],
                                     "as_of": r["as_of"], "url": f"https://www.ecfr.gov/on/{r['as_of']}/title-5/part-{r['part']}",
                                     "sections": []})
        citation = f"5 CFR {r['section']}"
        heading = re.sub(r"^§\s*[\d.]+\s*", "", r["title"]).strip().rstrip(".")
        sid = f"cfr5-{r['section']}"
        d["sections"].append({"section_id": sid, "citation": citation, "heading": heading,
                              "path": [p for p in (r.get("subpart"), r.get("subject_group")) if p],
                              "url": r["url"], "clauses": legal_clauses(r["text"], citation, sid)})
    return list(docs.values())


def adapt_uscode(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.open()]
    if not rows:
        return []
    doc = {"doc_id": "usc5-ch63", "title": "5 U.S.C. Chapter 63: Leave", "source_type": "statute",
           "authority": "US Code (govinfo)", "version": rows[0]["edition"], "as_of": rows[0]["edition"],
           "url": "https://www.govinfo.gov/app/collection/uscode", "sections": []}
    for r in rows:
        citation = f"5 U.S.C. {r['section']}"
        sid = f"usc5-{r['section']}"
        doc["sections"].append({"section_id": sid, "citation": citation, "heading": r["title"],
                                "path": [p for p in (r.get("subchapter"),) if p], "url": r["url"],
                                "clauses": legal_clauses(r["text"], citation, sid)})
    return [doc]


def factsheet_sections(text: str, base_cite: str, id_prefix: str) -> list[dict]:
    """Headings (#..####) -> sections; paragraphs, bullet groups and tables -> clauses."""
    sections: list[dict] = []
    cur = {"heading": "Overview", "blocks": []}

    def flush():
        if cur["blocks"]:
            sections.append(cur.copy())

    block: list[str] = []
    for raw in text.split("\n") + [""]:
        line = raw.strip()
        h = re.match(r"^#{2,6}\s+(.*)", line)
        if h:
            if block:
                cur["blocks"].append(block)
                block = []
            flush()
            cur = {"heading": h.group(1).strip(), "blocks": []}
            continue
        if not line:
            if block:
                cur["blocks"].append(block)
                block = []
            continue
        block.append(line)
    flush()

    out, used = [], Counter()
    for s in sections:
        if s["heading"].lower().startswith("fact sheet:") and len(s["blocks"]) <= 1:
            continue
        clauses: list[dict] = []
        pending_lead: str | None = None
        table: list[str] = []
        for b in s["blocks"]:
            joined = " ".join(b)
            is_bullet = all(x.startswith("- ") for x in b)
            is_table = " | " in joined or joined.endswith("|")
            if is_table:
                table.append(joined.replace(" |", ";").rstrip("; "))
                continue
            if table:
                clauses.append({"text": "\n".join(table), "lead_in": pending_lead, "kind": "table"})
                table, pending_lead = [], None
            if is_bullet:
                items = [x[2:].strip() for x in b]
                if clauses and pending_lead and clauses[-1].get("kind") == "list" and clauses[-1]["lead_in"] == pending_lead:
                    clauses[-1]["text"] += "\n" + "\n".join(f"- {i}" for i in items)
                else:
                    clauses.append({"text": "\n".join(f"- {i}" for i in items), "lead_in": pending_lead, "kind": "list"})
                continue
            pending_lead = joined if joined.rstrip().endswith(("-", ":", "—", "–")) or len(joined) < 60 else None
            if pending_lead and len(joined) < 60 and not joined.rstrip().endswith(("-", ":", "—", "–")):
                continue  # short title line ("Sick Leave Accrual") -> lead-in for the next block
            if pending_lead is None:
                clauses.append({"text": joined, "lead_in": None, "kind": "para"})
        if table:
            clauses.append({"text": "\n".join(table), "lead_in": pending_lead, "kind": "table"})
        if not clauses:
            continue
        sid = f"{id_prefix}-{slug(s['heading'])[:60]}"
        used[sid] += 1
        if used[sid] > 1:
            sid += f"-{used[sid]}"
        for i, c in enumerate(clauses, 1):
            c.update(clause_id=f"{sid}-{i}", citation=f"{base_cite} > {s['heading']}", marker=str(i))
            c.pop("kind", None)
        out.append({"section_id": sid, "citation": f"{base_cite} > {s['heading']}", "heading": s["heading"],
                    "path": [], "clauses": clauses})
    return out


def adapt_factsheets(path: Path) -> list[dict]:
    docs = []
    for r in (json.loads(l) for l in path.open()):
        name = r["url"].rstrip("/").rsplit("/", 1)[-1]
        doc_id = f"opm-fs-{slug(name)}"
        base_cite = f"OPM fact sheet: {r['title']}"
        secs = factsheet_sections(r["text"], base_cite, doc_id)
        for s in secs:
            s["url"] = r["url"]
        if secs:
            docs.append({"doc_id": doc_id, "title": base_cite, "source_type": "guidance", "authority": "OPM",
                         "version": r["retrieved"], "as_of": r["retrieved"], "url": r["url"], "kind": r["kind"],
                         "sections": secs})
    return docs


# ---------------------------------------------------------------- normalize, validate, publish

def normalize(s: str) -> str:
    s = s.replace(" ", " ").replace("‐", "-").replace("‑", "-")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def main() -> int:
    sources = [("ecfr", sorted((RAW / "ecfr").glob("title5_*.jsonl"))[-1:], adapt_ecfr),
               ("uscode", sorted((RAW / "uscode").glob("usc5_chap63_*.jsonl"))[-1:], adapt_uscode),
               ("opm-factsheets", [RAW / "opm" / "factsheets.jsonl"], adapt_factsheets)]
    docs: list[dict] = []
    for name, paths, fn in sources:
        for p in paths:
            if p.exists():
                got = fn(p)
                docs += got
                print(f"{name:15s} {len(got):4d} documents  <- {p.relative_to(ROOT)}")
            else:
                print(f"{name:15s} missing {p}", file=sys.stderr)

    OUT.mkdir(parents=True, exist_ok=True)
    problems: Counter = Counter()
    ids: Counter = Counter()
    n_sections = n_clauses = 0
    with (OUT / "clauses.jsonl").open("w") as flat:
        for d in docs:
            for s in d["sections"]:
                n_sections += 1
                keep = []
                for c in s["clauses"]:
                    c["text"] = normalize(c["text"])
                    if c.get("lead_in"):
                        c["lead_in"] = normalize(c["lead_in"])
                    if not c["text"] or c["text"].lower() in ("[reserved]", "reserved."):
                        problems["empty/reserved dropped"] += 1
                        continue
                    if len(c["text"].split()) > 400:
                        problems["clause > 400 words (kept; chunker must split)"] += 1
                    ids[c["clause_id"]] += 1
                    keep.append(c)
                    flat.write(json.dumps({**c, "section_id": s["section_id"], "section_citation": s["citation"],
                                           "heading": s["heading"], "path": s.get("path", []), "doc_id": d["doc_id"],
                                           "doc_title": d["title"], "source_type": d["source_type"],
                                           "as_of": d["as_of"], "url": s.get("url") or d["url"]}, ensure_ascii=False) + "\n")
                s["clauses"] = keep
                n_clauses += len(keep)
    dupes = [k for k, v in ids.items() if v > 1]
    with (OUT / "documents.jsonl").open("w") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n{len(docs)} documents, {n_sections} sections, {n_clauses} clauses -> data/processed/")
    for k, v in problems.items():
        print(f"  {k}: {v}")
    print(f"  duplicate clause_ids: {len(dupes)}" + (f" e.g. {dupes[:3]}" if dupes else ""))
    return 1 if dupes else 0


if __name__ == "__main__":
    sys.exit(main())
