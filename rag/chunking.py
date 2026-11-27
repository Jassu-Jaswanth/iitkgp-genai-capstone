"""Chunking strategies over the team corpus data/corpus/clauses.jsonl (FedHR-Policy, docs/DATASET.md).
Token counts use the bge-small tokenizer (the upGrad embedding endpoint's model; limit 512 tokens per text).

Every chunk keeps the clause IDs it contains, so recall@k can be scored against gold_clause_ids.
"""

import json
import os
from collections import defaultdict

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # see retriever.py
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
EMBED_MAX_TOKENS = 512


@lru_cache(maxsize=1)
def tokenizer():
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    return Tokenizer.from_file(hf_hub_download("BAAI/bge-small-en-v1.5", "tokenizer.json"))


def ntok(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


def split_tokens(text: str, size: int, overlap: int) -> list[str]:
    enc = tokenizer().encode(text, add_special_tokens=False)
    if len(enc.ids) <= size:
        return [text]
    out, start = [], 0
    while start < len(enc.ids):
        end = min(start + size, len(enc.ids))
        a, b = enc.offsets[start][0], enc.offsets[end - 1][1]
        out.append(text[a:b])
        if end == len(enc.ids):
            break
        start = end - overlap
    return out


MAX_LEAD_TOKENS = 80  # some official lead-ins run to 230+ tokens; a long lead-in pushed chunks past the 512 limit


def cap_tokens(text: str, n: int) -> str:
    """Trim text to at most n bge tokens (keeps the start; adds an ellipsis when cut)."""
    if not text:
        return text
    enc = tokenizer().encode(text, add_special_tokens=False)
    return text if len(enc.ids) <= n else text[:enc.offsets[n - 1][1]].rstrip() + " …"


def load(name: str = "clauses.jsonl") -> list[dict]:
    with (CORPUS / name).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def header(c: dict) -> str:
    """Contextual header '[section citation] heading' (helps BM25 + dense on short clauses)."""
    return f"[{c.get('section_citation') or c['citation']}] {c.get('heading') or ''}".rstrip() + "\n"


def _chunk(cid: str, text: str, clauses: list[dict], strategy: str) -> dict:
    first = clauses[0]
    cites = [c["citation"] for c in clauses]
    citation = cites[0] if len(cites) == 1 else f"{cites[0]} to {cites[-1].split(' ')[-1]}" \
        if first["source_type"] != "guidance" else first["citation"]
    return dict(chunk_id=cid, text=text, citation=citation,
                clause_ids=list(dict.fromkeys(c["clause_id"] for c in clauses)),
                section_id=first["section_id"], source_type=first["source_type"],
                source_url=first.get("url"), title=first.get("doc_title"), heading=first.get("heading"),
                strategy=strategy, tokens=ntok(text))


def clause_pack(clauses: list[dict], budget: int = 350, with_header: bool = True, name: str = "clause-pack-350") -> list[dict]:
    """Pack whole clauses of ONE section until the token budget; repeat a parent lead-in only if missing."""
    by_sec = defaultdict(list)
    for c in clauses:
        by_sec[c["section_id"]].append(c)
    chunks = []
    for sid, cl in by_sec.items():
        hdr = header(cl[0]) if with_header else ""
        buf, members, n = [], [], 0
        part = 0

        def flush():
            nonlocal buf, members, part
            if members:
                chunks.append(_chunk(f"{name}::{sid}::{part}", hdr + "\n".join(buf), members, name))
                part += 1
            buf, members = [], []

        for c in cl:
            body = c["text"]
            lead = cap_tokens(c.get("lead_in") or "", MAX_LEAD_TOKENS)
            pieces = split_tokens(body, budget - 40, 40) if ntok(body) > budget - 40 else [body]
            for i, p in enumerate(pieces):
                need_lead = lead and not any(lead[:60] in b for b in buf)
                txt = (lead + "\n" + p) if need_lead else p
                t = ntok(txt)
                if members and n + t > budget:
                    flush()
                    n = 0
                    txt = (lead + "\n" + p) if lead else p
                    t = ntok(txt)
                buf.append(txt)
                members.append(c)
                n += t
                if len(pieces) > 1 and i < len(pieces) - 1:
                    flush()
                    n = 0
        flush()
    return chunks


def section_split(clauses: list[dict], size: int = 256, overlap: int = 32, with_header: bool = True) -> list[dict]:
    """Baseline: concatenate a section, split on token windows (ignores clause boundaries)."""
    name = f"token-{size}/{overlap}" + ("+header" if with_header else "")
    by_sec = defaultdict(list)
    for c in clauses:
        by_sec[c["section_id"]].append(c)
    chunks = []
    for sid, cl in by_sec.items():
        hdr = header(cl[0]) if with_header else ""
        full, spans = "", []
        for c in cl:
            start = len(full)
            full += c["text"] + "\n"
            spans.append((start, len(full), c))
        for i, piece in enumerate(split_tokens(full, size, overlap)):
            a = full.find(piece[:50])
            b = a + len(piece)
            members = [c for s, e, c in spans if s < b and e > a] or [cl[0]]
            chunks.append(_chunk(f"{name}::{sid}::{i}", hdr + piece, members, name))
    return chunks


STRATEGIES = {
    "clause-pack-350": lambda cl: clause_pack(cl, 350, True, "clause-pack-350"),
    "clause-pack-350-nohdr": lambda cl: clause_pack(cl, 350, False, "clause-pack-350-nohdr"),
    "clause-pack-200": lambda cl: clause_pack(cl, 200, True, "clause-pack-200"),
    "clause-pack-256": lambda cl: clause_pack(cl, 256, True, "clause-pack-256"),
    "token-256/32+header": lambda cl: section_split(cl, 256, 32, True),
    "token-256/32": lambda cl: section_split(cl, 256, 32, False),
}
