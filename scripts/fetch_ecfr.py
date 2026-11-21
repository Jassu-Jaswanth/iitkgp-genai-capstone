"""Download federal HR regulations (5 CFR) from the eCFR API, one JSON line per section.

Run from the repo root:
    python scripts/fetch_ecfr.py                      # core parts, latest date
    python scripts/fetch_ecfr.py --parts 630 550      # specific parts
    python scripts/fetch_ecfr.py --date 2020-01-01    # point-in-time snapshot

Writes data/raw/ecfr/<date>/part-<N>.xml (raw) and data/raw/ecfr/title5_<date>.jsonl.
Stdlib only. Source is public domain (US government work).
"""

import argparse
import gzip
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://www.ecfr.gov/api/versioner/v1"
UA = "iitkgp-capstone-hr-navigator/0.1 (educational project)"

# Core HR parts: GS pay, pay administration, FLSA/overtime, incentives, hours of duty, leave, adverse actions.
CORE_PARTS = ["531", "550", "551", "575", "610", "630", "752"]


def get(url: str, retries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return data
        except Exception as exc:  # network hiccups or 429/5xx
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  retry in {wait}s ({exc})", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def latest_date() -> str:
    titles = json.loads(get(f"{API}/titles.json"))["titles"]
    return next(t["up_to_date_as_of"] for t in titles if t["number"] == 5)


def clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def element_text(el: ET.Element) -> str:
    return clean("".join(el.itertext()))


def parse_part(xml_bytes: bytes, date: str):
    root = ET.fromstring(xml_bytes)
    part = root if root.get("TYPE") == "PART" else root.find(".//DIV5[@TYPE='PART']")
    part_no = part.get("N")
    part_title = element_text(part.find("HEAD"))

    def walk(node, subpart_title=None, group_title=None):
        for child in node:
            kind = child.get("TYPE")
            if kind == "SUBPART":
                yield from walk(child, element_text(child.find("HEAD")), None)
            elif kind == "SUBJGRP":
                yield from walk(child, subpart_title, element_text(child.find("HEAD")))
            elif kind == "SECTION":
                section = child.get("N")
                head = element_text(child.find("HEAD"))
                paras = [element_text(e) for e in child if e.tag not in ("HEAD", "CITA", "SECAUTH")]
                text = "\n".join(p for p in paras if p)
                cita = child.find("CITA")
                yield {
                    "id": f"5 CFR {section}",
                    "part": part_no,
                    "part_title": part_title,
                    "subpart": subpart_title,
                    "subject_group": group_title,
                    "section": section,
                    "title": head,
                    "text": text,
                    "reserved": "[Reserved]" in head or not text,
                    "source_citation": element_text(cita) if cita is not None else None,
                    "as_of": date,
                    "url": f"https://www.ecfr.gov/on/{date}/title-5/section-{section}",
                }

    yield from walk(part)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", default=CORE_PARTS)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest available)")
    args = ap.parse_args()

    date = args.date or latest_date()
    out_dir = ROOT / "data" / "raw" / "ecfr"
    raw_dir = out_dir / date
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"title5_{date}.jsonl"

    total = kept = 0
    with out_file.open("w") as fh:
        for part in args.parts:
            xml_bytes = get(f"{API}/full/{date}/title-5.xml?part={part}")
            (raw_dir / f"part-{part}.xml").write_bytes(xml_bytes)
            n = 0
            for rec in parse_part(xml_bytes, date):
                total += 1
                if rec["reserved"]:
                    continue
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                n += 1
            print(f"part {part}: {n} sections")
            time.sleep(0.5)  # be polite to the API

    print(f"{kept} sections written ({total - kept} reserved/empty skipped) -> {out_file.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
