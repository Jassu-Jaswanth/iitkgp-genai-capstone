"""Download 5 U.S.C. chapter 63 (federal leave statute) from govinfo, one JSON line per section.

Run from the repo root:
    python scripts/fetch_uscode.py                 # chapter 63, 2024 edition
    python scripts/fetch_uscode.py --year 2023

Writes data/raw/uscode/<edition>/chap63.htm (raw) and data/raw/uscode/usc5_chap63_<edition>.jsonl.
Keeps statute text only (drops amendment/editorial notes). Stdlib only. Public domain.
"""

import argparse
import gzip
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = "Mozilla/5.0 (X11; Linux x86_64) iitkgp-capstone-hr-navigator/0.1 (educational project)"
CHAPTER_PATH = "partIII-subpartE-chap63"  # chapter 63 = Leave (subpart E since the 2024 edition)


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
    text = data.decode("utf-8", errors="replace")
    if "Page Not Found" in text[:3000]:
        raise SystemExit(f"govinfo returned 'Page Not Found' for {url}")
    return text


def to_text(fragment: str) -> str:
    fragment = re.sub(r"<br[^>]*>", "\n", fragment)
    fragment = re.sub(r"</p>", "\n", fragment)
    text = html.unescape(re.sub(r"<[^>]+>", "", fragment))
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def field(block: str, name: str) -> str:
    m = re.search(rf"<!-- field-start:{name} -->(.*?)<!-- field-end:{name} -->", block, re.S)
    return m.group(1) if m else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", default="2024", help="US Code edition year on govinfo")
    args = ap.parse_args()

    url = f"https://www.govinfo.gov/content/pkg/USCODE-{args.year}-title5/html/USCODE-{args.year}-title5-{CHAPTER_PATH}.htm"
    page = get(url)

    out_dir = ROOT / "data" / "raw" / "uscode"
    (out_dir / args.year).mkdir(parents=True, exist_ok=True)
    (out_dir / args.year / "chap63.htm").write_text(page)
    out_file = out_dir / f"usc5_chap63_{args.year}.jsonl"

    kept = skipped = 0
    with out_file.open("w") as fh:
        for block in re.split(r"<!-- documentid:", page)[1:]:
            itempath = re.search(r"itempath:(\S+)", block)
            expcite = re.search(r"expcite:(.*?) -->", block)
            head = to_text(field(block, "head"))
            statute = to_text(field(block, "statute"))
            m = re.match(r"§\s*(\d+[a-z]?)\.\s*(.*)", head)
            if not m or not statute:  # chapter/subchapter headings, repealed sections
                skipped += 1
                continue
            section, title = m.group(1), m.group(2)
            subchapter = next((p for p in (expcite.group(1).split("!@!") if expcite else []) if p.startswith("SUBCHAPTER")), None)
            rec = {
                "id": f"5 U.S.C. {section}",
                "section": section,
                "title": title,
                "subchapter": subchapter,
                "text": statute,
                "edition": args.year,
                "itempath": itempath.group(1) if itempath else None,
                "url": f"https://www.govinfo.gov/link/uscode/5/{section}?link-type=html&year={args.year}",
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"{kept} sections written ({skipped} headings/repealed skipped) -> {out_file.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
