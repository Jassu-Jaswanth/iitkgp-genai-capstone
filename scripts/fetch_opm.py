"""Download OPM pay & leave guidance: fact sheets, handbooks, GS pay tables and federal holidays.

Run from the repo root:
    python scripts/fetch_opm.py                     # everything
    python scripts/fetch_opm.py --only factsheets   # or: handbooks, pay, holidays
    python scripts/fetch_opm.py --pay-year 2025

Outputs under data/raw/opm/:
    factsheets.jsonl     one line per fact sheet (markdown-ish text with headings kept)
    handbooks/*.pdf      raw PDFs, plus handbooks.jsonl if `pdftotext` (poppler-utils) is installed
    gs_pay_<year>.csv    GS base pay table (grade, step, annual rate) from OPM's XLSX
    holidays.csv         federal holidays from OPM's iCalendar feed
Stdlib only (pdftotext optional). Public domain (US government work).
"""

import argparse
import csv
import gzip
import html
import io
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "opm"
BASE = "https://www.opm.gov"
PAY_LEAVE = f"{BASE}/policy-data-oversight/pay-leave"
UA = "iitkgp-capstone-hr-navigator/0.1 (educational project)"

SEEN_PDFS: set[str] = set()

FACT_SHEET_INDEXES = {
    "leave": f"{PAY_LEAVE}/leave-administration/fact-sheets/",
    "pay": f"{PAY_LEAVE}/pay-administration/fact-sheets/",
}


def get(url: str, retries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return data
        except urllib.error.HTTPError as exc:
            if exc.code == 404 or attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


# ---------- HTML -> text ----------

def html_to_text(fragment: str) -> str:
    fragment = re.sub(r"(?is)<(script|style|nav|form|button|svg)\b.*?</\1>", "", fragment)
    for level in range(1, 5):
        fragment = re.sub(rf"(?is)<h{level}[^>]*>(.*?)</h{level}>", lambda m, l=level: f"\n\n{'#' * (l + 1)} {m.group(1)}\n", fragment)
    fragment = re.sub(r"(?is)<li[^>]*>", "\n- ", fragment)
    fragment = re.sub(r"(?is)</(p|div|tr|table|ul|ol)>|<br[^>]*>", "\n", fragment)
    fragment = re.sub(r"(?is)</t[dh]>", " | ", fragment)
    text = html.unescape(re.sub(r"<[^>]+>", "", fragment))
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main_content(page: str) -> str:
    start = page.find('id="main-main-content"')
    if start < 0:
        start = page.find("<main")
    start = page.find(">", start) + 1  # skip the rest of the opening tag
    end = page.find("</main>", start)
    return page[start:end if end > 0 else None]


# ---------- fact sheets ----------

def fetch_factsheets() -> None:
    rows, pdf_links = [], set()
    for kind, index_url in FACT_SHEET_INDEXES.items():
        index = get(index_url).decode("utf-8", "replace")
        path = urllib.parse.urlparse(index_url).path
        links = sorted(set(re.findall(rf'href="({re.escape(path)}[^"#?]+)"', index)))
        for link in links:
            url = urllib.parse.urljoin(BASE, link)
            if url.lower().endswith(".pdf"):
                pdf_links.add(url)
                continue
            try:
                page = get(url).decode("utf-8", "replace")
            except urllib.error.HTTPError as exc:
                print(f"  skip {url}: HTTP {exc.code}")
                continue
            title = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", page)
            text = html_to_text(main_content(page))
            if len(text) < 300:
                continue
            slug = link.rstrip("/").rsplit("/", 1)[-1]
            rows.append({
                "id": f"OPM fact sheet: {slug}",
                "kind": f"{kind}-fact-sheet",
                "title": html_to_text(title.group(1)) if title else slug,
                "text": text,
                "url": url,
                "retrieved": time.strftime("%Y-%m-%d"),
            })
            time.sleep(0.3)
    with (OUT / "factsheets.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"fact sheets: {len(rows)} -> data/raw/opm/factsheets.jsonl")
    fetch_pdfs(sorted(pdf_links), "fact-sheet PDFs")


# ---------- handbooks (PDF) ----------

def fetch_pdfs(urls: list[str], label: str) -> None:
    pdf_dir = OUT / "handbooks"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    have_pdftotext = shutil.which("pdftotext") is not None
    rows = []
    for url in urls:
        name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        if name in SEEN_PDFS:  # same PDF linked from several index pages
            continue
        SEEN_PDFS.add(name)
        dest = pdf_dir / name
        try:
            dest.write_bytes(get(url))
        except urllib.error.HTTPError as exc:
            print(f"  skip {name}: HTTP {exc.code}")
            continue
        if have_pdftotext:
            text = subprocess.run(["pdftotext", "-layout", str(dest), "-"], capture_output=True, text=True).stdout
            rows.append({"id": f"OPM handbook: {dest.stem}", "kind": "handbook", "title": dest.stem,
                         "text": text.strip(), "url": url, "retrieved": time.strftime("%Y-%m-%d")})
    if rows:
        with (OUT / "handbooks.jsonl").open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    note = "" if have_pdftotext else " (install poppler-utils for text extraction)"
    print(f"{label}: {len(urls)} PDFs -> data/raw/opm/handbooks/{note}")


def fetch_handbooks() -> None:
    index = get(f"{PAY_LEAVE}/reference-materials/handbooks/").decode("utf-8", "replace")
    urls = sorted({urllib.parse.urljoin(BASE, h) for h in re.findall(r'href="([^"]+\.pdf)"', index, re.I)})
    fetch_pdfs(urls, "handbooks")


# ---------- GS pay table (XLSX -> CSV, stdlib) ----------

def xlsx_rows(data: bytes):
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    zf = zipfile.ZipFile(io.BytesIO(data))
    shared = []
    if "xl/sharedStrings.xml" in zf.namelist():
        for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).findall("m:si", ns):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
    sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    for row in sheet.iter(f"{{{ns['m']}}}row"):
        vals = []
        for c in row.findall("m:c", ns):
            v = c.find("m:v", ns)
            if v is None:
                inline = c.find("m:is/m:t", ns)
                vals.append(inline.text if inline is not None else "")
            elif c.get("t") == "s":
                vals.append(shared[int(v.text)])
            else:
                vals.append(v.text)
        yield vals


def fetch_pay(year: int) -> None:
    for y in (year, year - 1):
        url = f"{PAY_LEAVE}/salaries-wages/{y}/{y}-general-schedule-pay-rates.xlsx"
        try:
            data = get(url)
        except urllib.error.HTTPError:
            continue
        (OUT / f"gs_pay_{y}.xlsx").write_bytes(data)
        rows = list(xlsx_rows(data))
        with (OUT / f"gs_pay_{y}.csv").open("w", newline="") as fh:
            csv.writer(fh).writerows(rows)
        print(f"GS pay {y}: {len(rows)} rows (incl. header) -> data/raw/opm/gs_pay_{y}.csv  [{url}]")
        return
    print(f"GS pay: no XLSX found for {year} or {year - 1}")


# ---------- federal holidays (iCalendar -> CSV) ----------

def fetch_holidays() -> None:
    ics = get(f"{PAY_LEAVE}/federal-holidays/holidays.ics").decode("utf-8", "replace")
    ics = re.sub(r"\r?\n[ \t]", "", ics)  # unfold continuation lines
    events = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", ics, re.S):
        date = re.search(r"DTSTART[^:]*:(\d{8})", block)
        name = re.search(r"SUMMARY[^:]*:(.*)", block)
        if date and name:
            d = date.group(1)
            summary = re.sub(r"\\([,;\\])", r"\1", name.group(1).strip())  # iCalendar escapes
            events.append((f"{d[:4]}-{d[4:6]}-{d[6:]}", summary))
    events.sort()
    with (OUT / "holidays.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "holiday"])
        w.writerows(events)
    years = sorted({e[0][:4] for e in events})
    print(f"holidays: {len(events)} ({years[0]}–{years[-1]}) -> data/raw/opm/holidays.csv" if events else "holidays: none parsed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["factsheets", "handbooks", "pay", "holidays"])
    ap.add_argument("--pay-year", type=int, default=int(time.strftime("%Y")))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "handbooks", "factsheets"):
        (OUT / "handbooks.jsonl").unlink(missing_ok=True)
    steps = {"factsheets": fetch_factsheets, "handbooks": fetch_handbooks,
             "pay": lambda: fetch_pay(args.pay_year), "holidays": fetch_holidays}
    for name, fn in steps.items():
        if args.only in (None, name):
            fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
