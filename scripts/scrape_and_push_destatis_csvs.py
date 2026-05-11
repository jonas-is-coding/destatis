#!/usr/bin/env python3
"""Crawl Destatis Open Data pages for CSV files, download them, analyze ML-readiness,
and publish everything to a Hugging Face dataset repository.

Usage:
  HF_TOKEN=... python3 scripts/scrape_and_push_destatis_csvs.py

Default target repo:
  statistisches-bundesamt/destatis-open-data-csv-bulk
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from huggingface_hub import HfApi

DEST_DOMAIN = "www.destatis.de"
ORG = "statistisches-bundesamt"
REPO_NAME = "destatis-open-data-csv-bulk"
REPO_ID = f"{ORG}/{REPO_NAME}"
LICENSE_TEXT = "Datenlizenz Deutschland – Namensnennung – Version 2.0"

SEED_URLS = [
    "https://www.destatis.de/EN/Service/OpenData/short-term-indicators.html",
    "https://www.destatis.de/DE/Service/OpenData/konjunkturindikatoren.html",
    "https://www.destatis.de/EN/Service/OpenData/_node.html",
    "https://www.destatis.de/DE/Service/OpenData/_node.html",
]

MAX_PAGES = 120
MAX_CSV_FILES = 400
REQUEST_TIMEOUT = 45

HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


@dataclass
class CsvRecord:
    source_url: str
    relative_path: str
    bytes_size: int
    sha256: str
    delimiter: str
    columns_median: int
    row_count: int
    title_row_detected: bool
    decimal_comma_detected: bool
    missing_symbol_count: int
    inconsistent_column_rows: int
    ml_readiness: str
    notes: str


def is_internal(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc in {"", DEST_DOMAIN}


def normalize_url(base: str, href: str) -> str:
    return urljoin(base, href.split("#", 1)[0])


def looks_like_csv_url(url: str) -> bool:
    p = urlparse(url)
    return p.path.lower().endswith(".csv")


def looks_like_html_url(url: str) -> bool:
    p = urlparse(url)
    path = p.path.lower()
    return path.endswith(".html") or path.endswith("_node.html") or path.endswith("_inhalt.html")


def extract_links(html: str, base_url: str) -> tuple[set[str], set[str]]:
    csv_links: set[str] = set()
    html_links: set[str] = set()
    for href in HREF_RE.findall(html):
        abs_url = normalize_url(base_url, href)
        if not is_internal(abs_url):
            continue
        if looks_like_csv_url(abs_url):
            csv_links.add(abs_url)
        elif looks_like_html_url(abs_url):
            html_links.add(abs_url)
    return csv_links, html_links


def crawl_for_csv_links(session: requests.Session) -> list[str]:
    visited: set[str] = set()
    queue: deque[str] = deque(SEED_URLS)
    csv_links: set[str] = set()

    while queue and len(visited) < MAX_PAGES and len(csv_links) < MAX_CSV_FILES:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code >= 400:
                continue
        except requests.RequestException:
            continue

        page_csv, page_html = extract_links(r.text, url)

        for link in page_csv:
            if len(csv_links) >= MAX_CSV_FILES:
                break
            csv_links.add(link)

        for link in page_html:
            if link not in visited and len(visited) + len(queue) < MAX_PAGES * 2:
                queue.append(link)

    return sorted(csv_links)


def infer_delimiter(sample_text: str) -> str:
    # Destatis CSVs are often semicolon-delimited.
    candidates = [";", ",", "\t", "|"]
    counts = {c: sample_text.count(c) for c in candidates}
    return max(counts, key=counts.get) if any(counts.values()) else ";"


def analyze_csv_content(text: str, delimiter: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {
            "columns_median": 0,
            "row_count": 0,
            "title_row_detected": False,
            "decimal_comma_detected": False,
            "missing_symbol_count": 0,
            "inconsistent_column_rows": 0,
            "ml_readiness": "low",
            "notes": "empty file",
        }

    row_lengths: list[int] = []
    missing_tokens = {"...", "-", "–", ".", "x"}
    missing_symbol_count = 0
    decimal_comma_detected = False

    reader = csv.reader(io.StringIO("\n".join(lines[:3000])), delimiter=delimiter)
    parsed_rows = list(reader)
    for row in parsed_rows:
        row_lengths.append(len(row))
        for cell in row:
            cell_s = cell.strip()
            if cell_s in missing_tokens:
                missing_symbol_count += 1
            if re.search(r"\d,\d", cell_s):
                decimal_comma_detected = True

    if not row_lengths:
        return {
            "columns_median": 0,
            "row_count": len(lines),
            "title_row_detected": False,
            "decimal_comma_detected": decimal_comma_detected,
            "missing_symbol_count": missing_symbol_count,
            "inconsistent_column_rows": 0,
            "ml_readiness": "low",
            "notes": "unparseable rows",
        }

    row_count = len(parsed_rows)
    counter = Counter(row_lengths)
    columns_median = counter.most_common(1)[0][0]
    inconsistent_column_rows = sum(1 for n in row_lengths if n != columns_median)

    title_row_detected = False
    if len(parsed_rows) >= 2:
        first_len = len(parsed_rows[0])
        second_len = len(parsed_rows[1])
        if first_len <= 1 and second_len >= 2:
            title_row_detected = True

    notes: list[str] = []
    score = 100
    if title_row_detected:
        score -= 20
        notes.append("title row before header")
    if decimal_comma_detected:
        score -= 20
        notes.append("decimal comma normalization needed")
    if missing_symbol_count > 0:
        score -= 20
        notes.append("special missing value tokens present")
    if inconsistent_column_rows > max(2, int(row_count * 0.02)):
        score -= 25
        notes.append("inconsistent column counts")
    if columns_median > 80:
        score -= 10
        notes.append("very wide table")
    if row_count < 40:
        score -= 10
        notes.append("small sample size")

    if score >= 75:
        readiness = "high"
    elif score >= 50:
        readiness = "medium"
    else:
        readiness = "low"

    return {
        "columns_median": columns_median,
        "row_count": row_count,
        "title_row_detected": title_row_detected,
        "decimal_comma_detected": decimal_comma_detected,
        "missing_symbol_count": missing_symbol_count,
        "inconsistent_column_rows": inconsistent_column_rows,
        "ml_readiness": readiness,
        "notes": "; ".join(notes) if notes else "clean enough for baseline ML preprocessing",
    }


def safe_relpath_from_url(url: str) -> str:
    parsed = urlparse(url)
    raw = parsed.path.strip("/")
    safe = re.sub(r"[^A-Za-z0-9._/-]+", "_", raw)
    if not safe.lower().endswith(".csv"):
        safe += ".csv"
    return safe


def build_readme(total: int, high: int, medium: int, low: int) -> str:
    return f"""---
license: other
language:
- de
- en
tags:
- destatis
- germany
- open-data
- tabular
- time-series
---

# Destatis Open Data CSV Bulk

## Inoffizieller Hinweis / Unofficial note
Dieses Repository ist ein **privates Open-Source-Projekt** und **nicht offiziell** vom Statistischen Bundesamt (Destatis) betrieben.

This repository is a **private open-source project** and is **not an official** repository of the Federal Statistical Office of Germany (Destatis).

## Inhalt
- Automatisch gecrawlte CSV-Dateien aus dem Destatis-Open-Data-Umfeld
- Metadaten je Datei in `metadata/files.jsonl`
- ML-Einschätzung je Datei in `metadata/ml_readiness_report.csv`

## Quelle / Source
- Primäre Quelle: https://www.destatis.de/
- Lizenz der Quelldaten: {LICENSE_TEXT}

## Aktueller Snapshot
- Gesamtdateien: {total}
- ML-Readiness `high`: {high}
- ML-Readiness `medium`: {medium}
- ML-Readiness `low`: {low}

## Warum manche Datensätze nicht direkt ML-ready sind
- erste Zeile ist oft ein Titel statt Header
- Semikolon statt Komma als Trennzeichen
- Dezimal-Komma statt Dezimalpunkt
- Sonderzeichen für Missing Values (`...`, `.`, `x`, `-`, `–`)
- uneinheitliche Zeilenbreite und sehr breite Pivot-Tabellen
"""


def main() -> int:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN fehlt. Bitte als Umgebungsvariable setzen.")

    api = HfApi(token=token)
    session = requests.Session()

    out_root = Path("/tmp/destatis_bulk_repo")
    data_root = out_root / "data"
    meta_root = out_root / "metadata"
    data_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    csv_links = crawl_for_csv_links(session)
    if not csv_links:
        raise SystemExit("Keine CSV-Links gefunden.")

    records: list[CsvRecord] = []
    for idx, url in enumerate(csv_links, start=1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            raw = r.content
            text = raw.decode("utf-8", errors="replace")
        except requests.RequestException:
            continue

        rel_path = safe_relpath_from_url(url)
        file_path = data_root / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(raw)

        delimiter = infer_delimiter(text[:20000])
        analysis = analyze_csv_content(text, delimiter)

        records.append(
            CsvRecord(
                source_url=url,
                relative_path=f"data/{rel_path}",
                bytes_size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                delimiter=delimiter,
                columns_median=analysis["columns_median"],
                row_count=analysis["row_count"],
                title_row_detected=analysis["title_row_detected"],
                decimal_comma_detected=analysis["decimal_comma_detected"],
                missing_symbol_count=analysis["missing_symbol_count"],
                inconsistent_column_rows=analysis["inconsistent_column_rows"],
                ml_readiness=analysis["ml_readiness"],
                notes=analysis["notes"],
            )
        )

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(csv_links)} links ...")

    if not records:
        raise SystemExit("CSV-Download fehlgeschlagen: keine Dateien gespeichert.")

    jsonl_path = meta_root / "files.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

    report_path = meta_root / "ml_readiness_report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].__dict__.keys()))
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.__dict__)

    counts = Counter(r.ml_readiness for r in records)
    readme = build_readme(
        total=len(records),
        high=counts.get("high", 0),
        medium=counts.get("medium", 0),
        low=counts.get("low", 0),
    )
    (out_root / "README.md").write_text(readme, encoding="utf-8")

    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(out_root), repo_id=REPO_ID, repo_type="dataset")

    print(f"Uploaded {len(records)} CSV files to https://huggingface.co/datasets/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
