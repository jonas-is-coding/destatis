#!/usr/bin/env python3
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
from urllib.parse import urljoin, urlparse

import requests
from huggingface_hub import HfApi

DEST_DOMAIN = "www.destatis.de"
ORG = "statistisches-bundesamt"
REPO_NAME = os.getenv("HF_REPO_NAME", "destatis-open-data-ml-ready")
REPO_NAMESPACE = os.getenv("HF_REPO_NAMESPACE", ORG)
REPO_ID = f"{REPO_NAMESPACE}/{REPO_NAME}"
LICENSE_TEXT = "Datenlizenz Deutschland – Namensnennung – Version 2.0"

SEED_URLS = [
    "https://www.destatis.de/EN/Service/OpenData/short-term-indicators.html",
    "https://www.destatis.de/DE/Service/OpenData/konjunkturindikatoren.html",
    "https://www.destatis.de/EN/Service/OpenData/_node.html",
    "https://www.destatis.de/DE/Service/OpenData/_node.html",
]

MAX_PAGES = 180
MAX_CSV_FILES = 800
REQUEST_TIMEOUT = 45
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
MISSING_TOKENS = {"...", ".", "x", "-", "–", ""}


@dataclass
class Record:
    source_url: str
    raw_path: str
    normalized_path: str
    rows: int
    columns: int
    numeric_columns_ratio: float
    missing_ratio: float
    inconsistent_rows: int
    ml_ready: str
    notes: str
    sha256_raw: str


def is_internal(url: str) -> bool:
    return urlparse(url).netloc in {"", DEST_DOMAIN}


def normalize_url(base: str, href: str) -> str:
    return urljoin(base, href.split("#", 1)[0])


def looks_like_csv(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".csv")


def looks_like_html(url: str) -> bool:
    p = urlparse(url).path.lower()
    return p.endswith(".html") or p.endswith("_node.html") or p.endswith("_inhalt.html")


def extract_links(html: str, base_url: str) -> tuple[set[str], set[str]]:
    csv_links: set[str] = set()
    html_links: set[str] = set()
    for href in HREF_RE.findall(html):
        abs_url = normalize_url(base_url, href)
        if not is_internal(abs_url):
            continue
        if looks_like_csv(abs_url):
            csv_links.add(abs_url)
        elif looks_like_html(abs_url):
            html_links.add(abs_url)
    return csv_links, html_links


def crawl_csv_links(session: requests.Session) -> list[str]:
    visited: set[str] = set()
    queue = deque(SEED_URLS)
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

        found_csv, found_html = extract_links(r.text, url)
        csv_links.update(found_csv)
        for h in found_html:
            if h not in visited and len(queue) < MAX_PAGES * 2:
                queue.append(h)

    return sorted(csv_links)[:MAX_CSV_FILES]


def infer_delimiter(sample: str) -> str:
    cands = [";", ",", "\t", "|"]
    counts = {c: sample.count(c) for c in cands}
    return max(counts, key=counts.get) if any(counts.values()) else ";"


def sanitize_header(cells: list[str], n_cols: int) -> list[str]:
    out = []
    used: set[str] = set()
    for i in range(n_cols):
        raw = cells[i] if i < len(cells) else f"col_{i+1}"
        c = re.sub(r"\s+", "_", raw.strip().lower())
        c = re.sub(r"[^a-z0-9_]+", "", c)
        if not c:
            c = f"col_{i+1}"
        base = c
        k = 2
        while c in used:
            c = f"{base}_{k}"
            k += 1
        used.add(c)
        out.append(c)
    return out


def looks_numeric(v: str) -> bool:
    if not v or v in MISSING_TOKENS:
        return False
    t = v.strip().replace(" ", "")
    t = t.replace(".", "").replace(",", ".")
    return bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", t))


def normalize_numeric(v: str) -> str:
    t = v.strip()
    if t in MISSING_TOKENS:
        return ""
    # German thousands/decimal style to dotted decimal.
    t = t.replace(" ", "")
    if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?", t):
        t = t.replace(".", "").replace(",", ".")
        return t
    if re.fullmatch(r"[-+]?\d+,\d+", t):
        return t.replace(",", ".")
    return t


def safe_path_from_url(url: str) -> str:
    p = urlparse(url).path.strip("/")
    s = re.sub(r"[^A-Za-z0-9._/-]+", "_", p)
    if not s.endswith(".csv"):
        s += ".csv"
    return s


def parse_and_normalize(text: str, delimiter: str) -> tuple[list[str], list[list[str]], dict]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return [], [], {"notes": "too few lines", "ml_ready": "no"}

    rows = list(csv.reader(io.StringIO("\n".join(lines[:6000])), delimiter=delimiter))
    if not rows:
        return [], [], {"notes": "empty", "ml_ready": "no"}

    lens = [len(r) for r in rows]
    common_cols = Counter(lens).most_common(1)[0][0]
    inconsistent = sum(1 for n in lens if n != common_cols)

    start_idx = 0
    if len(rows) > 1 and len(rows[0]) <= 1 and len(rows[1]) >= 2:
        start_idx = 1

    header_idx = start_idx
    if header_idx >= len(rows):
        return [], [], {"notes": "no header", "ml_ready": "no"}

    header = sanitize_header(rows[header_idx], common_cols)
    body = rows[header_idx + 1 :]

    normalized_rows: list[list[str]] = []
    missing_count = 0
    total_cells = 0

    for row in body:
        fixed = (row + [""] * common_cols)[:common_cols]
        norm = []
        for c in fixed:
            c2 = c.strip()
            if c2 in MISSING_TOKENS:
                c2 = ""
            c2 = normalize_numeric(c2)
            if c2 == "":
                missing_count += 1
            total_cells += 1
            norm.append(c2)
        normalized_rows.append(norm)

    if not normalized_rows:
        return [], [], {"notes": "no data rows", "ml_ready": "no"}

    col_numeric = 0
    for ci in range(common_cols):
        vals = [r[ci] for r in normalized_rows if r[ci] != ""]
        if not vals:
            continue
        num_hits = sum(1 for v in vals if looks_numeric(v))
        if num_hits / max(1, len(vals)) >= 0.7:
            col_numeric += 1

    numeric_ratio = col_numeric / max(1, common_cols)
    missing_ratio = missing_count / max(1, total_cells)

    score = 100
    notes = []
    if inconsistent > max(2, int(len(rows) * 0.03)):
        score -= 30
        notes.append("many inconsistent rows")
    if numeric_ratio < 0.1:
        score -= 20
        notes.append("very low numeric signal")
    if missing_ratio > 0.4:
        score -= 25
        notes.append("high missing ratio")
    if common_cols > 120:
        score -= 15
        notes.append("very wide table")
    if len(normalized_rows) < 50:
        score -= 10
        notes.append("small table")

    ml_ready = "yes" if score >= 60 else "no"
    if not notes:
        notes.append("good baseline for tabular ML after split/feature engineering")

    meta = {
        "rows": len(normalized_rows),
        "columns": common_cols,
        "numeric_ratio": numeric_ratio,
        "missing_ratio": missing_ratio,
        "inconsistent_rows": inconsistent,
        "ml_ready": ml_ready,
        "notes": "; ".join(notes),
    }
    return header, normalized_rows, meta


def build_readme(total: int, kept: int, dropped: int) -> str:
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
- ml-ready
---

# Destatis Open Data ML-Ready (Curated)

## Inoffizieller Hinweis / Unofficial note
Dieses Repository ist ein **privates Open-Source-Projekt** und **nicht offiziell** vom Statistischen Bundesamt (Destatis) betrieben.

This repository is a **private open-source project** and is **not an official** repository of the Federal Statistical Office of Germany (Destatis).

## Quelle / Source
- Primäre Quelle: https://www.destatis.de/
- Lizenz der Quelldaten: {LICENSE_TEXT}

## Auswahlprinzip
- CSV gecrawlt und normalisiert (Header, Delimiter, Missing Values, Dezimalformat)
- Nur Datensätze mit Mindestqualität für ML-Baseline enthalten
- Details in `metadata/ml_ready_report.csv`

## Snapshot
- Gefundene CSVs: {total}
- Als ML-ready aufgenommen: {kept}
- Verworfen: {dropped}
"""


def main() -> int:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN fehlt")

    api = HfApi(token=token)
    sess = requests.Session()

    out = Path("/tmp/destatis_ml_ready")
    raw_dir = out / "raw"
    norm_dir = out / "normalized"
    meta_dir = out / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    csv_links = crawl_csv_links(sess)
    print(f"Found {len(csv_links)} CSV links")

    records: list[Record] = []
    kept = 0
    for i, url in enumerate(csv_links, 1):
        try:
            r = sess.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            raw = r.content
            text = raw.decode("utf-8", errors="replace")
        except requests.RequestException:
            continue

        rel = safe_path_from_url(url)
        raw_path = raw_dir / rel
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)

        delim = infer_delimiter(text[:20000])
        header, rows, meta = parse_and_normalize(text, delim)

        norm_rel = rel
        norm_path = norm_dir / norm_rel
        norm_path.parent.mkdir(parents=True, exist_ok=True)

        if meta.get("ml_ready") == "yes" and header and rows:
            with norm_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
            kept += 1
            ml_ready = "yes"
        else:
            ml_ready = "no"
            # don't keep failed normalized outputs
            if norm_path.exists():
                norm_path.unlink()

        rec = Record(
            source_url=url,
            raw_path=f"raw/{rel}",
            normalized_path=f"normalized/{norm_rel}" if ml_ready == "yes" else "",
            rows=int(meta.get("rows", 0)),
            columns=int(meta.get("columns", 0)),
            numeric_columns_ratio=float(meta.get("numeric_ratio", 0.0)),
            missing_ratio=float(meta.get("missing_ratio", 1.0)),
            inconsistent_rows=int(meta.get("inconsistent_rows", 0)),
            ml_ready=ml_ready,
            notes=str(meta.get("notes", "")),
            sha256_raw=hashlib.sha256(raw).hexdigest(),
        )
        records.append(rec)

        if i % 40 == 0:
            print(f"Processed {i}/{len(csv_links)}")

    if not records:
        raise SystemExit("No records processed")

    with (meta_dir / "files.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")

    with (meta_dir / "ml_ready_report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].__dict__.keys()))
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.__dict__)

    dropped = len(records) - kept
    (out / "README.md").write_text(build_readme(len(records), kept, dropped), encoding="utf-8")

    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=REPO_ID, repo_type="dataset")

    print(f"Uploaded curated dataset: https://huggingface.co/datasets/{REPO_ID}")
    print(f"ML-ready kept: {kept} / {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
