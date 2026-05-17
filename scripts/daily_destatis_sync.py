#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import hashlib
import io
import json
import os
import re
import unicodedata
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from huggingface_hub import HfApi

DEST_DOMAIN = "www.destatis.de"
SEED_URLS = [
    "https://www.destatis.de/EN/Service/OpenData/short-term-indicators.html",
    "https://www.destatis.de/DE/Service/OpenData/konjunkturindikatoren.html",
    "https://www.destatis.de/EN/Service/OpenData/_node.html",
    "https://www.destatis.de/DE/Service/OpenData/_node.html",
]

MAX_PAGES = int(os.getenv("MAX_PAGES", "200"))
MAX_CSV_FILES = int(os.getenv("MAX_CSV_FILES", "1000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "45"))
HF_TOKEN = os.getenv("HF_TOKEN")
HF_NAMESPACE = os.getenv("HF_NAMESPACE", "destatis")
HF_REQUIRED_NAMESPACE = os.getenv("HF_REQUIRED_NAMESPACE", "destatis")
HF_PUBLISH_MODE = os.getenv("HF_PUBLISH_MODE", "multi")  # multi|single
HF_REPO_ID = os.getenv("HF_REPO_ID", f"{HF_NAMESPACE}/destatis-open-data-ml-ready")
HF_DATASET_PREFIX = os.getenv("HF_DATASET_PREFIX", "destatis-ml-")
HF_README_ONLY_BACKFILL = os.getenv("HF_README_ONLY_BACKFILL", "false").lower() == "true"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ML_DIR = ROOT / "data" / "ml_ready"
META_DIR = ROOT / "metadata"
MANIFEST_PATH = META_DIR / "manifest.json"
REPORT_PATH = META_DIR / "latest_sync_report.csv"
RUNLOG_PATH = META_DIR / "runs.jsonl"

HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
ANCHOR_RE = re.compile(r'<a([^>]+)href=["\']([^"\']+\.csv)["\']([^>]*)>(.*?)</a>', re.IGNORECASE | re.DOTALL)
TITLE_RE = re.compile(r'title=["\']([^"\']+)["\']', re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
HEADING_RE = re.compile(r"<h[1-4][^>]*>(.*?)</h[1-4]>", re.IGNORECASE | re.DOTALL)
MISSING_TOKENS = {"...", ".", "x", "-", "–", ""}


@dataclass
class FileRecord:
    source_url: str
    rel_path: str
    sha256: str
    bytes_size: int
    rows: int
    columns: int
    numeric_ratio: float
    missing_ratio: float
    inconsistent_rows: int
    ml_ready: bool
    note: str
    hf_repo_id: str
    last_seen_utc: str


@dataclass
class CsvDoc:
    label: str
    context: str
    page_url: str


def safe_path_from_url(url: str) -> str:
    p = urlparse(url).path.strip("/")
    s = re.sub(r"[^A-Za-z0-9._/-]+", "_", p)
    if not s.endswith(".csv"):
        s += ".csv"
    return s


def repo_slug_from_relpath(rel_path: str) -> str:
    # Keep dataset repo names concise by using only the source file stem.
    # Example:
    # static/de_/opendata/data/private_konsumausgaben_preisbereinigt_x13.csv
    # -> private-konsumausgaben-preisbereinigt-x13
    base = Path(rel_path).stem.lower()
    base = re.sub(r"[^a-z0-9-]+", "-", base)
    base = re.sub(r"-+", "-", base).strip("-")
    if len(base) > 70:
        base = base[:70].rstrip("-")
    if HF_DATASET_PREFIX:
        return f"{HF_DATASET_PREFIX}{base}"
    return base


def legacy_repo_slug_from_relpath(rel_path: str) -> str:
    # Historical naming scheme used earlier in this project.
    # Example:
    # static/de_/opendata/data/private_konsumausgaben_preisbereinigt_x13.csv
    # -> destatis-ml-static-de-opendata-data-private-konsumausgaben-preisbereinigt-x13
    base = rel_path.lower().replace("/", "-")
    base = re.sub(r"\.csv$", "", base)
    base = re.sub(r"[^a-z0-9-]+", "-", base)
    base = re.sub(r"-+", "-", base).strip("-")
    return f"destatis-ml-{base}"


def is_internal(url: str) -> bool:
    return urlparse(url).netloc in {"", DEST_DOMAIN}


def normalize_url(base: str, href: str) -> str:
    return urljoin(base, href.split("#", 1)[0])


def looks_like_csv(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".csv")


def looks_like_html(url: str) -> bool:
    p = urlparse(url).path.lower()
    return p.endswith(".html") or p.endswith("_node.html") or p.endswith("_inhalt.html")


def clean_text(s: str) -> str:
    s = html.unescape(s)
    s = TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def ascii_normalize(s: str) -> str:
    s = html.unescape(s)
    repl = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
        "ß": "ss",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def extract_csv_docs(html: str, page_url: str) -> dict[str, CsvDoc]:
    docs: dict[str, CsvDoc] = {}
    matches = list(ANCHOR_RE.finditer(html))
    for i, m in enumerate(matches):
        attrs = f"{m.group(1)} {m.group(3)}"
        href, raw_label = m.group(2), m.group(4)
        csv_url = normalize_url(page_url, href)
        label = clean_text(raw_label)
        title_match = TITLE_RE.search(attrs)
        title_text = clean_text(title_match.group(1)) if title_match else ""
        # Prefer list-item context (clean prose), then nearest heading.
        context = ""
        li_start = html.rfind("<li", 0, m.start())
        li_end = html.find("</li>", m.end())
        if li_start != -1 and li_end != -1 and li_end > li_start:
            context = clean_text(html[li_start : li_end + 5])
        if not context:
            heading = ""
            for hm in HEADING_RE.finditer(html[: m.start()]):
                heading = clean_text(hm.group(1))
            if heading:
                context = f"Section heading: {heading}"
        if len(context) > 360:
            context = context[:360].rstrip() + "..."
        # Prefer official link title when available; it is usually more descriptive than short labels like "BV 4.1".
        official_label = title_text or label
        official_label = re.sub(r"^(External link\s*)?(CSV[- ]Datei:\s*)", "", official_label, flags=re.IGNORECASE).strip()
        docs[csv_url] = CsvDoc(
            label=official_label or Path(urlparse(csv_url).path).stem.replace("_", " "),
            context=context,
            page_url=page_url,
        )
    return docs


def crawl_csv_links(session: requests.Session) -> tuple[list[str], dict[str, CsvDoc]]:
    visited: set[str] = set()
    queue: deque[str] = deque(SEED_URLS)
    csv_links: set[str] = set()
    csv_docs: dict[str, CsvDoc] = {}

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
        csv_docs.update(extract_csv_docs(r.text, url))

        for href in HREF_RE.findall(r.text):
            abs_url = normalize_url(url, href)
            if not is_internal(abs_url):
                continue
            if looks_like_csv(abs_url):
                csv_links.add(abs_url)
            elif looks_like_html(abs_url) and abs_url not in visited:
                queue.append(abs_url)

    return sorted(csv_links), csv_docs


def infer_delimiter(sample: str) -> str:
    cands = [";", ",", "\t", "|"]
    counts = {c: sample.count(c) for c in cands}
    return max(counts, key=counts.get) if any(counts.values()) else ";"


def sanitize_header(cells: list[str], n_cols: int) -> list[str]:
    out: list[str] = []
    used: set[str] = set()
    for i in range(n_cols):
        raw = cells[i] if i < len(cells) else f"col_{i+1}"
        c = re.sub(r"\s+", "_", ascii_normalize(raw.strip().lower()))
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
    if not v:
        return False
    t = v.strip().replace(" ", "")
    t = t.replace(".", "").replace(",", ".")
    return bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", t))


def normalize_cell(v: str) -> str:
    t = v.strip()
    if t in MISSING_TOKENS:
        return ""
    t = t.replace(" ", "")
    if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+(,\d+)?", t):
        return t.replace(".", "").replace(",", ".")
    if re.fullmatch(r"[-+]?\d+,\d+", t):
        return t.replace(",", ".")
    return t


def reshape_to_long(header: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]], dict]:
    date_candidates = [
        i
        for i, col in enumerate(header)
        if any(x in col.lower() for x in ["date", "datum", "monat", "jahr", "zeit", "period"])
    ]
    if not date_candidates:
        return header, rows, {"reshaped": False, "reason": "no_date_column"}

    date_idx = date_candidates[0]
    numeric_indices: list[int] = []
    for i, _col in enumerate(header):
        if i == date_idx:
            continue
        values = [r[i] for r in rows if i < len(r) and r[i] != ""]
        if not values:
            continue
        numeric_share = sum(1 for v in values if looks_numeric(v)) / len(values)
        if numeric_share >= 0.7:
            numeric_indices.append(i)

    if not numeric_indices:
        return header, rows, {"reshaped": False, "reason": "no_numeric_value_columns"}

    long_header = ["date", "indicator", "value"]
    long_rows: list[list[str]] = []
    for row in rows:
        date_value = row[date_idx] if date_idx < len(row) else ""
        if date_value == "":
            continue
        for i in numeric_indices:
            if i >= len(row):
                continue
            value = row[i]
            if value == "":
                continue
            long_rows.append([date_value, header[i], value])

    if not long_rows:
        return header, rows, {"reshaped": False, "reason": "empty_after_reshape"}

    return long_header, long_rows, {
        "reshaped": True,
        "reason": "wide_to_long",
        "source_columns": len(header),
        "numeric_columns": len(numeric_indices),
        "long_rows": len(long_rows),
    }


def parse_ml_ready(raw_text: str) -> tuple[list[str], list[list[str]], dict]:
    lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return [], [], {"ml_ready": False, "note": "too few lines"}

    delimiter = infer_delimiter("\n".join(lines[:200]))
    rows = list(csv.reader(io.StringIO("\n".join(lines[:8000])), delimiter=delimiter))
    if not rows:
        return [], [], {"ml_ready": False, "note": "unparseable"}

    lens = [len(r) for r in rows]
    common_cols = Counter(lens).most_common(1)[0][0]
    inconsistent = sum(1 for n in lens if n != common_cols)

    start_idx = 1 if (len(rows) > 1 and len(rows[0]) <= 1 and len(rows[1]) >= 2) else 0
    if start_idx >= len(rows):
        return [], [], {"ml_ready": False, "note": "missing header"}

    header = sanitize_header(rows[start_idx], common_cols)
    body = rows[start_idx + 1 :]
    if not body:
        return [], [], {"ml_ready": False, "note": "no data rows"}

    normalized: list[list[str]] = []
    missing = 0
    total = 0
    for row in body:
        fixed = (row + [""] * common_cols)[:common_cols]
        nr = []
        for c in fixed:
            x = normalize_cell(c)
            if x == "":
                missing += 1
            total += 1
            nr.append(x)
        normalized.append(nr)

    numeric_cols = 0
    for i in range(common_cols):
        vals = [r[i] for r in normalized if r[i] != ""]
        if not vals:
            continue
        if (sum(1 for v in vals if looks_numeric(v)) / len(vals)) >= 0.7:
            numeric_cols += 1

    numeric_ratio = numeric_cols / max(1, common_cols)
    missing_ratio = missing / max(1, total)

    score = 100
    notes = []
    if inconsistent > max(2, int(len(rows) * 0.03)):
        score -= 30
        notes.append("inconsistent rows")
    if numeric_ratio < 0.08:
        score -= 20
        notes.append("low numeric signal")
    if missing_ratio > 0.45:
        score -= 25
        notes.append("high missing ratio")
    if common_cols > 150:
        score -= 15
        notes.append("very wide")
    if len(normalized) < 50:
        score -= 10
        notes.append("small")

    long_header, long_rows, reshape_meta = reshape_to_long(header, normalized)
    if reshape_meta["reshaped"]:
        header = long_header
        normalized = long_rows

    if not reshape_meta["reshaped"]:
        reshape_reason = reshape_meta.get("reason")
        if reshape_reason:
            notes.append(reshape_reason)

    ml_ready = score >= 60
    note = "; ".join(notes) if notes else "ok"
    return header, normalized, {
        "rows": len(normalized),
        "columns": common_cols,
        "numeric_ratio": numeric_ratio,
        "missing_ratio": missing_ratio,
        "inconsistent_rows": inconsistent,
        "ml_ready": ml_ready,
        "format": "long" if reshape_meta["reshaped"] else "original_normalized",
        "long_format": reshape_meta["reshaped"],
        "reshape_note": reshape_meta["reason"],
        "quality_score": score,
        "note": note,
    }


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"files": {}, "updated_at": None}


def write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(records: list[FileRecord]) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with REPORT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def write_runlog(payload: dict) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    with RUNLOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def upload_single_repo(api: HfApi) -> None:
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(ROOT / "data"), repo_id=HF_REPO_ID, repo_type="dataset", path_in_repo="data")
    api.upload_folder(folder_path=str(ROOT / "metadata"), repo_id=HF_REPO_ID, repo_type="dataset", path_in_repo="metadata")


def dataset_readme(
    title: str,
    source_url: str,
    doc: CsvDoc | None,
    rec: FileRecord,
    header: list[str],
    now_iso: str,
) -> str:
    label = doc.label if doc else title
    context = doc.context if doc else ""
    page_url = doc.page_url if doc else ""
    if len(label) < 8 or re.fullmatch(r"(bv\s*4\.?1|x13|originalwert)", label.lower()):
        label = Path(rec.rel_path).stem.replace("_", " ").replace("-", " ").title()
    column_lines = "\n".join([f"- `{c}`" for c in header[:12]]) if header else "- `value`"
    if len(header) > 12:
        column_lines += "\n- `...`"
    field_notes = []
    for c in header[:12]:
        lc = c.lower()
        if "datum" in lc or "date" in lc:
            field_notes.append(f"- `{c}`: time index / reporting period.")
        elif any(x in lc for x in ["veraenderung", "veranderung", "vernderung", "gegenueber", "gegenuber", "change"]):
            field_notes.append(f"- `{c}`: period-over-period or year-over-year change metric.")
        elif "trend" in lc or "x13" in lc or "bv41" in lc:
            field_notes.append(f"- `{c}`: seasonally/calendar adjusted trend component.")
        elif "original" in lc:
            field_notes.append(f"- `{c}`: raw/original value from source table.")
    field_notes_block = "\n".join(field_notes[:6]) if field_notes else "- Column semantics follow Destatis source naming."
    quality_note = rec.note if rec.note else "Passed baseline quality checks."
    context_clean = context.strip()
    if (not context_clean) or (len(context_clean) < 40) or ("|" in context_clean and len(context_clean) < 120):
        context_clean = (
            f"Official description from source metadata: {label}. "
            "This series is published by Destatis as part of its open data program."
        )
    return f"""---
license: other
language:
- en
tags:
- destatis
- germany
- open-data
- ml-ready
- tabular
---

# {title}

## Unofficial Notice
This repository is a **private open-source project** and is **not an official** repository of the Federal Statistical Office of Germany (Destatis).

## Data Explanation
- Official dataset label: {label}
- Source CSV: {source_url}
- Source page: {page_url or "n/a"}
- Snapshot timestamp (UTC): {now_iso}

## Overview
This dataset contains a machine-learning-ready tabular version of a Destatis open-data series.
The original CSV is transformed with deterministic preprocessing rules to support reproducible ML workflows.

## Dataset Structure
Files:
- `data.csv`: normalized table
- `README.md`: dataset card with provenance and processing notes

Columns (sample):
{column_lines}

## Field Notes
{field_notes_block}

## Processing Pipeline
The source CSV is processed without AI generation:
- delimiter normalization
- header normalization (`snake_case`)
- missing-value token normalization
- German numeric normalization (e.g. `1.234,56` -> `1234.56`)
- row consistency checks

## Data Quality Notes
- Rows: {rec.rows}
- Columns: {rec.columns}
- Numeric columns ratio: {rec.numeric_ratio:.2f}
- Missing ratio: {rec.missing_ratio:.2f}
- Quality note: {quality_note}

## Official Context Snippet
{context_clean}

## Intended Use
This dataset is suitable for:
- time-series baseline modeling
- tabular feature engineering
- analytics and reproducible benchmarking

## Limitations
- Official revisions can update historical values.
- Indicator semantics follow Destatis conventions and may require domain context.

## License
Source data rights follow the official Destatis open-data terms.
Repository metadata uses `license: other` for Hugging Face compatibility.

## Maintainer
Maintained by the `destatis` Hugging Face organization (community-run, unofficial).
"""


def upload_multi_repo(
    api: HfApi,
    rec: FileRecord,
    csv_path: Path,
    doc: CsvDoc | None,
    header: list[str],
    now_iso: str,
    existing_repo_ids: set[str] | None = None,
) -> str:
    short_repo_name = repo_slug_from_relpath(rec.rel_path)
    legacy_repo_name = legacy_repo_slug_from_relpath(rec.rel_path)
    preferred_repo_id = f"{HF_NAMESPACE}/{short_repo_name}"
    legacy_repo_id = f"{HF_NAMESPACE}/{legacy_repo_name}"
    repo_id = rec.hf_repo_id if repo_belongs_to_namespace(rec.hf_repo_id, HF_NAMESPACE) else preferred_repo_id
    if HF_README_ONLY_BACKFILL:
        if existing_repo_ids is None:
            return ""
        # README-only mode must target existing repos only.
        # Try in order: explicit repo_id from manifest (if org), new short slug, old legacy slug.
        candidates = [repo_id, preferred_repo_id, legacy_repo_id]
        repo_id = next((c for c in candidates if c in existing_repo_ids), "")
        if not repo_id:
            return ""
    else:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    if not HF_README_ONLY_BACKFILL:
        api.upload_file(path_or_fileobj=str(csv_path), path_in_repo="data.csv", repo_id=repo_id, repo_type="dataset")

    tmp_readme = ROOT / "metadata" / ".tmp_readme.md"
    # Keep README title clean and stable regardless of legacy repository naming.
    readme_title = short_repo_name
    tmp_readme.write_text(dataset_readme(readme_title, rec.source_url, doc, rec, header, now_iso), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(tmp_readme), path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
    tmp_readme.unlink(missing_ok=True)
    return repo_id


def repo_belongs_to_namespace(repo_id: str, namespace: str) -> bool:
    if not repo_id or "/" not in repo_id:
        return False
    return repo_id.split("/", 1)[0] == namespace


def main() -> int:
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN missing")
    if HF_NAMESPACE != HF_REQUIRED_NAMESPACE:
        raise SystemExit(
            f"Refusing to publish: HF_NAMESPACE='{HF_NAMESPACE}' does not match "
            f"HF_REQUIRED_NAMESPACE='{HF_REQUIRED_NAMESPACE}'."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ML_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    api = HfApi(token=HF_TOKEN)

    csv_links, csv_docs = crawl_csv_links(session)
    manifest = load_manifest()
    known = manifest.get("files", {})
    existing_repo_ids: set[str] | None = None
    if HF_PUBLISH_MODE == "multi" and HF_README_ONLY_BACKFILL:
        existing_repo_ids = {d.id for d in api.list_datasets(author=HF_NAMESPACE)}

    now = datetime.now(timezone.utc).isoformat()
    updated_records: list[FileRecord] = []
    new_or_changed = 0
    kept_ml = 0
    published_multi = 0
    backfilled_multi = 0

    for url in csv_links:
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            raw = resp.content
        except requests.RequestException:
            continue

        sha = hashlib.sha256(raw).hexdigest()
        rel = safe_path_from_url(url)
        prev = known.get(rel)
        if prev and prev.get("sha256") == sha:
            # No source change, but in multi mode we may still need to publish
            # a per-file dataset repo for already processed ML-ready files.
            if (
                HF_PUBLISH_MODE == "multi"
                and prev.get("ml_ready")
                and not repo_belongs_to_namespace(str(prev.get("hf_repo_id", "")), HF_NAMESPACE)
            ):
                ml_path = ML_DIR / rel
                if not ml_path.exists():
                    header, rows, meta = parse_ml_ready(decode_bytes(raw))
                    if bool(meta.get("ml_ready", False)) and header and rows:
                        ml_path.parent.mkdir(parents=True, exist_ok=True)
                        with ml_path.open("w", encoding="utf-8", newline="") as f:
                            w = csv.writer(f)
                            w.writerow(header)
                            w.writerows(rows)
                if ml_path.exists():
                    prev_rec = FileRecord(
                        source_url=prev.get("source_url", url),
                        rel_path=rel,
                        sha256=sha,
                        bytes_size=int(prev.get("bytes_size", len(raw))),
                        rows=int(prev.get("rows", 0)),
                        columns=int(prev.get("columns", 0)),
                        numeric_ratio=float(prev.get("numeric_ratio", 0)),
                        missing_ratio=float(prev.get("missing_ratio", 1)),
                        inconsistent_rows=int(prev.get("inconsistent_rows", 0)),
                        ml_ready=bool(prev.get("ml_ready", True)),
                        note=str(prev.get("note", "")),
                        hf_repo_id=str(prev.get("hf_repo_id", "")),
                        last_seen_utc=now,
                    )
                    existing_header: list[str] = []
                    try:
                        with ml_path.open(encoding="utf-8") as f:
                            existing_header = next(csv.reader(f))
                    except Exception:
                        existing_header = []
                    hf_repo_id = upload_multi_repo(
                        api, prev_rec, ml_path, csv_docs.get(url), existing_header, now, existing_repo_ids
                    )
                    if not hf_repo_id:
                        known[rel]["last_seen_utc"] = now
                        continue
                    known[rel]["hf_repo_id"] = hf_repo_id
                    known[rel]["last_seen_utc"] = now
                    published_multi += 1
                    backfilled_multi += 1
                    continue

            known[rel]["last_seen_utc"] = now
            continue

        new_or_changed += 1
        raw_path = RAW_DIR / rel
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)

        header, rows, meta = parse_ml_ready(decode_bytes(raw))
        ml_ready = bool(meta.get("ml_ready", False))

        ml_path = ML_DIR / rel
        hf_repo_id = ""
        if ml_ready and header and rows:
            ml_path.parent.mkdir(parents=True, exist_ok=True)
            with ml_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
            kept_ml += 1
        elif ml_path.exists():
            ml_path.unlink()

        rec = FileRecord(
            source_url=url,
            rel_path=rel,
            sha256=sha,
            bytes_size=len(raw),
            rows=int(meta.get("rows", 0)),
            columns=int(meta.get("columns", 0)),
            numeric_ratio=float(meta.get("numeric_ratio", 0)),
            missing_ratio=float(meta.get("missing_ratio", 1)),
            inconsistent_rows=int(meta.get("inconsistent_rows", 0)),
            ml_ready=ml_ready,
            note=str(meta.get("note", "")),
            hf_repo_id=str(prev.get("hf_repo_id", "")) if prev else "",
            last_seen_utc=now,
        )

        if ml_ready and HF_PUBLISH_MODE == "multi":
            hf_repo_id = upload_multi_repo(api, rec, ml_path, csv_docs.get(url), header, now, existing_repo_ids)
            if hf_repo_id:
                rec.hf_repo_id = hf_repo_id
                published_multi += 1

        updated_records.append(rec)
        known[rel] = asdict(rec)

    manifest["files"] = known
    manifest["updated_at"] = now
    write_manifest(manifest)
    write_report(updated_records)

    if HF_PUBLISH_MODE == "single":
        upload_single_repo(api)

    summary = {
        "timestamp_utc": now,
        "found_links": len(csv_links),
        "new_or_changed": new_or_changed,
        "ml_ready_added_or_updated": kept_ml,
        "publish_mode": HF_PUBLISH_MODE,
        "readme_only_backfill": HF_README_ONLY_BACKFILL,
        "single_repo_id": HF_REPO_ID if HF_PUBLISH_MODE == "single" else "",
        "multi_repo_namespace": HF_NAMESPACE if HF_PUBLISH_MODE == "multi" else "",
        "multi_repos_published": published_multi,
        "multi_repos_backfilled": backfilled_multi,
    }
    write_runlog(summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
