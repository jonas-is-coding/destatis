#!/usr/bin/env python3
"""Publish selected Destatis short-term indicator CSVs to a Hugging Face org.

Usage:
  HF_TOKEN=... python3 scripts/publish_destatis_short_term.py

Notes:
- Creates/updates dataset repositories in the configured org.
- Uploads source CSV and README with an explicit unofficial-project disclaimer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import HfApi

ORG = "statistisches-bundesamt"
LICENSE = "Datenlizenz Deutschland – Namensnennung – Version 2.0"


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    title: str
    source_url: str
    filename: str
    description: str


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        slug="destatis-short-term-overnight-stays-bv41",
        title="Destatis Short-term Indicators: Overnight stays (BV 4.1)",
        source_url="https://www.destatis.de/static/de_/opendata/data/uebernachtungen_bv41.csv",
        filename="uebernachtungen_bv41.csv",
        description="Calendar and seasonally adjusted overnight stays from Destatis short-term indicators.",
    ),
    DatasetSpec(
        slug="destatis-short-term-building-permits-total-original",
        title="Destatis Short-term Indicators: Building permits total (original values)",
        source_url="https://www.destatis.de/static/de_/opendata/data/baugenehmigungen_insgesamt_originalwert.csv",
        filename="baugenehmigungen_insgesamt_originalwert.csv",
        description="Total building permits (including work on existing buildings), original values.",
    ),
]


def make_readme(spec: DatasetSpec) -> str:
    return f"""---
license: other
language:
- de
- en
tags:
- destatis
- germany
- official-statistics
- open-data
- tabular
---

# {spec.title}

## Inoffizieller Hinweis / Unofficial note
Dieses Repository ist ein **privates Open-Source-Projekt** und **nicht offiziell** vom Statistischen Bundesamt (Destatis) betrieben.

This repository is a **private open-source project** and is **not an official** repository of the Federal Statistical Office of Germany (Destatis).

## Quelle / Source
- Originalquelle / Original source: {spec.source_url}
- Anbieter / Provider: Statistisches Bundesamt (Destatis)
- Lizenz der Quelldaten / Source data license: {LICENSE}

## Beschreibung / Description
{spec.description}

## Aktualisierung / Updates
Dieses Dataset wird aus einer öffentlich bereitgestellten CSV-Datei übernommen und kann regelmäßig neu synchronisiert werden.
"""


def main() -> int:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN fehlt. Bitte als Umgebungsvariable setzen.")

    api = HfApi(token=token)
    tmp_root = Path("/tmp/destatis_hf_upload")
    tmp_root.mkdir(parents=True, exist_ok=True)

    for spec in DATASETS:
        repo_id = f"{ORG}/{spec.slug}"
        print(f"\\n==> {repo_id}")

        response = requests.get(spec.source_url, timeout=60)
        response.raise_for_status()

        ds_dir = tmp_root / spec.slug
        ds_dir.mkdir(parents=True, exist_ok=True)

        data_path = ds_dir / spec.filename
        data_path.write_bytes(response.content)

        readme_path = ds_dir / "README.md"
        readme_path.write_text(make_readme(spec), encoding="utf-8")

        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(data_path),
            path_in_repo=spec.filename,
            repo_id=repo_id,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

        print(f"Uploaded: https://huggingface.co/datasets/{repo_id}")

    print("\\nFertig.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
