#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "metadata" / "manifest.json"

HF_TOKEN = os.getenv("HF_TOKEN")
HF_NAMESPACE = os.getenv("HF_NAMESPACE", "destatis")
LEGACY_PREFIX = "destatis-ml-static-de-opendata-data-"


def short_slug_from_repo_name(name: str) -> str:
    if name.startswith(LEGACY_PREFIX):
        name = name[len(LEGACY_PREFIX) :]
    name = re.sub(r"[^a-z0-9-]+", "-", name.lower())
    name = re.sub(r"-+", "-", name).strip("-")
    return name[:96].rstrip("-")


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"files": {}, "updated_at": None}


def save_manifest(obj: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN missing")

    api = HfApi(token=HF_TOKEN)
    manifest = load_manifest()
    files = manifest.get("files", {})

    renamed = 0
    skipped_existing = 0
    failed = 0

    datasets = list(api.list_datasets(author=HF_NAMESPACE, full=True))
    existing_ids = {d.id for d in datasets}

    for ds in datasets:
        repo_id = ds.id
        if "/" not in repo_id:
            continue
        ns, name = repo_id.split("/", 1)
        if ns != HF_NAMESPACE:
            continue
        if not name.startswith(LEGACY_PREFIX):
            continue

        new_name = short_slug_from_repo_name(name)
        new_repo_id = f"{HF_NAMESPACE}/{new_name}"
        if new_repo_id == repo_id:
            continue
        if new_repo_id in existing_ids:
            skipped_existing += 1
            continue

        try:
            api.move_repo(from_id=repo_id, to_id=new_repo_id, repo_type="dataset")
            existing_ids.remove(repo_id)
            existing_ids.add(new_repo_id)
            renamed += 1

            for rel_path, meta in files.items():
                if meta.get("hf_repo_id") == repo_id:
                    meta["hf_repo_id"] = new_repo_id
        except Exception:
            failed += 1

    manifest["files"] = files
    save_manifest(manifest)

    print(json.dumps({
        "renamed": renamed,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "namespace": HF_NAMESPACE,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
