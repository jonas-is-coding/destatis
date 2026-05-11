# Destatis Daily ML Sync

Automatischer Daily-Job, der Destatis-Open-Data-CSV-Dateien crawlt, neue/aktualisierte Dateien erkennt,
ML-Eignung prüft, normalisierte CSVs erzeugt und nach Hugging Face hochlädt.

## Inoffizieller Hinweis
Dieses Repository ist ein privates Open-Source-Projekt und nicht offiziell vom Statistischen Bundesamt (Destatis) betrieben.

## GitHub Secrets
- `HF_TOKEN`: Hugging Face Token mit Write-Rechten für die Organisation `destatis`

## Namespace Safety
- Der Workflow veröffentlicht standardmäßig in `destatis`.
- Ein Safety-Check bricht den Run ab, falls `HF_NAMESPACE` nicht `destatis` ist.

## Zeitplan
- Täglich 06:00 UTC (08:00 Berlin im Sommer)
- Manuell per `workflow_dispatch`

## Output
- `data/raw`: Rohdaten
- `data/ml_ready`: normalisierte, ML-geeignete CSVs
- `metadata/manifest.json`: Zustand/Hashes zum Delta-Vergleich
- `metadata/latest_sync_report.csv`: Übersicht geänderter Dateien im letzten Lauf
- `metadata/runs.jsonl`: Laufhistorie
