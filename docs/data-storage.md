# Data storage and privacy

Keep private configuration, documents, logs, databases and browser profiles out of
source control. Ignore rules prevent accidental inclusion; they do not encrypt data
or remove earlier Git history. The application does not provide storage encryption.

## Locations and retained data

A local checkout normally uses ignored `.local/`. For a network checkout, Windows
uses `%LOCALAPPDATA%/AIBrowserAgent/data`; other supported environments use the local
`ai-browser-agent` data directory. `BROWSER_DATA_ROOT` can select an explicit absolute
local directory. Live SQLite stores reject network storage. Existing stores are not
automatically relocated.

| Data | Handling |
| --- | --- |
| `backend/.env` | Private backend pairing token and provider secrets |
| Chrome local extension storage | Backend connection settings and pairing token; no provider API keys |
| `documents.sqlite3` | Extracted PDF metadata, pages and chunks |
| `retrieval.sqlite3` | Text retrieval index, vectors and provenance |
| `acquired/prepared/` | Retained, hash-validated PDF copies for later rendering |
| `acquired/provenance/` | Document acquisition provenance |
| Browser downloads under `AIBrowserAgent/` | Original files; preparation does not imply their deletion |
| Acquisition staging files | Normally cleaned by the workflow; interrupted processes can leave residue |
| Audit/log files | Operational records; do not assume automatic expiry or rotation |
| Model and visual caches | Local artifacts with their own paths and lifecycle |

## Backup and removal

Stop the backend and active document work before copying databases. Back up a
consistent source/index pair and any retained PDFs needed for rendering. Model caches
can be prepared again from their configured versions; preserve configuration and
model/license information needed to reproduce the environment.

Use `tools/pdf_ingest.py --help` and `tools/local_retrieval.py --help` for document
and index management. Removing one database record does not erase browser downloads,
retained copies, logs or backups. Review those separately before deleting private
files. Keep rollback copies private and verify compatibility before restoring indexes.

Use `tools/public_privacy_scan.py` in a Git working tree for value-redacted publication
findings. It is a pattern scan with limited coverage, not proof that data is safe to
publish. Review source snapshots and repository history separately. Cloud provider
requests transmit selected context off the machine; review the chosen provider's
handling of that data.

See [architecture](architecture.md) and the [README](../README.md).
