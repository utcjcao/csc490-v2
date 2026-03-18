# CLI

Command-line scaffold for local experiments and CI entry points.

Current commands mirror the architecture docs:

- `project create`
- `model register`
- `property register`
- `verify run`
- `verify status`
- `reuse preview`
- `report export`

Current wiring:

- `verify demo --input <path>` runs the canonical persisted verification flow
- `verify status <run_id>` fetches the stored run record from the local SQLite database
- the remaining commands still return structured placeholders
