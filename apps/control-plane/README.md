# Control Plane

Axum-based HTTP API scaffold for the incremental verification control plane.

Current state:

- `/healthz` is implemented
- `POST /v1/demo/verification-jobs` runs the canonical persisted verification flow
- `GET /v1/verification-runs/{run_id}` fetches a stored run record
- `POST /v1/verification-runs` remains a `501 Not Implemented` placeholder for the broader run API

TODO:

- replace the canonical SQLite-backed path with the broader production persistence model
- wire the non-demo run-submission endpoints
