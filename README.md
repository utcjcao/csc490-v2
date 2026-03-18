# Incremental Verification for ML Models

This repository contains the Phase 1 MVP scaffold for an incremental verification system for machine learning models. The current codebase is intentionally thin: it establishes the module boundaries, contracts, tooling, and development workflow from the architecture docs without implementing the full verification logic yet.

Source-of-truth design docs:

- [architecture.md](./architecture.md)
- [repo_structure.md](./repo_structure.md)
- [implementation_plan.md](./implementation_plan.md)

## Current Status

The scaffold includes:

- a Rust workspace for the control plane, CLI, domain, contracts, and infrastructure adapters
- Python package skeletons for shared worker code and the first verifier adapter
- a local SQLite-backed persisted run store for the canonical verification flow
- versioned schema placeholders for API and worker contracts
- local development scripts, lint/test config, Docker placeholders, and migration placeholders

Not implemented yet:

- real PostgreSQL persistence for the full system
- real blob storage integration
- real verification job execution
- real alpha-beta-CROWN integration
- real diffing and reuse planning logic

## Repository Layout

- `apps/`: binaries for the control plane API and CLI
- `crates/`: Rust libraries for domain logic, application workflows, contracts, and infra adapters
- `python/`: shared worker utilities and the alpha-beta-CROWN adapter package
- `db/`: database migrations
- `schemas/`: versioned OpenAPI and worker manifest schemas
- `infra/`: Docker, Compose, and future Kubernetes manifests
- `tests/`: integration, contract, and e2e test locations
- `scripts/`: local development helpers

## Prerequisites

- Rust `1.94+`
- Python `3.12+`
- Docker and Docker Compose for local infra later in Phase 1

## Quick Start

1. Check the Rust workspace:

   ```powershell
   cargo check --workspace
   ```

2. Run the control plane health server:

   ```powershell
   cargo run -p ivm-control-plane
   ```

3. Inspect the CLI skeleton:

   ```powershell
   cargo run -p ivm-cli -- --help
   ```

4. Install Python dev tooling and editable worker packages:

   ```powershell
   .\scripts\bootstrap.ps1
   ```

5. Run local checks:

   ```powershell
   .\scripts\check.ps1
   ```

## Canonical Demo

The first end-to-end Phase 1 vertical slice is a single canonical verification job flow:

- CLI or HTTP request
- application-layer parsing, semantic validation, and normalization
- persisted run lifecycle in SQLite: `pending -> running -> completed/failed`
- subprocess execution request to the Python adapter
- deterministic structured result returned to Rust

### Validation and normalization behavior

Before the adapter is called, the Rust application layer converts the canonical API request into a validated internal execution spec.

- `model_storage_uri` is trimmed, must include a URI scheme, and must point to an `.onnx` artifact
- `model_sha256` is trimmed and normalized to lowercase
- `input_region` must be JSON matching the Phase 1 canonical shape: `{"eps": <number>, "norm": "linf"}` where `norm` is optional and `0 < eps <= 1`
- `output_constraint` must be JSON matching the Phase 1 canonical shape: `{"label": <u32>}`
- requests are classified consistently as input-validation errors, unsupported-feature errors, adapter/runtime errors, or internal invariant violations

Fixture files for these cases live in [fixtures](./fixtures):

- `canonical_verification_request.valid.json`
- `canonical_verification_request.malformed.json`
- `canonical_verification_request.semantic_invalid.json`
- `canonical_verification_request.unsupported.json`

The persisted run store uses `IVM_CANONICAL_RUN_DB` when set. By default it writes to `data/dev/verification-runs.sqlite3`.

### Run the demo from the CLI

```powershell
cargo run -p ivm-cli -- verify demo --input .\fixtures\canonical_verification_request.json
```

Expected behavior:

- the command exits successfully
- the result JSON includes `status: "completed"` and `outcome: "proved"`
- the normalized execution spec sent to the adapter uses trimmed model URIs, lowercase model digests, and canonicalized property JSON
- the command also creates or updates a durable run record in the SQLite store

### Fetch a persisted run from the CLI

Use the `run_id` returned from the submit command:

```powershell
cargo run -p ivm-cli -- verify status <run_id>
```

Expected behavior:

- the command returns the persisted run record
- the payload includes persisted status, submitted/updated timestamps, the normalized execution spec snapshot, backend identifier, and any stored result or error snapshot

### Run the demo through the control plane

1. Start the server:

   ```powershell
   cargo run -p ivm-control-plane
   ```

2. In a second shell, submit the canonical request:

   ```powershell
   Invoke-RestMethod `
     -Uri http://127.0.0.1:3000/v1/demo/verification-jobs `
     -Method Post `
     -ContentType application/json `
     -InFile .\fixtures\canonical_verification_request.json
   ```

Expected behavior:

- the endpoint returns HTTP `200`
- the JSON body includes `status: "completed"` and `outcome: "proved"`

For validation checks:

- `canonical_verification_request.malformed.json` returns HTTP `400` with code `input_validation_error`
- `canonical_verification_request.semantic_invalid.json` returns HTTP `400` with code `input_validation_error`
- `canonical_verification_request.unsupported.json` returns HTTP `422` with code `unsupported_feature_error`

### Fetch a persisted run through the control plane

Use the `run_id` from the submit response:

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:3000/v1/verification-runs/<run_id> `
  -Method Get
```

Expected behavior:

- the endpoint returns HTTP `200` for a known run and `404` for an unknown run
- the body includes the persisted lifecycle state plus normalized spec, result, and error snapshots

### Trigger the deterministic adapter error path

Send the same request, but change `model_storage_uri` to a value containing `adapter-error`, for example:

```json
{
  "model_storage_uri": "demo://models/adapter-error.onnx"
}
```

Expected behavior:

- the endpoint still returns a structured JSON result
- the result includes `status: "failed"`, `outcome: "error"`, and failure code `stub_adapter_error`

## Development Workflow

- Rust formatting: `cargo fmt --all`
- Rust linting: `cargo clippy --workspace --all-targets -- -D warnings`
- Rust tests: `cargo test --workspace`
- Python linting: `python -m ruff check .`
- Python formatting: `python -m ruff format .`
- Python tests: `python -m pytest`

Use [NEXT_STEPS.md](./NEXT_STEPS.md) as the implementation queue for the next coding passes.
