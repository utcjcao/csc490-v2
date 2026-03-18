# Proposed Repository Structure

## Design Principles

- Keep the project as a single workspace / monorepo.
- Separate domain logic from orchestration and infrastructure.
- Keep worker contracts versioned and shared across Rust and Python.
- Treat verifier-specific code as adapters, not as core business logic.
- Keep fixtures and experiment assets in-repo for reproducibility.

## Top-Level Layout

```text
/
|-- architecture.md
|-- repo_structure.md
|-- implementation_plan.md
|-- Cargo.toml
|-- Cargo.lock
|-- apps/
|   |-- control-plane/
|   `-- cli/
|-- crates/
|   |-- domain/
|   |-- application/
|   |-- contracts/
|   |-- persistence/
|   |-- blobstore/
|   |-- execution/
|   `-- observability/
|-- python/
|   |-- adapter-common/
|   `-- alpha-beta-crown-adapter/
|-- db/
|   `-- migrations/
|-- schemas/
|   |-- api/
|   `-- worker/
|-- fixtures/
|   |-- models/
|   |-- properties/
|   `-- expected/
|-- tests/
|   |-- integration/
|   |-- contract/
|   `-- e2e/
|-- infra/
|   |-- docker/
|   |-- compose/
|   `-- k8s/
|-- scripts/
`-- docs/
    |-- adr/
    `-- experiments/
```

## Folder Responsibilities

### `apps/`

- `control-plane/`: Axum server entrypoint, routing, config, dependency wiring
- `cli/`: Clap entrypoint, CI-friendly commands, local operator UX

These apps should stay thin and call application-layer use cases.

### `crates/domain/`

Pure business logic:

- entities
- value objects
- invariants
- soundness rules
- domain errors

This crate must not depend on SQL, HTTP, Docker, or Python-specific code.

### `crates/application/`

Workflow orchestration:

- register model
- register property
- compute change set
- preview reuse
- submit verification run
- ingest worker result
- generate report

This layer should define ports such as `ModelRepository`, `ArtifactRepository`, `BlobStore`, `JobQueue`, and `WorkerLauncher`.

### `crates/contracts/`

Stable cross-boundary types:

- REST DTOs
- CLI DTOs when shared
- worker manifests
- schema version identifiers
- exported OpenAPI / JSON Schema

This crate is the source of truth for contracts shared between Rust and Python.

### `crates/persistence/`

Persistence adapters behind application ports:

- SQLite repositories for current Phase 1 local/dev persistence
- future PostgreSQL repositories for the broader metadata model
- queries and serialization helpers
- migration helpers for the eventual PostgreSQL cutover

No domain rules should live here.

### `crates/blobstore/`

Large binary handling:

- local filesystem backend for dev/test
- MinIO/S3 backend for real runs
- upload/download helpers
- content hash verification

### `crates/execution/`

Async job and worker lifecycle:

- job leasing
- retries and timeouts
- manifest staging
- container or subprocess launch integration
- worker heartbeat tracking

### `crates/observability/`

- tracing setup
- metrics emitters
- OpenTelemetry wiring
- correlation IDs and log helpers

### `python/adapter-common/`

Shared Python worker utilities:

- manifest parsing
- result writing
- blob download/upload helpers
- common error envelopes

### `python/alpha-beta-crown-adapter/`

First verifier adapter:

- verifier-specific translation
- model loading
- artifact extraction
- full and incremental execution

When a second verifier is added, it should be a sibling folder, not code mixed into the control plane.

### `db/migrations/`

- relational schema migrations
- optional seed data for verifier profiles

### `schemas/`

- `api/`: exported OpenAPI and JSON schemas for REST contracts
- `worker/`: versioned schemas for worker manifests and result payloads

### `fixtures/`

Static reproducible assets:

- small ONNX models
- property definitions
- expected result manifests
- counterexample fixtures

### `tests/`

- `integration/`: Rust integration tests for persistence and storage adapters; SQLite now, PostgreSQL and MinIO once those adapters are active
- `contract/`: schema compatibility tests between Rust and Python
- `e2e/`: full baseline and incremental flows

### `infra/`

- `docker/`: Dockerfiles
- `compose/`: local stack
- `k8s/`: future scale-out manifests

### `scripts/`

- schema export
- fixture import
- benchmark wrappers
- dev setup helpers

### `docs/`

- `adr/`: architecture decision records
- `experiments/`: benchmark notes and thesis outputs

## Shared Types and Contracts

Organize contracts around boundaries, not implementation details.

- Put stable cross-process types in `crates/contracts/`.
- Export worker schemas into `schemas/worker/`.
- Export API schemas into `schemas/api/`.
- Keep domain entities in `crates/domain/`; do not use transport DTOs as the domain model unless the boundary is lossless.

Recommended split:

- `contracts::api::*`
- `contracts::worker::*`
- `contracts::report::*`

## Domain vs Infrastructure Separation

Use a strict dependency direction:

```text
apps -> application -> domain
apps -> application -> contracts
infrastructure crates -> application + domain + contracts
python adapters -> worker contracts only
```

Rules:

- `domain` cannot import `persistence`, `blobstore`, or `execution`
- `application` depends on traits, not concrete DB or storage code
- `persistence`, `blobstore`, and `execution` are replaceable adapters
- Python adapters do not own reuse-policy business decisions

## Ownership Boundaries

- `domain`: invariants and soundness-sensitive rules
- `application`: use-case sequencing and state transitions
- `contracts`: API and worker schemas
- `persistence`: relational storage details
- `blobstore`: binary model and artifact storage
- `execution`: async job lifecycle and worker launch mechanics
- `python/*-adapter`: verifier-specific translation and execution only

## Recommended First Modules

1. `crates/domain`
2. `crates/contracts`
3. `crates/application`
4. `crates/persistence`
5. `crates/blobstore`
6. `crates/execution`
7. `apps/control-plane`
8. `apps/cli`
9. `python/adapter-common`
10. `python/alpha-beta-crown-adapter`

That order establishes the core invariants and contracts before infrastructure grows.
