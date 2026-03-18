# Architecture Design: Incremental Verification for ML Models

## Scope and Product Summary

### Purpose

Build a system that verifies neural-network safety properties across model updates without redoing all proof work from scratch. It should persist prior verification artifacts, detect what changed, reuse only artifacts that remain sound, recompute invalidated work, and report both correctness and speedup.

This is a verification orchestration platform, not a model-serving platform.

### Primary Users

- ML verification researcher or thesis student
- ML engineer integrating verification into model update workflows
- Reviewer consuming reproducible experiment outputs

### Core Workflows

1. Register a base model, property, and verifier profile.
2. Run full verification and store artifacts, metrics, and counterexamples.
3. Register a derived model version after checkpoint update, export, or quantization.
4. Compute a model diff and create a reuse plan.
5. Run incremental verification using valid prior artifacts plus targeted recomputation.
6. Compare baseline vs incremental runs and export benchmark reports.

### Ambiguities and Minimal Assumptions

The brief leaves several choices open. To keep the design buildable, the default architecture assumes:

- one verifier in MVP
- recommended first verifier adapter: `alpha-beta-CROWN`
- one model family in MVP: feed-forward ReLU networks exported to ONNX
- one property type in MVP: local `L_inf` robustness
- initial transformations: checkpoint update, ONNX re-export, one conservative quantization path
- trusted internal users first; multi-tenant SaaS concerns are deferred
- fail-closed reuse policy: if sound reuse cannot be proven, recompute

## Recommended Architecture

### Style

Use a **Rust modular monolith control plane** with **isolated Python verifier workers**.

Why:

- keeps core workflow, invariants, and contracts in one codebase
- avoids fighting the Python-heavy ML and verifier ecosystem
- isolates verifier crashes and dependency conflicts
- scales from laptop to worker pool without redesign

### Major Subsystems

| Subsystem | Responsibility | Interfaces | State Owned |
|---|---|---|---|
| Control Plane API | request validation, REST API, CLI-facing endpoints | REST/JSON, OpenAPI | API versioning, request metadata |
| Model & Property Registry | model ingestion, hashing, lineage, property registration | app services, blob store | model metadata, lineage edges, property specs |
| Diff Engine | compare parent and child models | internal service | `ChangeSet`, layer hashes, transform metadata |
| Artifact Catalog | store artifact manifests and provenance | app services, blob store | artifact refs, hashes, validity scope |
| Reuse Planner | decide what can be reused safely | planner rules | `ReusePlan`, invalidation reasons, planner version |
| Execution Engine | create jobs, lease jobs, enforce timeouts/retries | job table, worker launcher | run state, job state, worker leases |
| Verifier Workers | execute full or incremental verification | manifest in, result manifest out | ephemeral only |
| Results & Reporting | persist outcomes, metrics, reports | read APIs, CLI export | run metrics, counterexamples, reports |
| Audit & Observability | audit reruns, metrics, traces, logs | OpenTelemetry, Prometheus | audit results, telemetry |

### Boundary Rules

- `domain` owns invariants and soundness-sensitive logic.
- `application` orchestrates use cases and depends on traits, not concrete infra.
- infra adapters own SQL, blob storage, worker launch, and telemetry.
- workers never write directly to the metadata database.
- artifact payloads may be verifier-specific; artifact manifests must be platform-stable.

### Data Flow

1. User registers model and property.
2. Control plane hashes and stores the model in object storage; current Phase 1 local/dev metadata goes to SQLite through the persistence port. PostgreSQL remains the intended migration target once the broader metadata model and async job orchestration land.
3. A full or incremental verification run is created.
4. For incremental runs, the diff engine produces a `ChangeSet`.
5. The reuse planner emits a `ReusePlan` selecting reusable artifacts and required recomputation.
6. Execution engine creates a durable job and launches a verifier worker.
7. Worker downloads the model and referenced artifacts, runs verification, and emits a result manifest plus blobs.
8. Control plane validates the result, stores artifacts and metrics, and marks the run complete.
9. Audit jobs optionally re-run a sample of incremental runs in full mode to detect unsound reuse.

### Synchronous vs Asynchronous

**Synchronous**

- model registration and validation
- property validation
- verifier capability lookup
- small diff preview
- run status lookup
- report retrieval

**Asynchronous**

- full verification
- incremental verification
- quantization preprocessing
- artifact extraction and cleanup
- audit reruns
- benchmark matrix runs

## Concrete Tech Stack

| Area | Choice | Why |
|---|---|---|
| Control-plane language | Rust | strong typing for workflows, contracts, and state transitions |
| Worker language | Python | existing verifier and ML tooling live here |
| Backend framework | Axum + Tokio | minimal, async, production-proven |
| CLI | Clap | good fit for local experiments and CI |
| DB | SQLite for current Phase 1 local/dev; PostgreSQL later | SQLite fits the current single-node run-lifecycle workflow with minimal ops, while PostgreSQL becomes the better fit once the system grows into shared metadata and queueing |
| Blob storage | MinIO / S3 | large models and artifacts should not live in DB |
| Queue | none yet in current Phase 1; PostgreSQL job table later | the current slice is synchronous and single-process; durable queueing becomes necessary only once workers are decoupled |
| Worker isolation | Docker containers | reproducible verifier environments |
| Frontend | none in MVP; optional Next.js read-only dashboard later | avoid UI work before core pipeline is proven |
| Testing | `cargo nextest`, `pytest`, contract tests, Docker Compose e2e | covers Rust core, Python workers, and integration |
| Observability | `tracing`, OpenTelemetry, Prometheus, Grafana, Loki | enough to debug long-running verification jobs |
| Deployment | Docker Compose first, Kubernetes Jobs or ECS later | simple prototype path with clear scale-out option |

## System Contracts

### Core Entities

| Entity | Key Fields |
|---|---|
| `Project` | `id`, `name`, `created_at` |
| `ModelLineage` | `id`, `project_id`, `root_model_id` |
| `ModelVersion` | `id`, `lineage_id`, `parent_model_id`, `format`, `sha256`, `architecture_fingerprint`, `weights_digest`, `transform_type`, `transform_metadata`, `storage_uri` |
| `PropertySpec` | `id`, `project_id`, `property_type`, `input_region`, `output_constraint`, `normalization`, `sha256` |
| `VerifierProfile` | `id`, `name`, `version`, `adapter_image`, `supported_formats`, `supported_property_types`, `artifact_types` |
| `ChangeSet` | `id`, `source_model_id`, `target_model_id`, `change_class`, `layer_deltas`, `numeric_delta_summary`, `compatible_for_incremental` |
| `ArtifactBundle` | `id`, `source_run_id`, `artifact_type`, `storage_uri`, `artifact_hash`, `validity_scope`, `verifier_profile_id`, `schema_version` |
| `ReusePlan` | `id`, `changeset_id`, `baseline_run_id`, `selected_artifact_ids`, `invalidated_artifact_ids`, `recompute_steps`, `soundness_basis`, `planner_version` |
| `VerificationRun` | `id`, `model_id`, `property_id`, `verifier_profile_id`, `mode`, `status`, `outcome`, `metrics`, `reuse_plan_id` |
| `Counterexample` | `id`, `run_id`, `input_blob_uri`, `observed_output`, `expected_constraint` |
| `AuditRun` | `id`, `incremental_run_id`, `baseline_recheck_run_id`, `comparison_result` |

### Public API

- `POST /projects`
- `POST /projects/{project_id}/models`
- `POST /projects/{project_id}/properties`
- `POST /verifier-profiles`
- `POST /verification-runs`
- `GET /verification-runs/{run_id}`
- `POST /verification-runs/{run_id}/cancel`
- `GET /models/{model_id}/artifacts`
- `POST /reuse-plans/preview`
- `GET /reports/benchmark`

### Worker Contract

Input manifest should contain:

- run id and mode
- model ref and hash
- property spec
- baseline run ref and artifact refs for incremental mode
- reuse plan ref
- verifier profile
- execution limits

Output manifest should contain:

- final status and outcome
- runtime metrics
- emitted artifact manifests
- optional counterexample
- typed failure info

### Invariants

- model files are immutable once hashed and stored
- a result is valid only for `model hash + property hash + verifier profile + artifact set + planner version`
- reuse is allowed only when a planner rule explicitly marks it safe for the computed change set
- unknown compatibility means invalidate and recompute
- only the control plane commits durable metadata

### Failure Handling

1. Input errors: reject synchronously with validation details.
2. Planning errors: mark the run failed or downgrade to full verification if policy allows.
3. Worker failures: capture typed crash, timeout, or OOM status and retry only when safe.
4. Soundness-risk failures: fail closed, quarantine the rule or artifact type, and require full rerun.

## Engineering Quality

### Scalability Bottlenecks

- verification runtime dominates cost: keep runs async and horizontally scalable
- artifact storage grows quickly: keep payloads in object storage and manifests in DB
- diff computation for large models: precompute architecture fingerprints and layer hashes
- metadata hot spots: use immutable run records and append-only event history where possible

### Security Concerns

- treat uploaded models as untrusted input
- run verifiers in isolated containers
- hash all stored artifacts and bind them to source run and verifier profile
- keep secrets out of manifests and inject them only at deployment

### Reliability Concerns

- heartbeat leases and idempotent retries for long-running jobs
- commit run completion only after result manifest and blobs are durable
- add audit reruns early to catch unsound reuse assumptions
- isolate each run in its own worker container or clean process

### Prototype-to-Production Path

Prototype:

- single machine
- local process execution first; Docker Compose remains optional
- one Rust control plane
- one Python worker image
- SQLite for metadata
- local filesystem and later MinIO for blobs
- CLI-first

Production path:

- multiple worker replicas
- stronger auth and quotas
- scheduled audit and retention jobs
- Kubernetes Jobs or ECS tasks for worker scale-out

### What Must Be Real Early

Real early:

- model hashing and immutable storage
- one real verifier adapter
- one real artifact type
- one real diff and reuse path
- one real full-vs-incremental benchmark flow

Can be mocked later:

- web dashboard
- advanced auth
- multi-verifier support
- distributed broker beyond PostgreSQL

## Recommendation and Alternatives

### Recommended Default Architecture

For the current Phase 1 stage: Rust modular monolith control plane, SQLite for local metadata persistence, stable JSON worker manifests, one Python verifier worker, CLI plus REST API, and a persistence port that preserves PostgreSQL as the later migration target for shared metadata and queueing.

### Lightweight Alternative

If the project must stay even smaller than the current plan, reduce it to a CLI-only runner with SQLite, filesystem storage, and a subprocess-based Python adapter. That is viable for a thesis prototype but should be retired once shared services, concurrent workers, or larger artifact inventories appear.

### Biggest Risks / Unknowns

1. The chosen verifier may not expose reusable artifacts cleanly enough.
2. Reuse rules may need to be so conservative that speedup is limited at first.
3. Export or quantization transformations may invalidate more proof state than expected.
4. Artifact manifests may drift if verifier-specific details leak into core contracts too early.
5. Benchmark credibility depends on storing exact model hashes, worker image versions, and planner versions from day one.
