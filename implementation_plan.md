# Implementation Roadmap

## Delivery Principles

- Build the smallest system that can run one real baseline flow and one real incremental flow.
- Prove soundness and reproducibility before optimizing for scale.
- Add new verifiers, properties, and infrastructure only after the first end-to-end loop is stable.

## Phase 1: MVP

### Goal

Demonstrate end-to-end incremental verification for one verifier, one model family, and one property type.

### Exact Components to Build

1. Rust workspace with:
   - `apps/control-plane`
   - `apps/cli`
   - `crates/domain`
   - `crates/application`
   - `crates/contracts`
2. Python workspace with:
   - `adapter-common`
   - `alpha-beta-crown-adapter`
3. SQLite-backed persistence for current Phase 1 local/dev flows:
   - canonical verification runs first
   - then the minimum additional metadata needed for Phase 1
   - preserve repository boundaries so PostgreSQL can replace SQLite later without redesign
4. Blob storage abstraction with:
   - local filesystem backend for tests
   - MinIO/S3 backend for real runs
5. Core registration flows:
   - register project
   - register model version
   - register property
   - register verifier profile
6. Baseline verification pipeline:
   - create full run
   - generate worker manifest
   - launch Python adapter
   - ingest result manifest
   - persist metrics, artifacts, and counterexamples
7. Incremental planning pipeline:
   - compute `ChangeSet`
   - generate conservative `ReusePlan`
   - fall back to full verification when compatibility is unknown
8. Incremental execution:
   - pass prior artifact refs to the worker
   - reuse at least one real artifact type
   - record reuse metrics
9. CLI commands:
   - `model register`
   - `property register`
   - `verify run`
   - `verify status`
   - `reuse preview`
10. Local deployment:
   - local process execution with SQLite by default
   - optional Docker Compose for control plane, worker image, and later blob storage

### Exit Criteria

- a user can verify a base model and an updated model end-to-end
- the system stores reusable artifacts and reproducible run records
- incremental runs show measurable runtime difference from full runs
- every run is tied to exact model, property, verifier, and planner versions

## Phase 2: Hardening

### Goal

Make the MVP reliable enough for repeated experiments and moderate team use.

### Exact Components to Build

1. Job durability:
   - heartbeat leases
   - retry policy
   - timeout policy
   - cancellation support
2. Audit subsystem:
   - sample-based full reruns for incremental runs
   - audit comparison records
   - automatic quarantine of suspicious reuse rules
3. Contract versioning:
   - versioned worker manifests
   - versioned artifact manifest schema
   - backward-compatible API strategy
4. Richer diff analysis:
   - layer-hash indexing
   - explicit export and quantization metadata
   - unsupported-change diagnostics
5. Improved reporting:
   - benchmark matrix execution
   - CSV and JSON export
   - per-artifact reuse effectiveness metrics
6. Test expansion:
   - fixture-based end-to-end tests
   - contract tests between Rust and Python
   - golden tests for result manifests
   - failure-path tests for timeout, crash, and invalid artifacts
7. Observability:
   - structured tracing
   - Prometheus metrics
   - dashboards for queue depth, run latency, failure rate, and reuse hit rate
8. Basic security:
   - authenticated API access
   - pinned worker images
   - artifact hash verification on ingestion

### Exit Criteria

- repeated benchmark runs are reproducible
- contract drift between Rust and Python is caught automatically
- incremental runs have an audit path that can detect unsound reuse
- failures are diagnosable without manual ad hoc debugging

## Phase 3: Scale and Advanced Features

### Goal

Expand from a thesis-quality prototype into a broader internal or research platform.

### Exact Components to Build

1. Multi-worker scale-out:
   - multiple worker replicas
   - worker pool selection by verifier profile
   - Kubernetes Jobs or ECS task execution
2. Additional verifier support:
   - second verifier adapter
   - adapter capability negotiation
   - artifact compatibility matrix
3. Additional property and model support:
   - more network architectures
   - more property classes beyond local robustness
   - broader transformation classes
4. Smarter reuse planning:
   - artifact scoring
   - policy experimentation
   - planner plugins or rule packs
5. Read-only web dashboard:
   - run history
   - lineage visualization
   - benchmark views
   - counterexample inspection
6. Operational controls:
   - quotas
   - retention tiers
   - per-project worker limits
   - archival policies for old artifacts

### Exit Criteria

- the platform can run many verification jobs concurrently without manual orchestration
- adding a new verifier does not require redesigning the core domain
- reports are strong enough for thesis figures and recurring team reviews

## Recommended Build Order Inside Phase 1

1. Define domain entities and invariants.
2. Define API and worker contracts.
3. Stand up SQLite-backed persistence and only introduce PostgreSQL when async queueing or shared multi-process metadata becomes necessary.
4. Implement model and property registration.
5. Integrate one real verifier adapter for full verification.
6. Persist baseline artifact manifests.
7. Implement conservative diff and reuse planning.
8. Execute one true incremental run.
9. Add reproducible reports and fixture-based tests.

This ordering gets a usable system quickly while keeping soundness-critical decisions explicit.

## MVP Scope Lock

Do not let MVP expand beyond:

- one verifier adapter
- one model family
- one property type
- one reusable artifact type
- one nontrivial transformation path beyond trivial checkpoint updates

If those five pieces are stable, later phases are extension work rather than redesign work.
