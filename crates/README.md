# Crates

The Rust workspace is split by responsibility:

- `domain`: core entities, invariants, and soundness-sensitive rules
- `application`: use cases and ports
- `contracts`: API and worker DTOs
- `persistence`: SQLite adapters for current Phase 1 local/dev work and future PostgreSQL adapters
- `blobstore`: local and object-store adapters
- `execution`: queueing and worker launch mechanics
- `observability`: tracing and metrics wiring
