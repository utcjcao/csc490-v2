# Next Steps

The first 10 concrete coding tasks for Phase 1, in order:

1. Extend the current SQLite-backed persistence model beyond canonical runs to cover the minimum additional Phase 1 metadata needed next.
2. Keep the persistence ports storage-agnostic and only replace or add the `PostgresRepositories` adapter once async queueing or broader shared metadata makes PostgreSQL necessary.
3. Implement the local filesystem and MinIO/S3 blob store flows in `crates/blobstore`, including SHA-256 verification on write and read.
4. Add environment-based configuration loading for the control plane, persistence layer, and blob storage clients.
5. Implement the project, model, property, and verifier-profile registration use cases in `crates/application` and expose them through API and CLI commands.
6. Flesh out the worker manifest and result manifest contracts, then add Rust/Python contract serialization tests under `tests/contract`.
7. Build the verification-run submission flow: persist the run, materialize the manifest, enqueue the job, and expose status lookup.
8. Implement the first real alpha-beta-CROWN worker adapter path for baseline verification in `python/alpha-beta-crown-adapter`.
9. Add conservative ONNX parent/child diffing and the first reusable artifact policy to produce a real `ReusePlan`.
10. Add end-to-end fixture-driven tests that compare a full run against an incremental run and capture runtime and reuse metrics.
