# Apps

This directory contains the two user-facing binaries:

- `control-plane`: the Axum-based HTTP API
- `cli`: the local CLI for experiments and CI flows

Both apps should remain thin and delegate workflow logic to `crates/application`.

