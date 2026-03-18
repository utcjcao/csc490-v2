FROM rust:1.94-bookworm AS builder
WORKDIR /workspace

# TODO: copy workspace manifests, build the control-plane binary, and produce a slim runtime image.

FROM debian:bookworm-slim
WORKDIR /app

# TODO: copy the built control-plane binary into the runtime image.
CMD ["sh", "-c", "echo control-plane image scaffold; sleep infinity"]

