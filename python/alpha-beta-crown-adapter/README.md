# alpha-beta-crown-adapter

Phase 1 scaffold for the first verifier adapter.

Current behavior:

- accepts a worker input manifest
- writes a deterministic placeholder result manifest
- does not call the real verifier yet

TODO:

- load ONNX models
- translate properties into verifier inputs
- recover reusable artifacts
- execute baseline and incremental verification modes

