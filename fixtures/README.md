# Fixtures

Use this directory for reproducible test models, property specs, and expected outputs.

The scaffold keeps the subdirectories empty until the first real fixtures are added.

The Phase 1 vertical slice also includes:

- `canonical_verification_request.json`: example input for the canonical demo flow
- `canonical_verification_request.valid.json`: valid request used to demonstrate normalization
- `canonical_verification_request.malformed.json`: invalid JSON body for request parsing tests
- `canonical_verification_request.semantic_invalid.json`: valid JSON with invalid canonical semantics
- `canonical_verification_request.unsupported.json`: valid JSON that requests an unsupported feature
