from __future__ import annotations

import json
from pathlib import Path

from .contracts import (
    CanonicalVerificationRequest,
    CanonicalVerificationResponse,
    ResultManifest,
    canonical_request_from_dict,
    run_manifest_from_dict,
)


def load_canonical_request(path: str | Path) -> CanonicalVerificationRequest:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return canonical_request_from_dict(data)


def write_canonical_response(path: str | Path, response: CanonicalVerificationResponse) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(response.to_dict(), handle, indent=2)
        handle.write("\n")


def load_run_manifest(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return run_manifest_from_dict(data)


def write_result_manifest(path: str | Path, manifest: ResultManifest) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)
        handle.write("\n")
