from __future__ import annotations


def download_bytes(storage_uri: str) -> bytes:
    raise NotImplementedError(
        f"download_bytes is not implemented for the Phase 1 scaffold: {storage_uri}"
    )


def upload_bytes(storage_uri: str, payload: bytes) -> str:
    raise NotImplementedError("upload_bytes is not implemented for the Phase 1 scaffold")
