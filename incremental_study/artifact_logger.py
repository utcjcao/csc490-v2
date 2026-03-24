from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


class ArtifactLogger:
    """Writes bounded-size JSON artifacts for the paired intrinsic study."""

    def __init__(self, output_dir: str | Path, pair_id: str) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.pair_id = sanitize_name(pair_id)
        self.pair_dir = self.output_dir / self.pair_id
        self.pair_dir.mkdir(parents=True, exist_ok=True)
        (self.pair_dir / "original").mkdir(exist_ok=True)
        (self.pair_dir / "perturbed").mkdir(exist_ok=True)

    def write_pair_manifest(self, payload: dict[str, Any]) -> Path:
        return self._write_json(self.pair_dir / "pair_manifest.json", payload)

    def write_run_log(self, model_role: str, property_id: str, payload: dict[str, Any]) -> Path:
        if model_role not in {"original", "perturbed"}:
            raise ValueError(f"Unexpected model_role: {model_role}")
        filename = sanitize_name(property_id) + ".json"
        return self._write_json(self.pair_dir / model_role / filename, payload)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        return path

