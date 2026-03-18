from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "adapter-common" / "src"))
sys.path.insert(0, str(ROOT / "python" / "alpha-beta-crown-adapter" / "src"))

from adapter_common import load_canonical_request, write_canonical_response  # noqa: E402
from alpha_beta_crown_adapter import AlphaBetaCrownAdapter  # noqa: E402


def test_worker_contract_roundtrip(tmp_path: Path) -> None:
    input_request = {
        "run_id": "11111111-1111-1111-1111-111111111111",
        "model_storage_uri": "demo://models/example.onnx",
        "model_sha256": "abc123",
        "input_region": '{"eps":0.01}',
        "output_constraint": '{"label":1}',
        "verifier_name": "alpha-beta-crown",
        "timeout_seconds": 60,
        "memory_mb": 1024,
    }

    input_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    input_path.write_text(json.dumps(input_request), encoding="utf-8")

    request = load_canonical_request(input_path)
    result = AlphaBetaCrownAdapter().run(request)
    write_canonical_response(output_path, result)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == input_request["run_id"]
    assert payload["status"] == "completed"
    assert payload["outcome"] == "proved"
    assert payload["failure"] is None
