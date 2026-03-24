from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from .artifact_logger import ArtifactLogger
from .config import DatasetConfig, PerturbationConfig, VerificationConfig
from .model_loader import load_model
from .perturbations import create_perturbed_model, summarize_model_delta
from .property_loader import load_properties
from .verifier_adapter import LirpaVerificationAdapter


def _parse_indices(raw: str | None) -> list[int]:
    if not raw:
        return []
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def _build_pair_id(model_path: str, perturbation_mode: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_stem = Path(model_path).stem
    return f"{stamp}_{model_stem}_{perturbation_mode}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone paired intrinsic verification study using auto_LiRPA."
    )
    parser.add_argument("--model-path", required=True, help="Path to the original model checkpoint or ONNX file.")
    parser.add_argument("--model-arch", default=None, help="Optional architecture name for torch checkpoints.")
    parser.add_argument("--dataset", required=True, choices=["mnist", "cifar10"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--normalize", default="none", choices=["none", "ivan-default"])
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--indices", default=None, help="Comma-separated explicit dataset indices.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", default="CROWN-Optimized")
    parser.add_argument("--conv-mode", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--max-logged-split-steps", type=int, default=0)
    parser.add_argument("--input-lower", type=float, default=0.0)
    parser.add_argument("--input-upper", type=float, default=1.0)
    parser.add_argument("--perturbation", default="random_noise", choices=["random_noise", "quantize", "prune"])
    parser.add_argument("--random-std", type=float, default=1e-3)
    parser.add_argument("--random-relative", action="store_true", default=False)
    parser.add_argument("--quant-bits", type=int, default=8)
    parser.add_argument("--prune-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="incremental_study_runs")
    return parser


def run_paired_analysis(args: argparse.Namespace) -> dict[str, Any]:
    dataset_cfg = DatasetConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        eps=args.eps,
        count=args.count,
        start_index=args.start_index,
        indices=_parse_indices(args.indices),
        normalize=args.normalize,
    )
    verifier_cfg = VerificationConfig(
        backend="auto_LiRPA",
        method=args.method,
        device=args.device,
        top_k=args.top_k,
        input_lower=args.input_lower,
        input_upper=args.input_upper,
        timeout_sec=args.timeout_sec,
        max_logged_split_steps=args.max_logged_split_steps,
        conv_mode=args.conv_mode,
    )
    perturb_cfg = PerturbationConfig(
        mode=args.perturbation,
        random_std=args.random_std,
        random_relative=bool(args.random_relative),
        quant_bits=args.quant_bits,
        prune_fraction=args.prune_fraction,
        seed=args.seed,
    )

    pair_id = _build_pair_id(args.model_path, args.perturbation)
    logger = ArtifactLogger(args.output_dir, pair_id=pair_id)

    loaded_original = load_model(
        args.model_path,
        args.dataset,
        architecture=args.model_arch,
        normalization=args.normalize,
    )
    perturbed_model, perturbation_event = create_perturbed_model(loaded_original.model, perturb_cfg)
    delta_summary = summarize_model_delta(loaded_original.model, perturbed_model)

    properties = load_properties(
        args.dataset,
        args.data_root,
        eps=args.eps,
        count=args.count,
        start_index=args.start_index,
        indices=dataset_cfg.indices,
    )

    adapter = LirpaVerificationAdapter(verifier_cfg)
    run_paths: list[dict[str, Any]] = []

    for prop in properties:
        original_log = adapter.verify_property(
            loaded_original.model,
            prop,
            pair_id=pair_id,
            model_role="original",
            model_metadata=loaded_original.to_metadata(),
            perturbation_metadata={"mode": "none"},
        )
        original_path = logger.write_run_log("original", prop.property_id, original_log)

        perturbed_log = adapter.verify_property(
            perturbed_model,
            prop,
            pair_id=pair_id,
            model_role="perturbed",
            model_metadata=loaded_original.to_metadata(),
            perturbation_metadata={
                "mode": perturb_cfg.mode,
                "config": perturb_cfg.to_dict(),
                "realization_summary": delta_summary,
            },
        )
        perturbed_path = logger.write_run_log("perturbed", prop.property_id, perturbed_log)

        run_paths.append(
            {
                "property_id": prop.property_id,
                "original_log": str(original_path),
                "perturbed_log": str(perturbed_path),
            }
        )

    manifest = {
        "pair_id": pair_id,
        "created_at": dt.datetime.now().isoformat(),
        "study_type": "intrinsic_verification_similarity",
        "backend": verifier_cfg.backend,
        "verification_config": verifier_cfg.to_dict(),
        "dataset_config": dataset_cfg.to_dict(),
        "original_model": loaded_original.to_metadata(),
        "perturbation": {
            "spec": perturb_cfg.to_dict(),
            "event": perturbation_event,
            "realization_summary": delta_summary,
        },
        "properties": [prop.metadata() for prop in properties],
        "logs": run_paths,
    }
    manifest_path = logger.write_pair_manifest(manifest)
    return {"pair_id": pair_id, "manifest_path": str(manifest_path), "num_properties": len(properties)}


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_paired_analysis(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

