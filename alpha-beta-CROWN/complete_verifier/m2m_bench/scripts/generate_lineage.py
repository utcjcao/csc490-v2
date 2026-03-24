#!/usr/bin/env python3
"""Generate a synthetic m -> m' checkpoint lineage for benchmark V1."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
COMPLETE_VERIFIER_ROOT = SCRIPT_DIR.parents[1]
if str(COMPLETE_VERIFIER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPLETE_VERIFIER_ROOT))

import torch
from torchvision import datasets, transforms

from model_defs import resnet2b, resnet4b
from m2m_bench.common import DEFAULT_BASE_CHECKPOINT, ensure_dir, write_json


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
DRIFT_LABELS = ["tiny", "small", "moderate", "larger"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic checkpoint lineage for m -> m' benchmarking.")
    parser.add_argument(
        "--base-checkpoint",
        default=str(DEFAULT_BASE_CHECKPOINT),
        help="Base checkpoint path for the root model.",
    )
    parser.add_argument("--model-name", default="resnet2b", choices=["resnet2b", "resnet4b"])
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for checkpoints and lineage.json.",
    )
    parser.add_argument(
        "--lineage-id",
        default="",
        help="Optional lineage identifier. Defaults to <model>_synthetic_seed<seed>.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic drift.")
    parser.add_argument(
        "--drift-targets",
        default="1e-4,5e-4,1e-3,5e-3",
        help="Comma-separated target relative L2 drift buckets from the root checkpoint.",
    )
    parser.add_argument("--max-attempts-per-target", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument(
        "--dataset-root",
        default=str(COMPLETE_VERIFIER_ROOT / "datasets"),
        help="Path for CIFAR-10 used in functional drift evaluation.",
    )
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--skip-functional-metrics", action="store_true")
    parser.add_argument("--device", default="cpu", help="Device used for functional drift evaluation.")
    return parser.parse_args()


def model_factory(model_name: str):
    return {"resnet2b": resnet2b, "resnet4b": resnet4b}[model_name]


def parse_targets(raw: str) -> list[float]:
    targets = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise ValueError("Drift targets must be positive.")
        targets.append(value)
    if not targets:
        raise ValueError("At least one drift target is required.")
    if sorted(targets) != targets:
        raise ValueError("Drift targets must be in nondecreasing order.")
    return targets


def load_checkpoint_payload(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if isinstance(payload, list):
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("Unsupported checkpoint format; expected state_dict-like mapping.")
    return {name: tensor.detach().clone().cpu() for name, tensor in payload.items()}


def replace_state_dict(payload: Any, new_state_dict: dict[str, torch.Tensor]) -> Any:
    if isinstance(payload, dict) and "state_dict" in payload:
        updated = copy.deepcopy(payload)
        updated["state_dict"] = new_state_dict
        return updated
    if isinstance(payload, list):
        updated = copy.deepcopy(payload)
        updated[0] = new_state_dict
        return updated
    if isinstance(payload, dict):
        return new_state_dict
    raise ValueError("Unsupported checkpoint format; cannot write new state_dict.")


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in state_dict.items()}


def floating_items(state_dict: dict[str, torch.Tensor]):
    for name, tensor in state_dict.items():
        if tensor.is_floating_point():
            yield name, tensor


def state_sq_norm(state_dict: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for _, tensor in floating_items(state_dict):
        total += float(torch.sum(tensor.double() * tensor.double()).item())
    return total


def state_abs_max(state_dict: dict[str, torch.Tensor]) -> float:
    max_value = 0.0
    for _, tensor in floating_items(state_dict):
        if tensor.numel():
            max_value = max(max_value, float(tensor.abs().max().item()))
    return max_value


def state_rel_metrics(
    reference_state: dict[str, torch.Tensor],
    other_state: dict[str, torch.Tensor],
) -> tuple[float, float]:
    sq = 0.0
    linf = 0.0
    ref_sq = state_sq_norm(reference_state)
    ref_max = max(state_abs_max(reference_state), 1e-12)
    for name, ref_tensor in floating_items(reference_state):
        diff = other_state[name].double() - ref_tensor.double()
        sq += float(torch.sum(diff * diff).item())
        if diff.numel():
            linf = max(linf, float(diff.abs().max().item()))
    return math.sqrt(sq / max(ref_sq, 1e-18)), linf / ref_max


def sample_noise_like(
    state_dict: dict[str, torch.Tensor],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if tensor.is_floating_point():
            noise[name] = torch.randn(
                tensor.shape,
                generator=generator,
                dtype=torch.float64,
            )
        else:
            noise[name] = torch.zeros(tensor.shape, dtype=torch.float64)
    return noise


def solve_sigma_for_target(
    root_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    noise_state: dict[str, torch.Tensor],
    target_rel_l2: float,
) -> float | None:
    ref_sq = state_sq_norm(root_state)
    target_sq = (target_rel_l2 ** 2) * ref_sq
    current_sq = 0.0
    dot = 0.0
    noise_sq = 0.0
    for name, root_tensor in floating_items(root_state):
        current_delta = current_state[name].double() - root_tensor.double()
        noise_tensor = noise_state[name]
        current_sq += float(torch.sum(current_delta * current_delta).item())
        dot += float(torch.sum(current_delta * noise_tensor).item())
        noise_sq += float(torch.sum(noise_tensor * noise_tensor).item())
    if noise_sq <= 0:
        return None
    a = current_sq - target_sq
    b = 2.0 * dot
    c = noise_sq
    discriminant = b * b - 4.0 * c * a
    if discriminant < 0:
        return None
    root_disc = math.sqrt(discriminant)
    roots = [
        (-b - root_disc) / (2.0 * c),
        (-b + root_disc) / (2.0 * c),
    ]
    positive_roots = [value for value in roots if value > 0]
    if not positive_roots:
        return None
    return min(positive_roots)


def apply_noise(
    parent_state: dict[str, torch.Tensor],
    noise_state: dict[str, torch.Tensor],
    sigma: float,
) -> dict[str, torch.Tensor]:
    child_state: dict[str, torch.Tensor] = {}
    for name, tensor in parent_state.items():
        if tensor.is_floating_point():
            child_state[name] = (tensor.double() + sigma * noise_state[name]).to(dtype=tensor.dtype)
        else:
            child_state[name] = tensor.clone()
    return child_state


def load_eval_batch(
    dataset_root: Path,
    *,
    num_samples: int,
    seed: int,
    download: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)]
    )
    dataset = datasets.CIFAR10(str(dataset_root), train=False, download=download, transform=transform)
    count = min(num_samples, len(dataset))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    images = []
    labels = []
    for idx in indices:
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def load_model(model_name: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = model_factory(model_name)()
    payload = load_checkpoint_payload(checkpoint_path)
    state_dict = extract_state_dict(payload)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


@torch.no_grad()
def collect_logits(model_name: str, checkpoint_path: Path, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    model = load_model(model_name, checkpoint_path, device)
    return model(images.to(device)).cpu()


def drift_label(index: int) -> str:
    if index < len(DRIFT_LABELS):
        return DRIFT_LABELS[index]
    return f"bucket_{index + 1}"


def main() -> int:
    args = parse_args()
    drift_targets = parse_targets(args.drift_targets)
    base_checkpoint = Path(args.base_checkpoint).resolve()
    if not base_checkpoint.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")

    lineage_id = args.lineage_id or f"{args.model_name}_synthetic_seed{args.seed}"
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")

    base_payload = load_checkpoint_payload(base_checkpoint)
    root_state = extract_state_dict(base_payload)
    current_state = clone_state_dict(root_state)
    current_model_id = "m0"

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested for functional drift evaluation but is unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    eval_images = eval_labels = None
    parent_logits = None
    if not args.skip_functional_metrics:
        eval_images, eval_labels = load_eval_batch(
            Path(args.dataset_root).resolve(),
            num_samples=args.eval_samples,
            seed=args.seed,
            download=args.download_dataset,
        )
        del eval_labels
        parent_logits = collect_logits(args.model_name, base_checkpoint, eval_images, device)

    models: list[dict[str, Any]] = [
        {
            "model_id": current_model_id,
            "parent_id": None,
            "checkpoint_path": str(base_checkpoint),
            "lineage_id": lineage_id,
            "generation_mode": "synthetic_noise",
            "seed": args.seed,
            "drift_bucket": "root",
            "target_drift_rel_l2": 0.0,
            "delta_w_rel_l2": 0.0,
            "delta_w_rel_linf": 0.0,
            "delta_w_rel_l2_root": 0.0,
            "delta_w_rel_linf_root": 0.0,
            "delta_w_rel_l2_parent": 0.0,
            "delta_w_rel_linf_parent": 0.0,
            "label_agreement_parent": None,
            "logit_mse_parent": None,
        }
    ]

    for idx, target_rel_l2 in enumerate(drift_targets):
        sigma = None
        noise_state = None
        for attempt in range(args.max_attempts_per_target):
            noise_state = sample_noise_like(current_state, seed=args.seed + (idx + 1) * 1000 + attempt)
            sigma = solve_sigma_for_target(root_state, current_state, noise_state, target_rel_l2)
            if sigma is not None:
                break
        if sigma is None or noise_state is None:
            raise RuntimeError(f"Failed to synthesize drift bucket {target_rel_l2} after {args.max_attempts_per_target} attempts.")

        parent_state = current_state
        child_state = apply_noise(parent_state, noise_state, sigma)
        child_model_id = f"m{idx + 1}"
        child_path = checkpoints_dir / f"{child_model_id}.pth"
        child_payload = replace_state_dict(base_payload, clone_state_dict(child_state))
        torch.save(child_payload, child_path)

        rel_l2_root, rel_linf_root = state_rel_metrics(root_state, child_state)
        rel_l2_parent, rel_linf_parent = state_rel_metrics(parent_state, child_state)
        label_agreement = None
        logit_mse = None
        if parent_logits is not None and eval_images is not None:
            child_logits = collect_logits(args.model_name, child_path, eval_images, device)
            parent_labels = parent_logits.argmax(dim=1)
            child_labels = child_logits.argmax(dim=1)
            label_agreement = float((parent_labels == child_labels).float().mean().item())
            logit_mse = float(torch.mean((parent_logits - child_logits) ** 2).item())
            parent_logits = child_logits

        models.append(
            {
                "model_id": child_model_id,
                "parent_id": current_model_id,
                "checkpoint_path": str(child_path),
                "lineage_id": lineage_id,
                "generation_mode": "synthetic_noise",
                "seed": args.seed,
                "drift_bucket": drift_label(idx),
                "target_drift_rel_l2": target_rel_l2,
                "delta_w_rel_l2": rel_l2_root,
                "delta_w_rel_linf": rel_linf_root,
                "delta_w_rel_l2_root": rel_l2_root,
                "delta_w_rel_linf_root": rel_linf_root,
                "delta_w_rel_l2_parent": rel_l2_parent,
                "delta_w_rel_linf_parent": rel_linf_parent,
                "label_agreement_parent": label_agreement,
                "logit_mse_parent": logit_mse,
            }
        )
        current_state = child_state
        current_model_id = child_model_id

    manifest = {
        "schema_version": "m2m-lineage-v1",
        "created_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "lineage_id": lineage_id,
        "model_name": args.model_name,
        "generation_mode": "synthetic_noise",
        "base_checkpoint": str(base_checkpoint),
        "drift_targets_rel_l2": drift_targets,
        "functional_eval": {
            "enabled": not args.skip_functional_metrics,
            "dataset": "cifar10_test",
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "num_samples": args.eval_samples,
            "seed": args.seed,
        },
        "models": models,
    }
    write_json(manifest, output_dir / "lineage.json")
    print(f"Wrote lineage manifest to {(output_dir / 'lineage.json').resolve()}")
    print(f"Generated {len(models) - 1} child checkpoints in {checkpoints_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
