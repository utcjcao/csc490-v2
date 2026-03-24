from __future__ import annotations

import math
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import VerificationConfig
from .property_loader import RobustnessProperty


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTOLIRPA_ROOT = REPO_ROOT / "incremental_study" / "vendor"


def _ensure_autolirpa_importable() -> None:
    auto_lirpa_root = str(AUTOLIRPA_ROOT)
    if auto_lirpa_root not in sys.path:
        sys.path.insert(0, auto_lirpa_root)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().reshape(-1)[0].item())
    return float(value)


def _tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None or not torch.is_tensor(tensor):
        return None
    flat = tensor.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"shape": list(tensor.shape), "numel": 0}
    return {
        "shape": list(tensor.shape),
        "numel": int(flat.numel()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        "abs_max": float(flat.abs().max().item()),
    }


def _unravel_index(index: int, shape: Iterable[int]) -> list[int]:
    dims = list(shape)
    if not dims:
        return []
    coords = [0 for _ in dims]
    remainder = int(index)
    for dim_idx in range(len(dims) - 1, -1, -1):
        size = int(dims[dim_idx])
        coords[dim_idx] = remainder % size
        remainder //= size
    return coords


def _build_margin_spec(label: int, num_classes: int, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    negatives = [cls for cls in range(num_classes) if cls != label]
    C = torch.zeros((1, len(negatives), num_classes), device=device)
    C[0, :, label] = 1.0
    for row, neg in enumerate(negatives):
        C[0, row, neg] = -1.0
    return C, negatives


def _summarize_center_behavior(logits: torch.Tensor, true_label: int, negative_classes: list[int]) -> dict[str, Any]:
    probs = logits.detach().reshape(-1)
    pred = int(probs.argmax().item())
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)
    top2_gap = float(sorted_logits[0].item() - sorted_logits[1].item()) if probs.numel() > 1 else 0.0
    true_logit = float(probs[true_label].item())
    margins = [true_logit - float(probs[neg].item()) for neg in negative_classes]
    return {
        "predicted_label": pred,
        "true_label": int(true_label),
        "true_label_match": bool(pred == true_label),
        "top2_gap": top2_gap,
        "true_label_logit": true_logit,
        "constraint_margin_min": float(min(margins)) if margins else None,
        "constraint_margin_max": float(max(margins)) if margins else None,
        "constraint_margin_mean": float(sum(margins) / len(margins)) if margins else None,
        "logit_summary": _tensor_summary(probs),
        "top_classes": [
            {"class": int(sorted_indices[i].item()), "logit": float(sorted_logits[i].item())}
            for i in range(min(5, probs.numel()))
        ],
    }


def _top_relu_candidates(node: Any, pre_lower: torch.Tensor, pre_upper: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    unstable = node.get_unstable_mask(pre_lower, pre_upper)
    heuristics = node.compute_bound_improvement_heuristics(pre_lower, pre_upper)
    unstable_flat = unstable.reshape(-1)
    heuristic_flat = heuristics.reshape(-1)
    unstable_indices = torch.nonzero(unstable_flat, as_tuple=False).reshape(-1)
    if unstable_indices.numel() == 0:
        return []
    selected_scores = heuristic_flat[unstable_indices]
    k = min(int(top_k), int(selected_scores.numel()))
    top_scores, order = torch.topk(selected_scores, k=k)

    result = []
    shape = list(pre_lower.shape[1:] if pre_lower.ndim > 1 else pre_lower.shape)
    for rank, (score, chosen_idx) in enumerate(zip(top_scores, order), start=1):
        flat_index = int(unstable_indices[chosen_idx].item())
        result.append(
            {
                "layer": getattr(node, "name", "<unnamed>"),
                "rank": rank,
                "score": float(score.item()),
                "flat_index": flat_index,
                "index": _unravel_index(flat_index, shape),
            }
        )
    return result


def _relu_layer_summary(node: Any, top_k: int) -> dict[str, Any] | None:
    if not getattr(node, "inputs", None):
        return None
    preact = node.inputs[0]
    lower = getattr(preact, "lower", None)
    upper = getattr(preact, "upper", None)
    if lower is None or upper is None or not torch.is_tensor(lower) or not torch.is_tensor(upper):
        return None
    unstable = node.get_unstable_mask(lower, upper)
    heuristics = node.compute_bound_improvement_heuristics(lower, upper)
    return {
        "layer": getattr(node, "name", "<unnamed>"),
        "pre_activation_lower_summary": _tensor_summary(lower),
        "pre_activation_upper_summary": _tensor_summary(upper),
        "unstable_count": int(unstable.sum().item()),
        "candidate_score_summary": _tensor_summary(heuristics),
        "top_candidates": _top_relu_candidates(node, lower, upper, top_k=top_k),
    }


class LirpaVerificationAdapter:
    """Thin adapter around auto_LiRPA for root-stage robustness verification."""

    def __init__(self, cfg: VerificationConfig) -> None:
        self.cfg = cfg

    def verify_property(
        self,
        model: torch.nn.Module,
        prop: RobustnessProperty,
        *,
        pair_id: str,
        model_role: str,
        model_metadata: dict[str, Any],
        perturbation_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        start = time.perf_counter()

        try:
            _ensure_autolirpa_importable()
            from auto_LiRPA import BoundedModule, BoundedTensor  # type: ignore
            from auto_LiRPA.perturbations import PerturbationLpNorm  # type: ignore
            from auto_LiRPA.operators.relu import BoundRelu  # type: ignore

            device = torch.device(self.cfg.device)
            model = model.to(device)
            model.eval()

            x = prop.x.unsqueeze(0).to(device)
            with torch.no_grad():
                center_logits = model(x)
            num_classes = int(center_logits.shape[-1])
            C, negative_classes = _build_margin_spec(prop.label, num_classes, device=device)
            center_behavior = _summarize_center_behavior(center_logits, prop.label, negative_classes)

            x_l = torch.clamp(x - prop.eps, min=self.cfg.input_lower, max=self.cfg.input_upper)
            x_u = torch.clamp(x + prop.eps, min=self.cfg.input_lower, max=self.cfg.input_upper)
            ptb = PerturbationLpNorm(norm=float("inf"), x_L=x_l, x_U=x_u)
            bounded_x = BoundedTensor(x, ptb)

            bound_opts = {}
            if self.cfg.conv_mode:
                bound_opts["conv_mode"] = self.cfg.conv_mode

            bounded_model = BoundedModule(model, x, device=device, bound_opts=bound_opts)
            lb, ub = bounded_model.compute_bounds(
                x=(bounded_x,),
                C=C,
                method=self.cfg.method,
                bound_upper=True,
            )

            relu_summaries = []
            all_candidates = []
            layer_summaries = []
            total_unstable = 0

            for node in bounded_model.nodes():
                lower = getattr(node, "lower", None)
                upper = getattr(node, "upper", None)
                if torch.is_tensor(lower) or torch.is_tensor(upper):
                    layer_summaries.append(
                        {
                            "layer": getattr(node, "name", "<unnamed>"),
                            "type": type(node).__name__,
                            "lower_summary": _tensor_summary(lower if torch.is_tensor(lower) else None),
                            "upper_summary": _tensor_summary(upper if torch.is_tensor(upper) else None),
                        }
                    )

                if isinstance(node, BoundRelu):
                    relu_summary = _relu_layer_summary(node, top_k=self.cfg.top_k)
                    if relu_summary is not None:
                        relu_summaries.append(relu_summary)
                        total_unstable += int(relu_summary["unstable_count"])
                        all_candidates.extend(relu_summary["top_candidates"])

            all_candidates.sort(key=lambda item: item["score"], reverse=True)
            top_root_candidates = all_candidates[: self.cfg.top_k]
            for global_rank, candidate in enumerate(top_root_candidates, start=1):
                candidate["global_rank"] = global_rank

            if not center_behavior["true_label_match"]:
                status = "MISS_CLASSIFIED"
            elif bool((lb > 0).all().item()):
                status = "VERIFIED"
            else:
                status = "UNKNOWN"

            runtime = time.perf_counter() - start
            return {
                "metadata": {
                    "run_id": run_id,
                    "pair_id": pair_id,
                    "model_role": model_role,
                    "backend": self.cfg.backend,
                    "verification_method": self.cfg.method,
                    "device": self.cfg.device,
                    "timeout_sec": self.cfg.timeout_sec,
                    "epsilon": prop.eps,
                    "property_id": prop.property_id,
                    "property_index": prop.property_index,
                    "dataset_index": prop.dataset_index,
                    "label": prop.label,
                    "split_heuristic": None,
                    "max_logged_split_steps": self.cfg.max_logged_split_steps,
                    "model": model_metadata,
                    "perturbation": perturbation_metadata,
                },
                "center_behavior": center_behavior,
                "root_artifacts": {
                    "margin_lower_bounds_summary": _tensor_summary(lb),
                    "margin_upper_bounds_summary": _tensor_summary(ub),
                    "unstable_relu_count_per_layer": [
                        {"layer": entry["layer"], "unstable_count": entry["unstable_count"]}
                        for entry in relu_summaries
                    ],
                    "total_unstable_relus": total_unstable,
                    "layer_bound_summaries": layer_summaries,
                    "relu_layer_summaries": relu_summaries,
                    "root_branch_candidates": top_root_candidates,
                },
                "split_trace": {
                    "available": False,
                    "reason": "Plain auto_LiRPA root-stage verification does not expose an independent branch-and-bound trace.",
                    "events": [],
                },
                "search_summary": {
                    "available": False,
                    "backend_has_branch_and_bound": False,
                    "total_nodes_visited": None,
                    "total_pruned_nodes": None,
                    "max_depth_reached": None,
                    "first_chosen_splits": [],
                    "first_node_lower_bounds": [],
                    "final_tree_size": None,
                    "leaf_count": None,
                },
                "result": {
                    "status": status,
                    "runtime_sec": float(runtime),
                    "margin_lower_bounds": [float(v) for v in lb.detach().reshape(-1).cpu().tolist()],
                    "margin_upper_bounds": [float(v) for v in ub.detach().reshape(-1).cpu().tolist()],
                },
            }
        except Exception as exc:
            runtime = time.perf_counter() - start
            return {
                "metadata": {
                    "run_id": run_id,
                    "pair_id": pair_id,
                    "model_role": model_role,
                    "backend": self.cfg.backend,
                    "verification_method": self.cfg.method,
                    "device": self.cfg.device,
                    "timeout_sec": self.cfg.timeout_sec,
                    "epsilon": prop.eps,
                    "property_id": prop.property_id,
                    "property_index": prop.property_index,
                    "dataset_index": prop.dataset_index,
                    "label": prop.label,
                    "split_heuristic": None,
                    "max_logged_split_steps": self.cfg.max_logged_split_steps,
                    "model": model_metadata,
                    "perturbation": perturbation_metadata,
                },
                "center_behavior": None,
                "root_artifacts": None,
                "split_trace": {"available": False, "events": []},
                "search_summary": {"available": False},
                "result": {
                    "status": "ERROR",
                    "runtime_sec": float(runtime),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                },
            }
