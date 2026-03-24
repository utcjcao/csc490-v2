"""Structured logging for artifact observation without proof transfer."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from enum import Enum

import torch

from nnverify.specs.out_spec import OutSpecType


def _enum_name(value):
    if isinstance(value, Enum):
        return value.name
    return value


def _tensor_to_list(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().flatten().tolist()
    if isinstance(value, (list, tuple)):
        items = []
        for entry in value:
            if isinstance(entry, torch.Tensor):
                items.extend(entry.detach().cpu().flatten().tolist())
            else:
                items.append(entry)
        return items
    return [value]


def _scalarize(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return [float(v) for v in value.flatten().tolist()]
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, tuple):
        return [_scalarize(v) for v in value]
    if isinstance(value, list):
        return [_scalarize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _scalarize(v) for k, v in value.items()}
    return str(value)


def _digest_bytes(raw_bytes):
    return hashlib.sha1(raw_bytes).hexdigest()[:12]


def _tensor_digest(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _digest_bytes(value.detach().cpu().numpy().tobytes())
    return _digest_bytes(torch.tensor(value).detach().cpu().numpy().tobytes())


def summarize_tensor(value, sample_values=8):
    if value is None:
        return None

    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)

    tensor = value.detach().cpu().flatten()
    if tensor.numel() == 0:
        return {"shape": list(value.shape), "numel": 0, "sample": []}

    sample = tensor[:sample_values].tolist()
    return {
        "shape": list(value.shape),
        "numel": int(tensor.numel()),
        "min": float(torch.min(tensor).item()),
        "max": float(torch.max(tensor).item()),
        "mean": float(torch.mean(tensor.float()).item()),
        "sample": [float(v) for v in sample],
    }


def normalize_unstable_relus(unstable_relus):
    if unstable_relus is None:
        return []

    normalized = []
    for layer_idx, layer in enumerate(unstable_relus):
        indices = []
        if isinstance(layer, tuple):
            if len(layer) > 0 and isinstance(layer[0], torch.Tensor):
                indices = [int(v) for v in layer[0].detach().cpu().flatten().tolist()]
        elif isinstance(layer, torch.Tensor):
            indices = [int(v) for v in layer.detach().cpu().flatten().tolist()]
        else:
            indices = [int(v) for v in _tensor_to_list(layer)]

        normalized.append({"layer_index": layer_idx, "unstable_relu_ids": indices, "count": len(indices)})
    return normalized


def summarize_bounds(transformer, sample_values=8):
    try:
        if hasattr(transformer, "get_all_bounds"):
            lower_bounds, upper_bounds = transformer.get_all_bounds()
        elif hasattr(transformer, "lower_bounds") and hasattr(transformer, "upper_bounds"):
            lower_bounds = transformer.lower_bounds
            upper_bounds = transformer.upper_bounds
        else:
            return []
    except Exception as exc:  # pragma: no cover - best-effort logging path
        return [{"error": f"failed to read bounds: {exc}"}]

    summaries = []
    for layer_idx, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
        summaries.append(
            {
                "layer_index": layer_idx,
                "lower": summarize_tensor(lower, sample_values=sample_values),
                "upper": summarize_tensor(upper, sample_values=sample_values),
            }
        )
    return summaries


def summarize_domain_artifacts(transformer, sample_values=8):
    artifacts = {}

    if hasattr(transformer, "last_compute_lb_stats"):
        artifacts["lp_stats"] = _scalarize(transformer.last_compute_lb_stats)

    if hasattr(transformer, "cofs") and hasattr(transformer, "centers"):
        artifacts["deepz"] = {
            "num_noise_symbols": int(transformer.cofs[-1].shape[0]) if len(transformer.cofs) > 0 else 0,
            "final_center": summarize_tensor(transformer.centers[-1], sample_values=sample_values)
            if len(transformer.centers) > 0 else None,
            "final_coefficient_summary": summarize_tensor(transformer.cofs[-1], sample_values=sample_values)
            if len(transformer.cofs) > 0 else None,
            "noise_index_sample": [
                {
                    "noise_index": int(noise_idx),
                    "relu": serialize_split_identifier(relu_identifier),
                }
                for noise_idx, relu_identifier in list(transformer.map_for_noise_indices.items())[:sample_values]
            ] if hasattr(transformer, "map_for_noise_indices") else [],
            "relu_layer_coefficient_summaries": [
                {
                    "layer_index": layer_idx,
                    "coefficients": summarize_tensor(cof, sample_values=sample_values),
                }
                for layer_idx, cof in enumerate(getattr(transformer, "relu_layer_cofs", [])[:sample_values])
            ],
        }

    if hasattr(transformer, "lcof") and hasattr(transformer, "ucof"):
        artifacts["deeppoly"] = {
            "final_lower_coefficients": summarize_tensor(transformer.lcof[-1], sample_values=sample_values)
            if len(transformer.lcof) > 0 else None,
            "final_upper_coefficients": summarize_tensor(transformer.ucof[-1], sample_values=sample_values)
            if len(transformer.ucof) > 0 else None,
            "final_lower_constants": summarize_tensor(transformer.lcst[-1], sample_values=sample_values)
            if len(transformer.lcst) > 0 else None,
            "final_upper_constants": summarize_tensor(transformer.ucst[-1], sample_values=sample_values)
            if len(transformer.ucst) > 0 else None,
            "optimize_lambda": bool(getattr(transformer, "optimize_lambda", False)),
        }

    if hasattr(transformer, "model") and hasattr(transformer, "method"):
        artifacts["lirpa"] = {
            "method": transformer.method,
        }

    return artifacts


def serialize_split_identifier(split_identifier):
    if split_identifier is None:
        return None
    if isinstance(split_identifier, tuple) and len(split_identifier) == 2:
        return {"kind": "relu", "layer": int(split_identifier[0]), "neuron": int(split_identifier[1])}
    return {"kind": "input", "index": int(split_identifier)}


def property_identifier(prop, property_index, clause_index):
    digest = hashlib.sha1()
    digest.update(prop.input_lb.detach().cpu().numpy().tobytes())
    digest.update(prop.input_ub.detach().cpu().numpy().tobytes())
    if prop.out_constr.constr_mat is not None:
        digest.update(prop.out_constr.constr_mat[0].detach().cpu().numpy().tobytes())
    return f"prop_{property_index:04d}_clause_{clause_index:02d}_{digest.hexdigest()[:12]}"


def property_metadata(prop, property_index, clause_index):
    metadata = {
        "property_id": property_identifier(prop, property_index, clause_index),
        "property_index": property_index,
        "clause_index": clause_index,
        "dataset": _enum_name(prop.dataset),
        "input_size": int(prop.get_input_size()),
        "input_lower_bound_digest": hashlib.sha1(prop.input_lb.detach().cpu().numpy().tobytes()).hexdigest()[:12],
        "input_upper_bound_digest": hashlib.sha1(prop.input_ub.detach().cpu().numpy().tobytes()).hexdigest()[:12],
        "constraint_type": _enum_name(prop.out_constr.constr_type),
        "is_conjunctive": bool(prop.is_conjunctive()),
    }
    if prop.is_local_robustness():
        metadata["label"] = int(prop.get_label())
    if prop.out_constr.constr_type == OutSpecType.LOCAL_ROBUST and prop.out_constr.constr_mat is not None:
        metadata["constraint_shape"] = list(prop.out_constr.constr_mat[0].shape)
    return metadata


class ArtifactRunLogger:
    """Writes one JSON artifact log per independently verified property."""

    schema_version = 1

    def __init__(
        self,
        output_dir,
        pair_id,
        model_role,
        original_model_path,
        perturbation,
        model_path=None,
        session_id=None,
        run_id=None,
        max_logged_split_steps=50,
        top_k_candidates=5,
        sample_values=8,
        summary_prefix=20,
    ):
        self.output_dir = output_dir
        self.pair_id = pair_id
        self.model_role = model_role
        self.original_model_path = original_model_path
        self.model_path = model_path or original_model_path
        self.perturbation = perturbation
        self.session_id = session_id or uuid.uuid4().hex
        self.run_id = run_id or uuid.uuid4().hex
        self.max_logged_split_steps = max_logged_split_steps
        self.top_k_candidates = top_k_candidates
        self.sample_values = sample_values
        self.summary_prefix = summary_prefix
        self._current = None

    def _base_metadata(self):
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "pair_id": self.pair_id,
            "model_role": self.model_role,
            "original_model_path": self.original_model_path,
            "model_path": self.model_path,
            "perturbation": _scalarize(self.perturbation),
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def start_property(self, prop, property_index, clause_index, args):
        prop_meta = property_metadata(prop, property_index, clause_index)
        self._current = {
            "schema_version": self.schema_version,
            "metadata": {
                **self._base_metadata(),
                **prop_meta,
                "domain": _enum_name(args.domain),
                "split": _enum_name(args.split),
                "epsilon": float(args.eps),
                "count": args.count,
                "timeout": args.timeout,
                "pt_method": _scalarize(args.pt_method),
                "artifact_observation_only": True,
            },
            "root_artifacts": {},
            "split_trace": [],
            "search_summary": {},
            "result": {},
        }
        return prop_meta["property_id"]

    def set_center_input_behavior(self, center_input, output, predicted_label, true_label=None, constraint_margin=None,
                                  constraint_margin_summary=None, center_source=None, top2_margin=None,
                                  true_label_margin=None):
        if self._current is None:
            return
        root_artifacts = self._current.setdefault("root_artifacts", {})
        root_artifacts["center_input_behavior"] = {
            "center_source": center_source,
            "center_input_digest": _tensor_digest(center_input),
            "output_summary": summarize_tensor(output, sample_values=self.sample_values),
            "predicted_label": None if predicted_label is None else int(predicted_label),
            "true_label": None if true_label is None else int(true_label),
            "prediction_matches_true_label": None if true_label is None or predicted_label is None else bool(predicted_label == true_label),
            "top2_margin": None if top2_margin is None else float(top2_margin),
            "true_label_margin": None if true_label_margin is None else float(true_label_margin),
            "constraint_margin": _scalarize(constraint_margin),
            "constraint_margin_summary": _scalarize(constraint_margin_summary),
        }

    def set_root_artifacts(self, transformer, unstable_relus, root_lower_bound, root_upper_bound, root_candidates):
        if self._current is None:
            return

        normalized_relus = normalize_unstable_relus(unstable_relus)
        root_artifacts = self._current.setdefault("root_artifacts", {})
        root_artifacts.update({
            "unstable_relus_per_layer": normalized_relus,
            "total_unstable_relus": int(sum(layer["count"] for layer in normalized_relus)),
            "root_lower_bound": _scalarize(root_lower_bound),
            "root_upper_bound": _scalarize(root_upper_bound),
            "bounds_per_layer": summarize_bounds(transformer, sample_values=self.sample_values),
            "top_k_branch_candidates": _scalarize(root_candidates[: self.top_k_candidates]),
            "domain_artifacts": summarize_domain_artifacts(transformer, sample_values=self.sample_values),
        })

    def append_search_event(self, event):
        if self._current is None:
            return
        if len(self._current["split_trace"]) >= self.max_logged_split_steps:
            return
        self._current["split_trace"].append(_scalarize(event))

    def set_search_summary(self, summary):
        if self._current is None:
            return
        self._current["search_summary"] = _scalarize(summary)

    def finalize_property(self, status, runtime_sec, tree_size, leaf_count=None, extra_result=None):
        if self._current is None:
            return None

        result = {
            "verification_status": _scalarize(status),
            "total_runtime_sec": float(runtime_sec),
            "tree_size": int(tree_size),
            "leaf_count": None if leaf_count is None else int(leaf_count),
        }
        if extra_result is not None:
            result.update(_scalarize(extra_result))
        self._current["result"] = result

        property_id = self._current["metadata"]["property_id"]
        out_dir = os.path.join(self.output_dir, self.pair_id, self.model_role)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{property_id}.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(self._current, handle, indent=2, sort_keys=True)

        self._current = None
        return out_path
