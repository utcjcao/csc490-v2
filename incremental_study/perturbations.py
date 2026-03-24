from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn

from .config import PerturbationConfig


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def _float_named_parameters(model: nn.Module):
    for name, param in model.named_parameters():
        if torch.is_floating_point(param):
            yield name, param


def apply_random_noise(model: nn.Module, cfg: PerturbationConfig) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.seed))
    layer_stats = []

    for name, param in _float_named_parameters(model):
        reference = param.detach()
        scale = cfg.random_std
        if cfg.random_relative:
            ref_scale = float(reference.std().item())
            if ref_scale == 0.0:
                ref_scale = max(float(reference.abs().mean().item()), 1.0)
            scale *= ref_scale
        noise = torch.randn(reference.shape, generator=generator, dtype=reference.dtype, device=reference.device) * scale
        param.data.add_(noise)
        layer_stats.append(
            {
                "layer": name,
                "mode": "random_noise",
                "noise_scale_used": float(scale),
                "noise_l2": float(noise.norm().item()),
                "noise_linf": float(noise.abs().max().item()),
            }
        )

    return {
        "mode": "random_noise",
        "config": cfg.to_dict(),
        "layer_events": layer_stats,
    }


def apply_symmetric_quantization(model: nn.Module, cfg: PerturbationConfig) -> dict[str, Any]:
    if cfg.quant_bits < 2:
        raise ValueError("quant_bits must be at least 2")
    qmax = (2 ** (cfg.quant_bits - 1)) - 1
    layer_stats = []

    for name, param in _float_named_parameters(model):
        tensor = param.detach()
        max_abs = float(tensor.abs().max().item())
        if max_abs == 0.0:
            scale = 1.0
            quantized = tensor.clone()
        else:
            scale = max_abs / qmax
            quantized = torch.round(tensor / scale).clamp(-qmax, qmax) * scale
        changed = (quantized != tensor).float().mean().item()
        param.data.copy_(quantized)
        layer_stats.append(
            {
                "layer": name,
                "mode": "quantize",
                "bits": int(cfg.quant_bits),
                "scale": float(scale),
                "changed_fraction": float(changed),
            }
        )

    return {
        "mode": "quantize",
        "config": cfg.to_dict(),
        "layer_events": layer_stats,
    }


def apply_magnitude_pruning(model: nn.Module, cfg: PerturbationConfig) -> dict[str, Any]:
    if not 0.0 <= cfg.prune_fraction < 1.0:
        raise ValueError("prune_fraction must be in [0, 1)")
    layer_stats = []

    for name, param in _float_named_parameters(model):
        tensor = param.detach()
        flat = tensor.abs().flatten()
        if flat.numel() == 0:
            continue
        k = int(math.floor(cfg.prune_fraction * flat.numel()))
        if k <= 0:
            threshold = 0.0
            pruned = tensor.clone()
        else:
            threshold = float(torch.kthvalue(flat, k).values.item())
            pruned = tensor.clone()
            pruned[pruned.abs() <= threshold] = 0
        zeroed_fraction = float((pruned == 0).float().mean().item())
        changed_fraction = float((pruned != tensor).float().mean().item())
        param.data.copy_(pruned)
        layer_stats.append(
            {
                "layer": name,
                "mode": "prune",
                "threshold": float(threshold),
                "zeroed_fraction": zeroed_fraction,
                "changed_fraction": changed_fraction,
            }
        )

    return {
        "mode": "prune",
        "config": cfg.to_dict(),
        "layer_events": layer_stats,
    }


def create_perturbed_model(model: nn.Module, cfg: PerturbationConfig) -> tuple[nn.Module, dict[str, Any]]:
    perturbed = clone_model(model)
    if cfg.mode == "random_noise":
        perturbation_event = apply_random_noise(perturbed, cfg)
    elif cfg.mode == "quantize":
        perturbation_event = apply_symmetric_quantization(perturbed, cfg)
    elif cfg.mode == "prune":
        perturbation_event = apply_magnitude_pruning(perturbed, cfg)
    else:
        raise ValueError(f"Unsupported perturbation mode: {cfg.mode}")
    perturbed.eval()
    return perturbed, perturbation_event


def summarize_model_delta(original: nn.Module, perturbed: nn.Module) -> dict[str, Any]:
    original_params = dict(_float_named_parameters(original))
    perturbed_params = dict(_float_named_parameters(perturbed))

    per_layer = []
    diff_sq_total = 0.0
    ref_sq_total = 0.0
    changed_total = 0
    zeroed_total = 0
    numel_total = 0

    for name, orig in original_params.items():
        if name not in perturbed_params:
            continue
        new = perturbed_params[name].detach()
        ref = orig.detach()
        delta = new - ref
        numel = int(ref.numel())
        diff_sq_total += float((delta.float() ** 2).sum().item())
        ref_sq_total += float((ref.float() ** 2).sum().item())
        changed = int((delta != 0).sum().item())
        zeroed = int(((new == 0) & (ref != 0)).sum().item())
        changed_total += changed
        zeroed_total += zeroed
        numel_total += numel
        per_layer.append(
            {
                "layer": name,
                "shape": list(ref.shape),
                "delta_l2": float(delta.norm().item()),
                "delta_linf": float(delta.abs().max().item()) if numel else 0.0,
                "changed_fraction": float(changed / numel) if numel else 0.0,
                "zeroed_fraction": float(zeroed / numel) if numel else 0.0,
                "orig_l2": float(ref.norm().item()),
                "new_l2": float(new.norm().item()),
            }
        )

    overall_rel_l2 = math.sqrt(diff_sq_total) / max(math.sqrt(ref_sq_total), 1e-12)
    return {
        "overall_relative_l2": float(overall_rel_l2),
        "changed_fraction": float(changed_total / numel_total) if numel_total else 0.0,
        "zeroed_fraction": float(zeroed_total / numel_total) if numel_total else 0.0,
        "layer_summaries": per_layer,
    }

