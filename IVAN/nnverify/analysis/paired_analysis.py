"""Independent M vs M' artifact observation without proof transfer."""

from __future__ import annotations

import argparse
import copy
import json
import os
import uuid

import nnverify.specs.spec as specs
import nnverify.util as util
from nnverify import config
from nnverify.analysis.artifact_logger import ArtifactRunLogger, property_metadata
from nnverify.analyzer import Analyzer
from nnverify.bnb import Split
from nnverify.common import Domain
from nnverify.common.dataset import Dataset
from nnverify.common.network import LayerType
from nnverify.proof_transfer.approximate import Finetune, Prune, QuantizationType, Quantize, Random
from nnverify.training.training_args import TrainArgs


DATASET_MAP = {
    "mnist": Dataset.MNIST,
    "cifar10": Dataset.CIFAR10,
    "acas": Dataset.ACAS,
    "oval_cifar": Dataset.OVAL_CIFAR,
}

DOMAIN_MAP = {
    "box": Domain.BOX,
    "deepz": Domain.DEEPZ,
    "deeppoly": Domain.DEEPPOLY,
    "lp": Domain.LP,
    "lirpa_ibp": Domain.LIRPA_IBP,
    "lirpa_crown": Domain.LIRPA_CROWN,
    "lirpa_crown_ibp": Domain.LIRPA_CROWN_IBP,
    "lirpa_crown_opt": Domain.LIRPA_CROWN_OPT,
}

SPLIT_MAP = {
    "none": None,
    "relu_rand": Split.RELU_RAND,
    "relu_grad": Split.RELU_GRAD,
    "relu_esip_score": Split.RELU_ESIP_SCORE,
    "relu_esip_score2": Split.RELU_ESIP_SCORE2,
    "relu_kfsb": Split.RELU_KFSB,
    "input": Split.INPUT,
    "input_grad": Split.INPUT_GRAD,
    "input_sb": Split.INPUT_SB,
}

SPEC_TYPE_MAP = {
    "linf": specs.InputSpecType.LINF,
    "patch": specs.InputSpecType.PATCH,
    "global": specs.InputSpecType.GLOBAL,
}


def resolve_net_path(net):
    if os.path.exists(net):
        return net
    candidate = os.path.join(config.NET_HOME, net)
    if os.path.exists(candidate):
        return candidate
    return net


def build_verification_args(net_path, domain, dataset, eps, count, split, timeout, spec_type):
    args = config.Args(
        net="",
        domain=domain,
        dataset=dataset,
        eps=eps,
        count=count,
        split=split,
        pt_method=None,
        timeout=timeout,
        spec_type=spec_type,
    )
    args.net = net_path
    return args


def parse_key_value_pairs(arg):
    params = {}
    if not arg:
        return params
    for chunk in arg.split(","):
        if not chunk:
            continue
        if "=" not in chunk:
            params["_value"] = chunk.strip()
            continue
        key, value = chunk.split("=", 1)
        params[key.strip()] = value.strip()
    return params


def parse_approximation(spec):
    if ":" in spec:
        mode, raw_params = spec.split(":", 1)
    else:
        mode, raw_params = spec, ""
    mode = mode.strip().lower()
    params = parse_key_value_pairs(raw_params)

    if mode == "quantize":
        quant_type = params.get("type", params.get("_value", raw_params.strip())).lower()
        quant_map = {
            "int8": QuantizationType.INT8,
            "int16": QuantizationType.INT16,
            "int32": QuantizationType.INT32,
            "fp16": QuantizationType.FP16,
        }
        if quant_type not in quant_map:
            raise ValueError(f"Unsupported quantization type: {quant_type}")
        approximation = Quantize(quant_map[quant_type])
        return {"type": "quantize", "params": {"type": quant_type}, "approximation": approximation}

    if mode == "prune":
        percent = float(params.get("percent", params.get("_value", raw_params.strip())))
        approximation = Prune(percent)
        return {"type": "prune", "params": {"percent": percent}, "approximation": approximation}

    if mode == "random":
        scale = float(params.get("scale", params.get("_value", raw_params.strip())))
        approximation = Random(scale)
        return {"type": "random", "params": {"scale": scale}, "approximation": approximation}

    if mode == "finetune":
        epochs = int(params.get("epochs", 20))
        lr = float(params.get("lr", 0.001))
        batch_size = int(params.get("batch_size", 100))
        train_args = TrainArgs(epochs=epochs, lr=lr, batch_size=batch_size)
        approximation = Finetune(train_args=train_args)
        return {
            "type": "finetune",
            "params": {"epochs": epochs, "lr": lr, "batch_size": batch_size},
            "approximation": approximation,
        }

    raise ValueError(f"Unsupported approximation mode: {mode}")


def build_pair_manifest(pair_id, session_id, output_dir, original_path, perturbation, args, properties, results):
    manifest = {
        "pair_id": pair_id,
        "session_id": session_id,
        "artifact_observation_only": True,
        "original_model_path": original_path,
        "perturbation": perturbation,
        "verification_config": {
            "domain": args.domain.name,
            "dataset": args.dataset.name,
            "epsilon": args.eps,
            "count": args.count,
            "split": None if args.split is None else args.split.name,
            "timeout": args.timeout,
            "spec_type": args.spec_type.name,
            "pt_method": None,
        },
        "properties": properties,
        "runs": results,
    }
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, pair_id, "pair_manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path


def summarize_parameter_delta(original_tensor, perturbed_tensor):
    original = original_tensor.detach().cpu().float().flatten()
    perturbed = perturbed_tensor.detach().cpu().float().flatten()
    delta = perturbed - original
    denom = torch.norm(original, p=2).item()
    changed_count = int(torch.count_nonzero(delta != 0).item())
    return {
        "num_parameters": int(delta.numel()),
        "changed_count": changed_count,
        "mean_abs_delta": float(torch.mean(torch.abs(delta)).item()),
        "max_abs_delta": float(torch.max(torch.abs(delta)).item()),
        "l2_delta": float(torch.norm(delta, p=2).item()),
        "relative_l2_delta": float(torch.norm(delta, p=2).item() / (denom + 1e-12)),
        "changed_fraction": float(changed_count / max(delta.numel(), 1)),
    }


def compare_networks(original_net, perturbed_net):
    layer_summaries = []
    total_params = 0
    total_changed = 0
    total_l2_delta_sq = 0.0
    total_base_l2_sq = 0.0

    for layer_index, (original_layer, perturbed_layer) in enumerate(zip(original_net, perturbed_net)):
        if original_layer.type != perturbed_layer.type:
            continue
        if original_layer.type not in [LayerType.Linear, LayerType.Conv2D]:
            continue

        layer_summary = {
            "layer_index": layer_index,
            "layer_type": original_layer.type.name,
        }

        if getattr(original_layer, "weight", None) is not None and getattr(perturbed_layer, "weight", None) is not None:
            weight_delta = summarize_parameter_delta(original_layer.weight, perturbed_layer.weight)
            layer_summary["weight_delta"] = weight_delta
            total_params += weight_delta["num_parameters"]
            total_changed += weight_delta["changed_count"]
            total_l2_delta_sq += weight_delta["l2_delta"] ** 2
            total_base_l2_sq += float(torch.norm(original_layer.weight.detach().cpu().float().flatten(), p=2).item() ** 2)

        if getattr(original_layer, "bias", None) is not None and getattr(perturbed_layer, "bias", None) is not None:
            bias_delta = summarize_parameter_delta(original_layer.bias, perturbed_layer.bias)
            layer_summary["bias_delta"] = bias_delta
            total_params += bias_delta["num_parameters"]
            total_changed += bias_delta["changed_count"]
            total_l2_delta_sq += bias_delta["l2_delta"] ** 2
            total_base_l2_sq += float(torch.norm(original_layer.bias.detach().cpu().float().flatten(), p=2).item() ** 2)

        layer_summaries.append(layer_summary)

    return {
        "layer_deltas": layer_summaries,
        "overall": {
            "total_parameters": total_params,
            "changed_fraction": float(total_changed / max(total_params, 1)),
            "l2_delta": float(total_l2_delta_sq ** 0.5),
            "relative_l2_delta": float((total_l2_delta_sq ** 0.5) / ((total_base_l2_sq ** 0.5) + 1e-12)),
        },
    }


def run_independent_pair_analysis(
    net,
    dataset,
    domain,
    split,
    approximation_spec,
    output_dir,
    count=10,
    eps=0.01,
    timeout=30,
    spec_type=specs.InputSpecType.LINF,
    max_logged_split_steps=50,
    top_k_candidates=5,
    sample_values=8,
    summary_prefix=20,
):
    resolved_net = resolve_net_path(net)
    pair_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex

    args = build_verification_args(
        net_path=resolved_net,
        domain=domain,
        dataset=dataset,
        eps=eps,
        count=count,
        split=split,
        timeout=timeout,
        spec_type=spec_type,
    )

    base_props, _ = specs.get_specs(dataset, spec_type=spec_type, count=count, eps=eps)
    property_manifest = []
    for property_index, prop in enumerate(base_props):
        for clause_index in range(prop.get_input_clause_count()):
            property_manifest.append(property_metadata(prop.get_input_clause(clause_index), property_index, clause_index))

    original_net = util.get_net(resolved_net, dataset)
    perturbed_net = approximation_spec["approximation"].approximate(resolved_net, dataset)
    perturbation_delta = compare_networks(original_net, perturbed_net)

    original_logger = ArtifactRunLogger(
        output_dir=output_dir,
        pair_id=pair_id,
        model_role="original",
        original_model_path=resolved_net,
        model_path=resolved_net,
        perturbation={
            "type": approximation_spec["type"],
            "parameters": approximation_spec["params"],
            "realized_delta": perturbation_delta,
            "applied_to_this_run": False,
        },
        session_id=session_id,
        max_logged_split_steps=max_logged_split_steps,
        top_k_candidates=top_k_candidates,
        sample_values=sample_values,
        summary_prefix=summary_prefix,
    )
    perturbed_logger = ArtifactRunLogger(
        output_dir=output_dir,
        pair_id=pair_id,
        model_role="perturbed",
        original_model_path=resolved_net,
        model_path=f"{resolved_net}::{approximation_spec['type']}",
        perturbation={
            "type": approximation_spec["type"],
            "parameters": approximation_spec["params"],
            "realized_delta": perturbation_delta,
            "applied_to_this_run": True,
        },
        session_id=session_id,
        max_logged_split_steps=max_logged_split_steps,
        top_k_candidates=top_k_candidates,
        sample_values=sample_values,
        summary_prefix=summary_prefix,
    )

    original_results = Analyzer(
        args,
        net=original_net,
        template_store=None,
        artifact_logger=original_logger,
        enable_template_store=False,
    ).run_analyzer(props=copy.deepcopy(base_props))
    perturbed_results = Analyzer(
        args,
        net=perturbed_net,
        template_store=None,
        artifact_logger=perturbed_logger,
        enable_template_store=False,
    ).run_analyzer(props=copy.deepcopy(base_props))

    manifest_path = build_pair_manifest(
        pair_id=pair_id,
        session_id=session_id,
        output_dir=output_dir,
        original_path=resolved_net,
        perturbation={
            "type": approximation_spec["type"],
            "parameters": approximation_spec["params"],
            "realized_delta": perturbation_delta,
        },
        args=args,
        properties=property_manifest,
        results={
            "original": {
                "run_id": original_logger.run_id,
                "model_role": "original",
                "output_count": {k.name if hasattr(k, "name") else str(k): v for k, v in original_results.output_count.items()},
                "avg_time": original_results.avg_time,
                "avg_tree_size": original_results.avg_tree_size,
            },
            "perturbed": {
                "run_id": perturbed_logger.run_id,
                "model_role": "perturbed",
                "output_count": {k.name if hasattr(k, "name") else str(k): v for k, v in perturbed_results.output_count.items()},
                "avg_time": perturbed_results.avg_time,
                "avg_tree_size": perturbed_results.avg_tree_size,
            },
        },
    )

    return {
        "pair_id": pair_id,
        "session_id": session_id,
        "manifest_path": manifest_path,
        "original_run_id": original_logger.run_id,
        "perturbed_run_id": perturbed_logger.run_id,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run independent verification on M and M' and log comparable internal artifacts."
    )
    parser.add_argument("--net", required=True, help="Model path or model name under nnverify/nets/.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_MAP.keys()))
    parser.add_argument("--domain", required=True, choices=sorted(DOMAIN_MAP.keys()))
    parser.add_argument("--split", default="none", choices=sorted(SPLIT_MAP.keys()))
    parser.add_argument("--spec-type", default="linf", choices=sorted(SPEC_TYPE_MAP.keys()))
    parser.add_argument("--approx", required=True,
                        help="Approximation spec, e.g. quantize:int8, prune:percent=50, random:scale=0.001.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output-dir", default="results/artifact_observation")
    parser.add_argument("--max-logged-split-steps", type=int, default=50)
    parser.add_argument("--top-k-candidates", type=int, default=5)
    parser.add_argument("--sample-values", type=int, default=8)
    parser.add_argument("--summary-prefix", type=int, default=20)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    approximation_spec = parse_approximation(args.approx)

    summary = run_independent_pair_analysis(
        net=args.net,
        dataset=DATASET_MAP[args.dataset],
        domain=DOMAIN_MAP[args.domain],
        split=SPLIT_MAP[args.split],
        approximation_spec=approximation_spec,
        output_dir=args.output_dir,
        count=args.count,
        eps=args.eps,
        timeout=args.timeout,
        spec_type=SPEC_TYPE_MAP[args.spec_type],
        max_logged_split_steps=args.max_logged_split_steps,
        top_k_candidates=args.top_k_candidates,
        sample_values=args.sample_values,
        summary_prefix=args.summary_prefix,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
