#!/usr/bin/env python3
"""Run the m -> m' benchmark over a lineage manifest and selected properties."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
COMPLETE_VERIFIER_ROOT = SCRIPT_DIR.parents[1]
if str(COMPLETE_VERIFIER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPLETE_VERIFIER_ROOT))

from m2m_bench.common import (
    COMPLETE_VERIFIER_ROOT as CV_ROOT,
    DEFAULT_TEMPLATE_CONFIG,
    benchmark_key,
    build_comparison_rows,
    default_python,
    ensure_dir,
    load_manifest_rows,
    materialize_config,
    parse_extra,
    read_json,
    run_abcrown,
    summary_by_field,
    write_csv_rows,
    write_json,
    write_single_column_csv,
)


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {"source_mode": None, "alpha": False, "branch": False},
    "alpha_only_parent_to_child": {"source_mode": "parent", "alpha": True, "branch": False},
    "branch_only_parent_to_child": {"source_mode": "parent", "alpha": False, "branch": True},
    "alpha_plus_branch_parent_to_child": {"source_mode": "parent", "alpha": True, "branch": True},
    "alpha_only_root_to_child": {"source_mode": "root", "alpha": True, "branch": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline and reuse variants for the m -> m' benchmark.")
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--properties-csv", required=True)
    parser.add_argument("--template-config", default=str(DEFAULT_TEMPLATE_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", default="baseline,alpha_only_parent_to_child")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--target-model-ids", default="", help="Optional comma-separated target model_id filter.")
    parser.add_argument("--property-ids", default="", help="Optional comma-separated property_id filter.")
    parser.add_argument("--strict-recompute", action="store_true")
    return parser.parse_args()


def parse_variant_list(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    if not variants:
        raise ValueError("At least one variant must be provided.")
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
    if "baseline" not in variants:
        variants = ["baseline", *variants]
    return variants


def split_filter(raw: str) -> set[str] | None:
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def validate_lineage_manifest(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("lineage.json must contain a non-empty 'models' list.")
    model_map: dict[str, dict[str, Any]] = {}
    root = None
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("Each model record in lineage.json must be a mapping.")
        model_id = model.get("model_id")
        if not model_id:
            raise ValueError("Every model record requires model_id.")
        if model_id in model_map:
            raise ValueError(f"Duplicate model_id in lineage manifest: {model_id}")
        checkpoint_path = Path(str(model.get("checkpoint_path", ""))).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for {model_id}: {checkpoint_path}")
        model["checkpoint_path"] = str(checkpoint_path)
        model_map[model_id] = model
        if model.get("parent_id") is None:
            if root is not None:
                raise ValueError("lineage.json must have exactly one root model.")
            root = model
    if root is None:
        raise ValueError("lineage.json must define one root model.")
    for model in models:
        parent_id = model.get("parent_id")
        if parent_id is not None and parent_id not in model_map:
            raise ValueError(f"Unknown parent_id '{parent_id}' for model '{model['model_id']}'.")
    return model_map, root


def validate_properties(path: Path) -> list[dict[str, Any]]:
    rows = load_manifest_rows(path)
    if not rows:
        raise ValueError(f"No rows found in properties CSV: {path}")
    seen = set()
    for row in rows:
        property_id = row.get("property_id")
        if not property_id:
            raise ValueError("Each property record requires property_id.")
        if property_id in seen:
            raise ValueError(f"Duplicate property_id in properties CSV: {property_id}")
        seen.add(property_id)
        if not row.get("vnnlib_path"):
            raise ValueError(f"Property '{property_id}' is missing vnnlib_path.")
    return rows


def variant_target_args(variant: str, cache_path: Path, model_group: str, strict_recompute: bool) -> list[str]:
    spec = VARIANTS[variant]
    if variant == "baseline":
        return []
    args = [
        "--enable_instance_cache",
        "--instance_cache_path",
        str(cache_path),
        "--instance_cache_model_group",
        model_group,
    ]
    if spec["alpha"]:
        args.append("--instance_cache_alpha_warmstart")
    if spec["branch"]:
        args.append("--instance_cache_branching_hints")
    if strict_recompute:
        args.append("--instance_cache_strict_recompute")
    return args


def source_prep_args(cache_path: Path, model_group: str) -> list[str]:
    return [
        "--enable_instance_cache",
        "--instance_cache_path",
        str(cache_path),
        "--instance_cache_model_group",
        model_group,
    ]


def format_wall_time(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}s"
    return "n/a"


def run_baseline(
    *,
    template_config: Path,
    target_model: dict[str, Any],
    property_row: dict[str, Any],
    output_dir: Path,
    python_exec: str,
    device: str,
    extra_args: list[str],
    repeat: int,
) -> dict[str, Any]:
    trial_dir = ensure_dir(
        output_dir
        / "trials"
        / str(target_model["lineage_id"])
        / target_model["model_id"]
        / property_row["property_id"]
        / f"baseline_rep{repeat}"
    )
    property_csv = trial_dir / "property.csv"
    config_path = trial_dir / "config.yaml"
    log_path = trial_dir / "abcrown.log"
    results_path = trial_dir / "results.txt"
    write_single_column_csv(property_csv, [property_row["vnnlib_path"]])
    materialize_config(
        template_path=template_config,
        checkpoint_path=Path(target_model["checkpoint_path"]),
        property_csv_path=property_csv,
        output_path=config_path,
        results_file=results_path,
    )
    metrics = run_abcrown(
        python_exec=python_exec,
        config_path=config_path,
        device=device,
        workdir=CV_ROOT,
        log_file=log_path,
        extra_args=extra_args,
    )
    row = base_row(target_model, property_row, repeat)
    row.update(metrics)
    row.update(
        {
            "reuse_variant": "baseline",
            "reuse_source": "none",
            "source_model_id": "",
            "alpha_loaded": False,
            "alpha_reset": False,
            "prep_source_ok": False,
            "skipped_due_to_source_prep": False,
            "trial_dir": str(trial_dir),
            "config_path": str(config_path),
            "property_csv_path": str(property_csv),
        }
    )
    return row


def run_reuse_variant(
    *,
    template_config: Path,
    source_model: dict[str, Any],
    target_model: dict[str, Any],
    property_row: dict[str, Any],
    variant: str,
    output_dir: Path,
    python_exec: str,
    device: str,
    extra_args: list[str],
    repeat: int,
    strict_recompute: bool,
) -> dict[str, Any]:
    trial_dir = ensure_dir(
        output_dir
        / "trials"
        / str(target_model["lineage_id"])
        / target_model["model_id"]
        / property_row["property_id"]
        / f"{variant}_rep{repeat}"
    )
    cache_dir = ensure_dir(trial_dir / "cache")
    property_csv = trial_dir / "property.csv"
    source_config = trial_dir / "source_config.yaml"
    target_config = trial_dir / "target_config.yaml"
    source_log = trial_dir / "source_seed.log"
    target_log = trial_dir / "target.log"
    source_results = trial_dir / "source_results.txt"
    target_results = trial_dir / "target_results.txt"
    write_single_column_csv(property_csv, [property_row["vnnlib_path"]])

    model_group = ":".join(
        [
            "m2m",
            str(target_model["lineage_id"]),
            source_model["model_id"],
            target_model["model_id"],
            property_row["property_id"],
            f"rep{repeat}",
        ]
    )

    materialize_config(
        template_path=template_config,
        checkpoint_path=Path(source_model["checkpoint_path"]),
        property_csv_path=property_csv,
        output_path=source_config,
        results_file=source_results,
    )
    prep_metrics = run_abcrown(
        python_exec=python_exec,
        config_path=source_config,
        device=device,
        workdir=CV_ROOT,
        log_file=source_log,
        extra_args=source_prep_args(cache_dir, model_group) + extra_args,
    )

    row = base_row(target_model, property_row, repeat)
    row.update(
        {
            "reuse_variant": variant,
            "reuse_source": VARIANTS[variant]["source_mode"],
            "source_model_id": source_model["model_id"],
            "prep_source_return_code": prep_metrics.get("return_code"),
            "prep_source_wall_time_sec": prep_metrics.get("wall_time_sec"),
            "prep_source_final_status": prep_metrics.get("final_status"),
            "prep_source_ok": prep_metrics.get("return_code") == 0,
            "prep_source_log_file": prep_metrics.get("log_file"),
            "prep_source_command": prep_metrics.get("command"),
            "trial_dir": str(trial_dir),
            "config_path": str(target_config),
            "property_csv_path": str(property_csv),
        }
    )
    if prep_metrics.get("return_code") != 0:
        row.update(
            {
                "return_code": None,
                "final_status": "source-prep-failed",
                "skipped_due_to_source_prep": True,
                "alpha_loaded": False,
                "alpha_reset": False,
            }
        )
        return row

    materialize_config(
        template_path=template_config,
        checkpoint_path=Path(target_model["checkpoint_path"]),
        property_csv_path=property_csv,
        output_path=target_config,
        results_file=target_results,
    )
    metrics = run_abcrown(
        python_exec=python_exec,
        config_path=target_config,
        device=device,
        workdir=CV_ROOT,
        log_file=target_log,
        extra_args=variant_target_args(variant, cache_dir, model_group, strict_recompute) + extra_args,
    )
    row.update(metrics)
    row["skipped_due_to_source_prep"] = False
    row["alpha_loaded"] = bool(VARIANTS[variant]["alpha"] and metrics.get("cache_status_last") == "hit-exact")
    row["alpha_reset"] = bool(metrics.get("fallback_count"))
    return row


def base_row(target_model: dict[str, Any], property_row: dict[str, Any], repeat: int) -> dict[str, Any]:
    return {
        "lineage_id": target_model["lineage_id"],
        "target_model_id": target_model["model_id"],
        "drift_bucket": target_model.get("drift_bucket"),
        "delta_w_rel_l2": target_model.get("delta_w_rel_l2"),
        "delta_w_rel_linf": target_model.get("delta_w_rel_linf"),
        "delta_w_rel_l2_root": target_model.get("delta_w_rel_l2_root"),
        "delta_w_rel_linf_root": target_model.get("delta_w_rel_linf_root"),
        "delta_w_rel_l2_parent": target_model.get("delta_w_rel_l2_parent"),
        "delta_w_rel_linf_parent": target_model.get("delta_w_rel_linf_parent"),
        "label_agreement_parent": target_model.get("label_agreement_parent"),
        "logit_mse_parent": target_model.get("logit_mse_parent"),
        "property_id": property_row["property_id"],
        "property_index": property_row.get("property_index"),
        "vnnlib_path": property_row["vnnlib_path"],
        "difficulty_bin": property_row.get("difficulty_bin"),
        "repeat": repeat,
    }


def choose_source_model(model_map: dict[str, dict[str, Any]], root_model: dict[str, Any], target_model: dict[str, Any], variant: str) -> dict[str, Any]:
    source_mode = VARIANTS[variant]["source_mode"]
    if source_mode == "parent":
        parent_id = target_model.get("parent_id")
        if parent_id is None:
            raise ValueError(f"Target model {target_model['model_id']} has no parent for variant {variant}.")
        return model_map[parent_id]
    if source_mode == "root":
        return root_model
    raise ValueError(f"Variant {variant} does not use a source model.")


def main() -> int:
    args = parse_args()
    lineage_manifest = Path(args.lineage_manifest).resolve()
    properties_csv = Path(args.properties_csv).resolve()
    template_config = Path(args.template_config).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())

    if not lineage_manifest.exists():
        raise FileNotFoundError(f"Lineage manifest not found: {lineage_manifest}")
    if not properties_csv.exists():
        raise FileNotFoundError(f"Properties CSV not found: {properties_csv}")
    if not template_config.exists():
        raise FileNotFoundError(f"Template config not found: {template_config}")

    variants = parse_variant_list(args.variants)
    extra_args = parse_extra(args.extra_args)
    target_filter = split_filter(args.target_model_ids)
    property_filter = split_filter(args.property_ids)

    manifest = read_json(lineage_manifest)
    model_map, root_model = validate_lineage_manifest(manifest)
    property_rows = validate_properties(properties_csv)

    target_models = [
        model
        for model in model_map.values()
        if model.get("parent_id") is not None and (target_filter is None or model["model_id"] in target_filter)
    ]
    if not target_models:
        raise ValueError("No target models selected for the benchmark.")
    if property_filter is not None:
        property_rows = [row for row in property_rows if row["property_id"] in property_filter]
    if not property_rows:
        raise ValueError("No properties selected for the benchmark.")

    rows: list[dict[str, Any]] = []
    total_trials = len(target_models) * len(property_rows) * args.repeats
    trial_index = 0
    for target_model in sorted(target_models, key=lambda model: model["model_id"]):
        for property_row in sorted(property_rows, key=lambda row: row["property_index"]):
            for repeat in range(1, args.repeats + 1):
                trial_index += 1
                print(
                    f"[{trial_index}/{total_trials}] target={target_model['model_id']} "
                    f"property={property_row['property_id']} repeat={repeat}"
                )
                baseline_row = run_baseline(
                    template_config=template_config,
                    target_model=target_model,
                    property_row=property_row,
                    output_dir=output_dir,
                    python_exec=args.python,
                    device=args.device,
                    extra_args=extra_args,
                    repeat=repeat,
                )
                rows.append(baseline_row)
                print(
                    f"  baseline status={baseline_row.get('final_status')} "
                    f"wall={format_wall_time(baseline_row.get('wall_time_sec'))}"
                )

                for variant in variants:
                    if variant == "baseline":
                        continue
                    source_model = choose_source_model(model_map, root_model, target_model, variant)
                    variant_row = run_reuse_variant(
                        template_config=template_config,
                        source_model=source_model,
                        target_model=target_model,
                        property_row=property_row,
                        variant=variant,
                        output_dir=output_dir,
                        python_exec=args.python,
                        device=args.device,
                        extra_args=extra_args,
                        repeat=repeat,
                        strict_recompute=args.strict_recompute,
                    )
                    rows.append(variant_row)
                    print(
                        f"  {variant} source={source_model['model_id']} "
                        f"status={variant_row.get('final_status')} "
                        f"wall={format_wall_time(variant_row.get('wall_time_sec'))}"
                    )

    comparisons = build_comparison_rows(rows)
    comparison_lookup = {
        (benchmark_key(row), row["reuse_variant"]): row for row in comparisons
    }
    for row in rows:
        key = (benchmark_key(row), row["reuse_variant"])
        if key in comparison_lookup:
            row.update(
                {
                    "baseline_status": comparison_lookup[key].get("baseline_status"),
                    "baseline_wall_time_sec": comparison_lookup[key].get("baseline_wall_time_sec"),
                    "baseline_final_status_time_sec": comparison_lookup[key].get("baseline_final_status_time_sec"),
                    "baseline_alpha_beta_opt_time_sum_sec": comparison_lookup[key].get("baseline_alpha_beta_opt_time_sum_sec"),
                    "baseline_alpha_beta_opt_calls": comparison_lookup[key].get("baseline_alpha_beta_opt_calls"),
                    "baseline_domains_visited_last": comparison_lookup[key].get("baseline_domains_visited_last"),
                    "baseline_timeout_marker_count": comparison_lookup[key].get("baseline_timeout_marker_count"),
                    "status_parity_with_baseline": comparison_lookup[key].get("status_parity_with_baseline"),
                    "speedup_ratio_vs_baseline": comparison_lookup[key].get("speedup_ratio_vs_baseline"),
                }
            )

    write_csv_rows(
        output_dir / "runs.csv",
        rows,
        preferred_order=[
            "lineage_id",
            "source_model_id",
            "target_model_id",
            "property_id",
            "property_index",
            "difficulty_bin",
            "drift_bucket",
            "reuse_variant",
            "reuse_source",
            "repeat",
            "final_status",
            "baseline_status",
            "status_parity_with_baseline",
            "return_code",
            "wall_time_sec",
            "baseline_wall_time_sec",
            "speedup_ratio_vs_baseline",
            "alpha_beta_opt_time_sum_sec",
            "baseline_alpha_beta_opt_time_sum_sec",
            "domains_visited_last",
            "baseline_domains_visited_last",
            "entered_bab",
            "timeout_marker_count",
            "cache_status_last",
            "alpha_loaded",
            "alpha_reset",
            "prep_source_ok",
            "prep_source_return_code",
            "prep_source_wall_time_sec",
            "vnnlib_path",
            "config_path",
            "property_csv_path",
            "trial_dir",
            "log_file",
        ],
    )

    summary = {
        "schema_version": "m2m-runs-summary-v1",
        "lineage_manifest": str(lineage_manifest),
        "properties_csv": str(properties_csv),
        "variants": variants,
        "repeats": args.repeats,
        "n_rows": len(rows),
        "n_comparisons": len(comparisons),
        "by_variant": summary_by_field(comparisons, "reuse_variant"),
        "by_difficulty_bin": summary_by_field(comparisons, "difficulty_bin"),
        "by_drift_bucket": summary_by_field(comparisons, "drift_bucket"),
        "by_lineage_id": summary_by_field(comparisons, "lineage_id"),
    }
    write_json(summary, output_dir / "summary.json")
    print(f"Wrote benchmark runs to {(output_dir / 'runs.csv').resolve()}")
    print(f"Wrote summary to {(output_dir / 'summary.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
