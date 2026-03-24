#!/usr/bin/env python3
"""Select a balanced VNNLIB subset for the m -> m' benchmark."""

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
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_PROPERTIES_SOURCE,
    DEFAULT_TEMPLATE_CONFIG,
    classify_property,
    default_python,
    ensure_dir,
    materialize_config,
    parse_extra,
    read_single_column_csv,
    run_abcrown,
    slugify,
    write_csv_rows,
    write_single_column_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan root-model properties and select a balanced benchmark subset.")
    parser.add_argument("--template-config", default=str(DEFAULT_TEMPLATE_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_BASE_CHECKPOINT))
    parser.add_argument("--source-csv", default=str(DEFAULT_PROPERTIES_SOURCE))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--per-bin", type=int, default=8)
    parser.add_argument("--medium-time-threshold", type=float, default=5.0)
    parser.add_argument("--hard-time-threshold", type=float, default=30.0)
    parser.add_argument("--hard-domains-threshold", type=int, default=1000)
    parser.add_argument("--max-properties", type=int, default=0, help="Optional cap for debugging.")
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Allow output when one difficulty bin has fewer than --per-bin items.",
    )
    return parser.parse_args()


def make_property_id(index: int, vnnlib_path: str) -> str:
    return f"p{index:03d}_{slugify(Path(vnnlib_path).stem)}"


def format_wall_time(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}s"
    return "n/a"


def select_bin(rows: list[dict[str, Any]], difficulty_bin: str, per_bin: int) -> list[dict[str, Any]]:
    if difficulty_bin == "easy":
        ordered = sorted(
            rows,
            key=lambda row: (
                row.get("wall_time_sec") if row.get("wall_time_sec") is not None else float("inf"),
                row["property_index"],
            ),
        )
    elif difficulty_bin == "medium":
        ordered = sorted(
            rows,
            key=lambda row: (
                -(row.get("domains_visited_max") or 0),
                -(row.get("wall_time_sec") or 0.0),
                row["property_index"],
            ),
        )
    else:
        ordered = sorted(
            rows,
            key=lambda row: (
                -(row.get("timeout_marker_count") or 0),
                -(row.get("domains_visited_max") or 0),
                -(row.get("wall_time_sec") or 0.0),
                row["property_index"],
            ),
        )
    return ordered[:per_bin]


def main() -> int:
    args = parse_args()
    template_config = Path(args.template_config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    source_csv = Path(args.source_csv).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    scan_dir = ensure_dir(output_dir / "scan_runs")

    if not template_config.exists():
        raise FileNotFoundError(f"Template config not found: {template_config}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    vnnlibs = read_single_column_csv(source_csv)
    if args.max_properties > 0:
        vnnlibs = vnnlibs[: args.max_properties]
    if not vnnlibs:
        raise ValueError(f"No properties found in {source_csv}")

    extra_args = parse_extra(args.extra_args)
    scan_rows: list[dict[str, Any]] = []
    for index, vnnlib_path in enumerate(vnnlibs):
        property_id = make_property_id(index, vnnlib_path)
        run_dir = ensure_dir(scan_dir / property_id)
        property_csv = run_dir / "property.csv"
        config_path = run_dir / "config.yaml"
        log_path = run_dir / "abcrown.log"
        result_path = run_dir / "results.txt"

        write_single_column_csv(property_csv, [vnnlib_path])
        materialize_config(
            template_path=template_config,
            checkpoint_path=checkpoint,
            property_csv_path=property_csv,
            output_path=config_path,
            results_file=result_path,
        )
        metrics = run_abcrown(
            python_exec=args.python,
            config_path=config_path,
            device=args.device,
            workdir=CV_ROOT,
            log_file=log_path,
            extra_args=extra_args,
        )
        difficulty_bin = classify_property(
            metrics,
            medium_time_threshold=args.medium_time_threshold,
            hard_time_threshold=args.hard_time_threshold,
            hard_domains_threshold=args.hard_domains_threshold,
        )
        row = {
            "property_id": property_id,
            "property_index": index,
            "vnnlib_path": vnnlib_path,
            "difficulty_bin": difficulty_bin,
            "baseline_status_root": metrics.get("final_status"),
            "baseline_time_root_sec": metrics.get("wall_time_sec"),
            "domains_root": metrics.get("domains_visited_last"),
            "entered_bab_root": metrics.get("entered_bab"),
            "timeout_marker_count_root": metrics.get("timeout_marker_count"),
            "return_code": metrics.get("return_code"),
            "wall_time_sec": metrics.get("wall_time_sec"),
            "final_status_time_sec": metrics.get("final_status_time_sec"),
            "alpha_beta_opt_time_sum_sec": metrics.get("alpha_beta_opt_time_sum_sec"),
            "alpha_beta_opt_calls": metrics.get("alpha_beta_opt_calls"),
            "domains_visited_last": metrics.get("domains_visited_last"),
            "domains_visited_max": metrics.get("domains_visited_max"),
            "entered_bab": metrics.get("entered_bab"),
            "timeout_marker_count": metrics.get("timeout_marker_count"),
            "global_lower_bound_last": metrics.get("global_lower_bound_last"),
            "cache_status_last": metrics.get("cache_status_last"),
            "fallback_count": metrics.get("fallback_count"),
            "log_file": metrics.get("log_file"),
            "command": metrics.get("command"),
        }
        scan_rows.append(row)
        print(
            f"[{index + 1}/{len(vnnlibs)}] {property_id}: "
            f"status={row['baseline_status_root']} bin={difficulty_bin} "
            f"wall={format_wall_time(row['baseline_time_root_sec'])} domains={row['domains_root']}"
        )

    write_csv_rows(
        output_dir / "property_scan.csv",
        scan_rows,
        preferred_order=[
            "property_id",
            "property_index",
            "vnnlib_path",
            "difficulty_bin",
            "baseline_status_root",
            "baseline_time_root_sec",
            "domains_root",
            "entered_bab_root",
            "timeout_marker_count_root",
            "return_code",
            "log_file",
            "command",
        ],
    )

    selected: list[dict[str, Any]] = []
    for difficulty_bin in ("easy", "medium", "hard"):
        bucket_rows = [row for row in scan_rows if row["difficulty_bin"] == difficulty_bin]
        chosen = select_bin(bucket_rows, difficulty_bin, args.per_bin)
        if len(chosen) < args.per_bin and not args.allow_shortfall:
            raise RuntimeError(
                f"Difficulty bin '{difficulty_bin}' has only {len(chosen)} rows, "
                f"but --per-bin={args.per_bin}."
            )
        for row in chosen:
            selected.append(
                {
                    "property_id": row["property_id"],
                    "property_index": row["property_index"],
                    "vnnlib_path": row["vnnlib_path"],
                    "difficulty_bin": row["difficulty_bin"],
                    "baseline_status_root": row["baseline_status_root"],
                    "baseline_time_root_sec": row["baseline_time_root_sec"],
                    "domains_root": row["domains_root"],
                    "entered_bab_root": row["entered_bab_root"],
                    "timeout_marker_count_root": row["timeout_marker_count_root"],
                }
            )

    selected = sorted(selected, key=lambda row: (row["difficulty_bin"], row["property_index"]))
    write_csv_rows(
        output_dir / "properties.csv",
        selected,
        preferred_order=[
            "property_id",
            "property_index",
            "vnnlib_path",
            "difficulty_bin",
            "baseline_status_root",
            "baseline_time_root_sec",
            "domains_root",
            "entered_bab_root",
            "timeout_marker_count_root",
        ],
    )
    print(f"Wrote property scan to {(output_dir / 'property_scan.csv').resolve()}")
    print(f"Wrote selected properties to {(output_dir / 'properties.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
