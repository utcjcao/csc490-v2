#!/usr/bin/env python3
"""Analyze m -> m' benchmark output from runs.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
COMPLETE_VERIFIER_ROOT = SCRIPT_DIR.parents[1]
if str(COMPLETE_VERIFIER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPLETE_VERIFIER_ROOT))

from m2m_bench.common import build_comparison_rows, ensure_dir, load_manifest_rows, median_or_none, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze m -> m' benchmark runs.csv output.")
    parser.add_argument("--runs-csv", required=True)
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional summary path. Defaults to <runs-csv-dir>/analysis_summary.json.",
    )
    parser.add_argument(
        "--output-text",
        default="",
        help="Optional text summary path. Defaults to <runs-csv-dir>/analysis_summary.txt.",
    )
    return parser.parse_args()


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parity = [row.get("status_parity_with_baseline") for row in rows if row.get("status_parity_with_baseline") is not None]
    wall = [row["wall_time_sec"] for row in rows if isinstance(row.get("wall_time_sec"), (int, float))]
    speedup = [row["speedup_ratio_vs_baseline"] for row in rows if isinstance(row.get("speedup_ratio_vs_baseline"), (int, float))]
    alpha_delta = []
    domains_delta = []
    for row in rows:
        baseline_alpha = row.get("baseline_alpha_beta_opt_time_sum_sec")
        run_alpha = row.get("alpha_beta_opt_time_sum_sec")
        if isinstance(baseline_alpha, (int, float)) and isinstance(run_alpha, (int, float)):
            alpha_delta.append(baseline_alpha - run_alpha)
        baseline_domains = row.get("baseline_domains_visited_last")
        run_domains = row.get("domains_visited_last")
        if isinstance(baseline_domains, (int, float)) and isinstance(run_domains, (int, float)):
            domains_delta.append(baseline_domains - run_domains)
    return {
        "n": len(rows),
        "status_parity_rate": (sum(1 for value in parity if value) / len(parity)) if parity else None,
        "median_wall_time_sec": median_or_none(wall),
        "median_speedup_ratio_vs_baseline": median_or_none(speedup),
        "median_alpha_time_delta_sec": median_or_none(alpha_delta),
        "median_domains_delta": median_or_none(domains_delta),
    }


def grouped_summary(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = " | ".join(str(row.get(field) or "unknown") for field in fields)
        grouped.setdefault(key, []).append(row)
    return {key: metric_summary(items) for key, items in sorted(grouped.items())}


def render_text_summary(comparisons: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = []
    alpha_rows = [row for row in comparisons if "alpha" in str(row.get("reuse_variant", ""))]
    medium_hard = [row for row in alpha_rows if row.get("difficulty_bin") in {"medium", "hard"}]
    if alpha_rows:
        alpha_summary = metric_summary(alpha_rows)
        lines.append(
            "Alpha reuse overall: "
            f"n={alpha_summary['n']}, parity={alpha_summary['status_parity_rate']}, "
            f"median_speedup={alpha_summary['median_speedup_ratio_vs_baseline']}, "
            f"median_alpha_time_delta={alpha_summary['median_alpha_time_delta_sec']}."
        )
    else:
        lines.append("Alpha reuse overall: no alpha-based comparison rows found.")

    if medium_hard:
        mh_summary = metric_summary(medium_hard)
        lines.append(
            "Alpha reuse on medium/hard properties: "
            f"n={mh_summary['n']}, parity={mh_summary['status_parity_rate']}, "
            f"median_speedup={mh_summary['median_speedup_ratio_vs_baseline']}, "
            f"median_domains_delta={mh_summary['median_domains_delta']}."
        )
    else:
        lines.append("Alpha reuse on medium/hard properties: no rows found.")

    drift_groups = summary["by_variant_and_drift_bucket"]
    if drift_groups:
        best_drift = max(
            drift_groups.items(),
            key=lambda item: item[1]["median_speedup_ratio_vs_baseline"]
            if item[1]["median_speedup_ratio_vs_baseline"] is not None
            else float("-inf"),
        )
        lines.append(
            "Best drift slice by median speedup: "
            f"{best_drift[0]} -> {best_drift[1]['median_speedup_ratio_vs_baseline']}."
        )
    else:
        lines.append("Best drift slice by median speedup: no grouped rows found.")

    difficulty_groups = summary["by_variant_and_difficulty"]
    if difficulty_groups:
        best_difficulty = max(
            difficulty_groups.items(),
            key=lambda item: item[1]["median_speedup_ratio_vs_baseline"]
            if item[1]["median_speedup_ratio_vs_baseline"] is not None
            else float("-inf"),
        )
        lines.append(
            "Best difficulty slice by median speedup: "
            f"{best_difficulty[0]} -> {best_difficulty[1]['median_speedup_ratio_vs_baseline']}."
        )
    else:
        lines.append("Best difficulty slice by median speedup: no grouped rows found.")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    runs_csv = Path(args.runs_csv).resolve()
    if not runs_csv.exists():
        raise FileNotFoundError(f"runs.csv not found: {runs_csv}")

    rows = load_manifest_rows(runs_csv)
    comparisons = build_comparison_rows(rows)
    summary = {
        "schema_version": "m2m-analysis-summary-v1",
        "runs_csv": str(runs_csv),
        "n_rows": len(rows),
        "n_comparisons": len(comparisons),
        "overall": metric_summary(comparisons),
        "by_variant": grouped_summary(comparisons, ["reuse_variant"]),
        "by_difficulty_bin": grouped_summary(comparisons, ["difficulty_bin"]),
        "by_drift_bucket": grouped_summary(comparisons, ["drift_bucket"]),
        "by_lineage_id": grouped_summary(comparisons, ["lineage_id"]),
        "by_variant_and_difficulty": grouped_summary(comparisons, ["reuse_variant", "difficulty_bin"]),
        "by_variant_and_drift_bucket": grouped_summary(comparisons, ["reuse_variant", "drift_bucket"]),
    }
    output_json = Path(args.output_json).resolve() if args.output_json else runs_csv.parent / "analysis_summary.json"
    output_text = Path(args.output_text).resolve() if args.output_text else runs_csv.parent / "analysis_summary.txt"
    ensure_dir(output_json.parent)
    ensure_dir(output_text.parent)
    write_json(summary, output_json)
    text_summary = render_text_summary(comparisons, summary)
    output_text.write_text(text_summary + "\n", encoding="utf-8")

    print(json.dumps(summary["overall"], indent=2))
    print()
    print(text_summary)
    print()
    print(f"Wrote analysis summary to {output_json}")
    print(f"Wrote text summary to {output_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
