"""Shared helpers for the m -> m' benchmark pipeline."""

from __future__ import annotations

import csv
import json
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


RESULT_RE = re.compile(r"Result:\s+(.+?)\s+in\s+([0-9.]+)\s+seconds")
ALPHABETA_TIME_RE = re.compile(r"alpha/beta optimization time:\s*([0-9.]+)")
DOMAINS_RE = re.compile(r"(\d+)\s+domains visited")
GLOBAL_LB_RE = re.compile(r"Global lower bound:\s*([^\s]+)")
CACHE_STATUS_RE = re.compile(r"Instance-cache status:\s*(\S+)")
FALLBACK_RE = re.compile(r"Instance-cache alpha warm-start rejected; falling back")

FLOAT_FIELDS = {
    "alpha_beta_opt_time_max_sec",
    "alpha_beta_opt_time_sum_sec",
    "baseline_alpha_beta_opt_time_sum_sec",
    "baseline_final_status_time_sec",
    "baseline_time_root_sec",
    "baseline_wall_time_sec",
    "delta_w_rel_l2",
    "delta_w_rel_l2_parent",
    "delta_w_rel_l2_root",
    "delta_w_rel_linf",
    "delta_w_rel_linf_parent",
    "delta_w_rel_linf_root",
    "final_status_time_sec",
    "global_lower_bound_last",
    "label_agreement_parent",
    "logit_mse_parent",
    "prep_source_wall_time_sec",
    "speedup_ratio_vs_baseline",
    "status_parity_rate",
    "wall_time_sec",
}

INT_FIELDS = {
    "alpha_beta_opt_calls",
    "alpha_fallback_count",
    "baseline_alpha_beta_opt_calls",
    "baseline_domains_visited_last",
    "baseline_property_index",
    "baseline_timeout_marker_count",
    "domains_visited_last",
    "domains_visited_max",
    "domains_root",
    "fallback_count",
    "prep_source_return_code",
    "property_index",
    "repeat",
    "return_code",
    "timeout_marker_count",
    "timeout_marker_count_root",
}

BOOL_FIELDS = {
    "alpha_loaded",
    "alpha_reset",
    "entered_bab",
    "entered_bab_root",
    "prep_source_ok",
    "skipped_due_to_source_prep",
    "status_parity_with_baseline",
}

COMPLETE_VERIFIER_ROOT = Path(__file__).resolve().parent.parent
M2M_BENCH_ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_CONFIG = M2M_BENCH_ROOT / "configs" / "resnet2b_vnnlib_template.yaml"
DEFAULT_PROPERTIES_SOURCE = (
    COMPLETE_VERIFIER_ROOT / "exp_configs" / "tutorial_examples" / "cifar10_resnet2b_instances.csv"
)
DEFAULT_BASE_CHECKPOINT = COMPLETE_VERIFIER_ROOT / "models" / "cifar10_resnet" / "resnet2b.pth"


def _safe_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_int(s: str | None) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def slugify(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    safe = safe.strip("._")
    return safe or "item"


def parse_extra(extra: str) -> list[str]:
    return shlex.split(extra) if extra else []


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML in {path}")
    return data


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping JSON in {path}")
    return data


def write_single_column_csv(path: Path, rows: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow([row])


def read_single_column_csv(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        values = []
        for row in reader:
            if not row:
                continue
            values.append(row[0].strip())
    return values


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], preferred_order: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV to {path}")
    fieldnames: list[str] = []
    if preferred_order:
        for name in preferred_order:
            if any(name in row for row in rows):
                fieldnames.append(name)
    extras = sorted({key for row in rows for key in row.keys()} - set(fieldnames))
    fieldnames.extend(extras)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def materialize_config(
    *,
    template_path: Path,
    checkpoint_path: Path,
    property_csv_path: Path,
    output_path: Path,
    results_file: Path | None = None,
    extra_config: dict[str, Any] | None = None,
) -> Path:
    config = load_yaml(template_path)
    config.setdefault("general", {})
    config.setdefault("model", {})
    config["general"]["root_path"] = str(property_csv_path.resolve().parent)
    config["general"]["csv_name"] = property_csv_path.resolve().name
    if results_file is not None:
        config["general"]["results_file"] = str(results_file.resolve())
    config["model"]["path"] = str(checkpoint_path.resolve())
    if extra_config:
        deep_update(config, extra_config)
    dump_yaml(config, output_path)
    return output_path


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def parse_metrics(text: str) -> dict[str, Any]:
    result_matches = RESULT_RE.findall(text)
    final_status = result_matches[-1][0] if result_matches else None
    final_status_time_sec = _safe_float(result_matches[-1][1]) if result_matches else None

    alpha_times = [_safe_float(match) for match in ALPHABETA_TIME_RE.findall(text)]
    alpha_times = [value for value in alpha_times if value is not None]
    domains = [int(match) for match in DOMAINS_RE.findall(text)]
    glb_matches = GLOBAL_LB_RE.findall(text)
    cache_hits = CACHE_STATUS_RE.findall(text)

    timeout_hits = text.count("Time out!!!!!!!!") + text.count("Time out!")
    safe_cnt = len(re.findall(r"Result:\s+safe", text))
    unsafe_cnt = len(re.findall(r"Result:\s+unsafe", text))
    unknown_cnt = len(re.findall(r"Result:\s+unknown", text))

    return {
        "final_status": final_status,
        "final_status_time_sec": final_status_time_sec,
        "num_result_lines": len(result_matches),
        "num_safe_results": safe_cnt,
        "num_unsafe_results": unsafe_cnt,
        "num_unknown_results": unknown_cnt,
        "alpha_beta_opt_time_sum_sec": sum(alpha_times) if alpha_times else None,
        "alpha_beta_opt_time_max_sec": max(alpha_times) if alpha_times else None,
        "alpha_beta_opt_calls": len(alpha_times),
        "domains_visited_last": domains[-1] if domains else None,
        "domains_visited_max": max(domains) if domains else None,
        "global_lower_bound_last": _safe_float(glb_matches[-1]) if glb_matches else None,
        "cache_status_last": cache_hits[-1] if cache_hits else None,
        "cache_statuses": ",".join(cache_hits) if cache_hits else "",
        "timeout_marker_count": timeout_hits,
        "entered_bab": bool(domains),
        "fallback_count": len(FALLBACK_RE.findall(text)),
    }


def run_abcrown(
    *,
    python_exec: str,
    config_path: Path,
    device: str,
    workdir: Path,
    log_file: Path,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    cmd = [
        python_exec,
        "abcrown.py",
        "--config",
        str(config_path.resolve()),
        "--device",
        device,
    ]
    if extra_args:
        cmd.extend(extra_args)
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(workdir.resolve()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wall_time = time.time() - start
    ensure_dir(log_file.parent)
    log_file.write_text(proc.stdout, encoding="utf-8")
    metrics = parse_metrics(proc.stdout)
    metrics["return_code"] = proc.returncode
    metrics["wall_time_sec"] = wall_time
    metrics["command"] = " ".join(shlex.quote(part) for part in cmd)
    metrics["log_file"] = str(log_file.resolve())
    return metrics


def classify_property(
    metrics: dict[str, Any],
    *,
    medium_time_threshold: float,
    hard_time_threshold: float,
    hard_domains_threshold: int,
) -> str:
    wall_time = _safe_float(str(metrics.get("wall_time_sec", "")))
    domains = _safe_int(str(metrics.get("domains_visited_max", "")))
    final_status = str(metrics.get("final_status") or "")
    if metrics.get("return_code", 0) != 0:
        return "hard"
    if metrics.get("timeout_marker_count", 0):
        return "hard"
    if final_status.lower() == "unknown":
        return "hard"
    if domains is not None and domains >= hard_domains_threshold:
        return "hard"
    if wall_time is not None and wall_time >= hard_time_threshold:
        return "hard"
    if metrics.get("entered_bab"):
        return "medium"
    if wall_time is not None and wall_time >= medium_time_threshold:
        return "medium"
    return "easy"


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    normalized = []
    for row in rows:
        normalized.append(coerce_row_types(row))
    return normalized


def coerce_row_types(row: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in row.items():
        if key in FLOAT_FIELDS:
            converted[key] = _safe_float(value)
        elif key in INT_FIELDS:
            converted[key] = _safe_int(value)
        elif key in BOOL_FIELDS:
            converted[key] = str(value).lower() in {"1", "true", "yes"}
        else:
            converted[key] = value
    return converted


def benchmark_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("lineage_id"),
        row.get("target_model_id"),
        row.get("property_id"),
        row.get("repeat", 1),
    )


def build_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        benchmark_key(row): row
        for row in rows
        if row.get("reuse_variant") == "baseline"
    }
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        if row.get("reuse_variant") == "baseline":
            continue
        baseline = baselines.get(benchmark_key(row))
        merged = dict(row)
        if baseline is None:
            merged["status_parity_with_baseline"] = None
            merged["speedup_ratio_vs_baseline"] = None
        else:
            merged["baseline_status"] = baseline.get("final_status")
            merged["baseline_wall_time_sec"] = baseline.get("wall_time_sec")
            merged["baseline_final_status_time_sec"] = baseline.get("final_status_time_sec")
            merged["baseline_alpha_beta_opt_time_sum_sec"] = baseline.get("alpha_beta_opt_time_sum_sec")
            merged["baseline_alpha_beta_opt_calls"] = baseline.get("alpha_beta_opt_calls")
            merged["baseline_domains_visited_last"] = baseline.get("domains_visited_last")
            merged["baseline_timeout_marker_count"] = baseline.get("timeout_marker_count")
            merged["status_parity_with_baseline"] = (
                baseline.get("final_status") == row.get("final_status")
            )
            base_wall = baseline.get("wall_time_sec")
            row_wall = row.get("wall_time_sec")
            if isinstance(base_wall, (int, float)) and isinstance(row_wall, (int, float)) and row_wall > 0:
                merged["speedup_ratio_vs_baseline"] = base_wall / row_wall
            else:
                merged["speedup_ratio_vs_baseline"] = None
        comparisons.append(merged)
    return comparisons


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def relative_improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return (baseline - candidate) / baseline


def summary_by_field(
    comparisons: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in comparisons:
        key = str(row.get(field) or "unknown")
        grouped.setdefault(key, []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for key, items in grouped.items():
        parity_values = [row.get("status_parity_with_baseline") for row in items if row.get("status_parity_with_baseline") is not None]
        wall_values = [row["wall_time_sec"] for row in items if isinstance(row.get("wall_time_sec"), (int, float))]
        speedups = [row["speedup_ratio_vs_baseline"] for row in items if isinstance(row.get("speedup_ratio_vs_baseline"), (int, float))]
        summary[key] = {
            "n": len(items),
            "status_parity_rate": (
                sum(1 for value in parity_values if value) / len(parity_values)
                if parity_values
                else None
            ),
            "median_wall_time_sec": median_or_none(wall_values),
            "median_speedup_ratio_vs_baseline": median_or_none(speedups),
        }
    return summary


def default_python() -> str:
    return sys.executable
