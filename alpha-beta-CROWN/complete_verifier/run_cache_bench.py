#!/usr/bin/env python3
"""Small benchmark harness for alpha-beta-CROWN cache experiments.

Runs a set of variants repeatedly, stores full logs, and writes parsed metrics
to CSV for quick A/B analysis.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


RESULT_RE = re.compile(r"Result:\s+(.+?)\s+in\s+([0-9.]+)\s+seconds")
ALPHABETA_TIME_RE = re.compile(r"alpha/beta optimization time:\s*([0-9.]+)")
DOMAINS_RE = re.compile(r"(\d+)\s+domains visited")
GLOBAL_LB_RE = re.compile(r"Global lower bound:\s*([^\s]+)")
CACHE_STATUS_RE = re.compile(r"Instance-cache status:\s*(\S+)")


def _parse_extra(extra: str) -> List[str]:
    if not extra:
        return []
    return shlex.split(extra)


def _variant_args(variant: str, cache_path: str) -> List[str]:
    if variant == "baseline":
        return []
    if variant == "cache":
        return [
            "--enable_instance_cache",
            "--instance_cache_alpha_warmstart",
            "--instance_cache_branching_hints",
            "--instance_cache_path",
            cache_path,
        ]
    if variant == "cache_strict":
        return [
            "--enable_instance_cache",
            "--instance_cache_alpha_warmstart",
            "--instance_cache_branching_hints",
            "--instance_cache_strict_recompute",
            "--instance_cache_path",
            cache_path,
        ]
    raise ValueError(f"Unknown variant: {variant}")


def _safe_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_metrics(text: str) -> Dict[str, object]:
    result_matches = RESULT_RE.findall(text)
    final_status = result_matches[-1][0] if result_matches else None
    final_status_time_sec = _safe_float(result_matches[-1][1]) if result_matches else None

    alpha_times = [_safe_float(x) for x in ALPHABETA_TIME_RE.findall(text)]
    alpha_times = [x for x in alpha_times if x is not None]

    domains = [int(x) for x in DOMAINS_RE.findall(text)]
    glb_matches = GLOBAL_LB_RE.findall(text)
    glb_last = _safe_float(glb_matches[-1]) if glb_matches else None
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
        "global_lower_bound_last": glb_last,
        "cache_status_last": cache_hits[-1] if cache_hits else None,
        "cache_statuses": ",".join(cache_hits) if cache_hits else "",
        "timeout_marker_count": timeout_hits,
    }


def run_once(
    python_exec: str,
    config: str,
    device: str,
    variant: str,
    cache_path: str,
    extra_args: List[str],
    workdir: Path,
    log_file: Path,
) -> Dict[str, object]:
    cmd = [
        python_exec,
        "abcrown.py",
        "--config",
        config,
        "--device",
        device,
    ]
    cmd.extend(_variant_args(variant, cache_path))
    cmd.extend(extra_args)

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wall = time.time() - t0

    log_file.write_text(proc.stdout, encoding="utf-8")
    metrics = parse_metrics(proc.stdout)
    metrics["return_code"] = proc.returncode
    metrics["wall_time_sec"] = wall
    metrics["command"] = " ".join(shlex.quote(p) for p in cmd)
    metrics["log_file"] = str(log_file)
    return metrics


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_variant.setdefault(str(r["variant"]), []).append(r)

    summary: Dict[str, object] = {}
    for variant, items in by_variant.items():
        wall = [float(x["wall_time_sec"]) for x in items if x.get("wall_time_sec") is not None]
        dom = [int(x["domains_visited_last"]) for x in items if x.get("domains_visited_last") is not None]
        rc = [int(x["return_code"]) for x in items]
        statuses = [str(x["final_status"]) for x in items if x.get("final_status") is not None]
        summary[variant] = {
            "n_runs": len(items),
            "avg_wall_time_sec": (sum(wall) / len(wall)) if wall else None,
            "min_wall_time_sec": min(wall) if wall else None,
            "max_wall_time_sec": max(wall) if wall else None,
            "avg_domains_visited_last": (sum(dom) / len(dom)) if dom else None,
            "nonzero_return_codes": sum(1 for x in rc if x != 0),
            "status_hist": {s: statuses.count(s) for s in sorted(set(statuses))},
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cache-vs-baseline harness for abcrown.")
    parser.add_argument("--config", required=True, help="Config file path relative to complete_verifier.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Device for abcrown.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for runs.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per variant.")
    parser.add_argument(
        "--variants",
        default="baseline,cache,cache_strict",
        help="Comma-separated variants from: baseline,cache,cache_strict",
    )
    parser.add_argument("--cache-path", default=".abcrown_cache_bench", help="Cache dir passed to abcrown.")
    parser.add_argument("--extra-args", default="", help="Extra abcrown CLI args as one quoted string.")
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    parser.add_argument("--clean-cache-before", action="store_true", help="Delete cache dir before run.")
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.output_dir) if args.output_dir else (workdir / "bench_outputs" / ts)
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.cache_path
    if not os.path.isabs(cache_path):
        cache_path = str((workdir / cache_path).resolve())

    if args.clean_cache_before and os.path.isdir(cache_path):
        import shutil
        shutil.rmtree(cache_path, ignore_errors=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    extra_args = _parse_extra(args.extra_args)

    rows: List[Dict[str, object]] = []
    total_runs = len(variants) * args.repeats
    run_idx = 0
    for variant in variants:
        for rep in range(1, args.repeats + 1):
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] variant={variant} rep={rep}")
            log_file = logs_dir / f"{variant}_rep{rep}.log"
            metrics = run_once(
                python_exec=args.python,
                config=args.config,
                device=args.device,
                variant=variant,
                cache_path=cache_path,
                extra_args=extra_args,
                workdir=workdir,
                log_file=log_file,
            )
            metrics["variant"] = variant
            metrics["repeat"] = rep
            rows.append(metrics)
            print(
                f"  status={metrics.get('final_status')} rc={metrics.get('return_code')} "
                f"wall={metrics.get('wall_time_sec'):.2f}s domains={metrics.get('domains_visited_last')}"
            )

    csv_path = outdir / "runs.csv"
    fieldnames = [
        "variant",
        "repeat",
        "return_code",
        "final_status",
        "final_status_time_sec",
        "wall_time_sec",
        "num_result_lines",
        "num_safe_results",
        "num_unsafe_results",
        "num_unknown_results",
        "alpha_beta_opt_time_sum_sec",
        "alpha_beta_opt_time_max_sec",
        "alpha_beta_opt_calls",
        "domains_visited_last",
        "domains_visited_max",
        "global_lower_bound_last",
        "cache_status_last",
        "cache_statuses",
        "timeout_marker_count",
        "log_file",
        "command",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    summary = summarize(rows)
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote run CSV: {csv_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

