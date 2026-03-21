# Verification Reuse TODOs (Tomorrow Plan)

## Goal
Find 3-4 strong reuse optimizations for alpha-beta-CROWN, with measurable speedups and no correctness regressions, suitable for capstone paper results.

## Priority Order
1. Get stable GPU execution.
2. Build repeatable benchmark harness.
3. Lock baseline metrics.
4. Add one doptimization at a time.
5. Keep only optimizations with status parity and measurable gains.

## 1) GPU Bring-Up (First 60-90 min)
1. Verify CUDA availability:
   - `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
2. Run one known config on GPU:
   - `python abcrown.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml --device cuda`
3. Save baseline log and config dump.
4. Confirm run is stable (no CUDA OOM / environment errors).

## 2) Build Benchmark Harness (Before More Coding)
Create a small script to run repeated experiments and write CSV metrics.

Track per instance:
1. `instance_id`
2. `status` (safe/unsafe/unknown)
3. `wall_time_sec`
4. `incomplete_stage_time_sec` (if available)
5. `alpha_beta_opt_time_sec`
6. `domains_visited`
7. `timed_out`
8. `cache_hit_type` (`miss`, `hit-model`, `hit-exact`)

## 3) Baseline + Cache Matrix
Run each setup with at least 3 repeats on the same instance set.

1. Baseline (no cache)
2. V1 cache:
   - `--enable_instance_cache --instance_cache_alpha_warmstart --instance_cache_branching_hints`
3. V1 strict mode:
   - add `--instance_cache_strict_recompute`

Hard gate: final statuses must match baseline.

## 4) Optimization Candidates (Target 3-4 Total)

### Optimization A: Exact-Spec Alpha Warm-Start
1. Reuse cached alpha only on exact spec match.
2. Expected gain: lower alpha optimization time.
3. Compare warm-start on/off.

### Optimization B: Model-Level Gated Alpha Transfer
1. On exact miss, reuse limited alpha subset (last layers only) for same model signature.
2. Add compatibility gate (shape/spec metadata checks).
3. Measure fallback rate and speedup.

### Optimization C: Branching Priority Reuse
1. Use cached branching-layer stats to bias candidate selection.
2. Keep policy conservative initially.
3. Measure domains visited and BaB time.

### Optimization D: Conflict/Nogood Reuse
1. Cache patterns that frequently prune/infeasible quickly.
2. Deprioritize those paths on later instances (heuristic-only).
3. Measure prune ratio, rounds, domains visited.

## 5) Implementation Order (Fastest Path)
1. Harness + CSV logging.
2. Finish and evaluate Optimization A.
3. Implement and evaluate Optimization B.
4. Implement and evaluate Optimization C.
5. Implement and evaluate Optimization D if time allows.

## 6) Paper-Ready Outputs
1. A/B table per optimization alone.
2. Combined optimization table.
3. Status parity table vs baseline.
4. Runtime and domains-visited plots.
5. Cache hit-rate analysis (`exact` vs `model-level`) and fallback counts.

## 7) Practical Rules While Iterating
1. One code change at a time.
2. Keep experiment seeds fixed.
3. Save raw logs for every run.
4. Reject any optimization that changes final status semantics.
5. Prefer conservative heuristics first, then aggressive variants.

