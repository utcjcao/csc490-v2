# Instance Caching V1 (alpha-beta-CROWN)

## Goal
Speed up verification for future instances on the same model/config by reusing low-risk artifacts as warm starts and heuristics.

## Scope (V1)
- Reuse across instances in the same benchmark family.
- No cross-model proof reuse in V1.
- No reuse of final prune/proof decisions.

## V1 Artifacts To Cache
1. `alpha` warm-start state
- Source: incomplete verifier output (`ret["alphas"]`).
- Use: initialize alpha optimization in later instances.
- Safety: warm-start only; always re-optimize.

2. Branching heuristic cache
- Source: per-instance branch/search outcomes (`bab_ret`, unstable summaries from masks).
- Use: prioritize split candidates/layers likely to prune faster.
- Safety: heuristic-only; does not affect soundness.

3. Optional selective root bounds (phase 1.5, guarded)
- Source: `ret["lower_bounds"]`, `ret["upper_bounds"]` at root/near-root.
- Use: initialization/reference only for faster convergence.
- Safety: must recompute before prune/proof decisions.

## Out of Scope (V1)
- Reusing full BaB trees as proof.
- Reusing final statuses (`safe`, `unsafe`, `unknown`) as truth.
- Reusing deep-node tensor caches broadly.
- Cross-run/cross-model direct bound validity assumptions.

## Safety Rules
1. Cached artifacts are hints, not certificates.
2. Always recompute current bounds before pruning domains.
3. Keep `--no_skip_with_refined_bound` enabled when testing cross-instance bound reuse.
4. Never accept cached verdicts without recomputation.

## Why These Artifacts
- `alpha` warm-start has high upside and low correctness risk.
- Branching hints are cheap, transferable, and soundness-neutral.
- Root-level bound init may help but is riskier; include only with strict recompute policy.

## Minimal Data Schema (Draft)
- Cache key:
  - model hash
  - config hash (norm, epsilon, verifier mode, key solver settings)
  - spec signature (e.g., class target pattern / vnnlib structure class)
- Cached payload:
  - alpha tensors (selected layers/nodes)
  - branching stats (layer/node hit rates, prune contribution)
  - optional root bounds summary tensors
  - metadata (created_at, hit_count, avg_gain_ms)

## Integration Points
1. After incomplete stage:
- Capture `ret["alphas"]` and selected summary metrics.

2. Before incomplete/complete optimization for next instance:
- Retrieve nearest compatible cache entry.
- Inject alpha init and branching priority hints.

3. During complete stage:
- Log domain count, prune ratio, and per-instance runtime for A/B comparison.

## Evaluation Plan (V1)
1. Baseline run (no cache).
2. Alpha-only cache run.
3. Alpha + branching-hint run.
4. Measure:
- per-instance wall time
- BaB domains visited
- prune ratio
- final status parity vs baseline (must match).

## Success Criteria
- No soundness regressions (status parity with baseline on test set).
- Measurable runtime reduction on similar instances.
- Graceful fallback to baseline when cache miss or low benefit.
