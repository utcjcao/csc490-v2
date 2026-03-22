# Alpha Reuse Evaluation (Repo-Specific)

## Why Alpha Reuse Underperformed
- **Instance/domain specificity**: Alpha tensors are indexed by current unstable masks and intervals. Small changes in bounds, masks, or spec reshape alpha and change optima.
- **No sound skip**: Soundness requires recomputing bounds with current masks; reused alpha can only be a warm start, never a certified shortcut.
- **Observed cost/benefit**: On CIFAR resnet2b (`--start 0 --end 3`), alpha-warmstart (final-only scope, sim gate 0.9) ran ~5% slower than baseline (≈86s vs 82s). Earlier full runs (~40+ min) also saw no improvement and sometimes more domains.
- **Existing intra-run reuse**: The solver already reuses alpha across batches/specs within a run; cross-instance reuse adds overhead but little headroom.
- **Shape fragility**: Different OR specs or BaB splits change alpha shapes; cached tensors often need pruning/expansion, eroding any speedup.
- **Dominant costs elsewhere**: BaB search and bound recomputation dominate runtime; shaving a few alpha iterations is negligible compared to domain expansion.

## Conclusion
- Alpha reuse is only a heuristic warm start and showed neutral-to-negative impact in practice. It is not a promising primary direction for incremental speedups in this codebase.

## Next Experiment to Close Alpha Reuse
- One final A/B: harder slice that enters BaB, last-layer-only alpha warm start, similarity ≥0.95, compare domains/time vs baseline. If no gain, retire alpha reuse.

## What to Try Instead (Higher-ROI Reuse Targets)
- **Intermediate bounds / lA / masks**: cache and reuse between close instances or BaB siblings to avoid recomputation.
- **Branching artifacts**: reuse nogoods/pruned splits or layer priority stats to cut domains.
- **Cut constraints**: persist inferred cuts (GCP) across nearby specs.
- **Neuron stability info**: cache stable neuron sets to reduce split/opt workload.
- **Verified subproblems**: memoize solved subtrees for repeated specs/runs.
