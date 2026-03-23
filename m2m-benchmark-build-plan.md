# M -> M' Benchmark Build Plan

## Goal

Implement the new benchmark in the smallest sequence that produces a usable
experiment quickly, while keeping the code structure extensible for additional
lineages and models.

This plan is about how to build the benchmark described in
`m2m-benchmark-plan.md`.

## Build Strategy

Build in this order:

1. benchmark layout and manifest schema
2. YAML materialization for swapping checkpoints cleanly
3. lineage generation in synthetic-noise mode
4. property selection from existing VNNLIBs
5. benchmark runner for baseline vs reuse
6. analysis and summary scripts
7. validation and hardening

The key idea is to get one full benchmark loop working before adding realism.

## Phase 1: Benchmark Layout and Schemas

### Deliverables

- directory `alpha-beta-CROWN/complete_verifier/m2m_bench/`
- `README.md`
- `configs/resnet2b_vnnlib_template.yaml`
- documented schema for:
  - `lineage.json`
  - `properties.csv`

### Tasks

1. Create the benchmark directory tree.
2. Copy the existing batch-VNNLIB config into a template config dedicated to the
   benchmark.
3. Replace hard-coded assumptions with placeholders documented in comments:
   - checkpoint path
   - VNNLIB csv path
   - output path
4. Define the lineage manifest schema.
5. Define the property manifest schema.

### Validation

- template YAML should be loadable after path substitution
- manifest schemas should be stable enough that later scripts can consume them
  without redesign

## Phase 2: YAML Materialization Helper

### Deliverables

- script or helper module to materialize a concrete config from the template for
  a given checkpoint and property subset

### Tasks

1. Create a small helper that takes:
   - template path
   - checkpoint path
   - property manifest or VNNLIB csv path
   - output path
2. Write a concrete YAML file for execution.
3. Keep the generated YAMLs in per-run output directories for reproducibility.

### Validation

- generated config runs `abcrown.py` successfully on the base checkpoint

## Phase 3: Lineage Generation MVP

### Deliverables

- `scripts/generate_lineage.py`
- one synthetic lineage for `resnet2b`
- `lineage.json`

### Tasks

1. Load `resnet2b` base checkpoint.
2. Implement synthetic perturbation mode:
   - perturb each parameter tensor with scaled Gaussian noise
   - save child checkpoints
3. After each child is produced, compute:
   - relative L2 drift
   - relative Linf drift
4. Keep children that match drift buckets.
5. Save metadata into `lineage.json`.

### Validation

- all generated checkpoints load into the verifier
- drift buckets are monotone and non-degenerate

### Notes

This phase should not depend on fine-tuning support. Fine-tuning can be added
later once the benchmark plumbing exists.

## Phase 4: Parent-Child Functional Drift Metrics

### Deliverables

- additional metadata in `lineage.json`

### Tasks

1. Evaluate parent and child on a held-out CIFAR-10 slice.
2. Compute:
   - label agreement
   - logit MSE
3. Store those values for every child checkpoint.

### Validation

- checkpoint drift metadata should expose whether a tiny parameter change caused
  large functional change

## Phase 5: Property Selection

### Deliverables

- `scripts/select_properties.py`
- `properties.csv`

### Tasks

1. Run root-model baseline on all entries in:
   `exp_configs/tutorial_examples/cifar10_resnet2b_instances.csv`
2. Parse per-property metrics from the log output:
   - status
   - wall time
   - domains visited
   - whether BaB was entered
3. Bin each property into:
   - easy
   - medium
   - hard
4. Select a balanced subset of 24 properties.
5. Write `properties.csv`.

### Validation

- selected properties must be stable and reproducible across reruns
- selected subset should contain a nontrivial number of BaB-heavy cases

## Phase 6: Benchmark Runner

### Deliverables

- `scripts/run_m2m_bench.py`
- `runs.csv`
- `summary.json`
- raw log directory

### Tasks

1. Read `lineage.json`.
2. Read `properties.csv`.
3. For each `(parent, child, property)` triple:
   - run child baseline
   - run child with reuse seeded from parent
4. Record all required metrics.
5. Write one row per run to `runs.csv`.
6. Write grouped summaries to `summary.json`.

### Required MVP variants

- `baseline`
- `alpha_only_parent_to_child`

### Optional next variants

- `branch_only_parent_to_child`
- `alpha_plus_branch_parent_to_child`
- `alpha_only_root_to_child`

### Validation

- status parity is checked automatically
- failed runs are preserved, not discarded silently

## Phase 7: Analysis Script

### Deliverables

- `scripts/analyze_m2m_bench.py`

### Tasks

1. Aggregate by:
   - reuse variant
   - difficulty bin
   - drift bucket
   - lineage
2. Report:
   - status parity rate
   - median wall time
   - speedup ratio
   - alpha optimization time change
   - domains visited change
3. Produce a short text summary that answers:
   - does alpha reuse help at all?
   - only for tiny drift or also larger drift?
   - only on easy cases or also BaB-heavy cases?

### Validation

- analysis must highlight the medium/hard slice, not only aggregate averages

## Phase 8: Validation and Hardening

### Deliverables

- benchmark sanity checks
- benchmark documentation updates

### Tasks

1. Add validation checks for:
   - missing checkpoints
   - malformed manifests
   - duplicate property IDs
   - incompatible checkpoint/model name combinations
2. Add a reproducibility note to the benchmark README.
3. Add a small sample command sequence for one minimal benchmark run.

### Validation

- a new user should be able to reproduce one benchmark run with the docs alone

## Recommended File-Level Ownership

These are the first files I would expect to add:

- `alpha-beta-CROWN/complete_verifier/m2m_bench/README.md`
- `alpha-beta-CROWN/complete_verifier/m2m_bench/configs/resnet2b_vnnlib_template.yaml`
- `alpha-beta-CROWN/complete_verifier/m2m_bench/scripts/generate_lineage.py`
- `alpha-beta-CROWN/complete_verifier/m2m_bench/scripts/select_properties.py`
- `alpha-beta-CROWN/complete_verifier/m2m_bench/scripts/run_m2m_bench.py`
- `alpha-beta-CROWN/complete_verifier/m2m_bench/scripts/analyze_m2m_bench.py`

## MVP Execution Order

The shortest path to the first useful result is:

1. create benchmark directory and template YAML
2. implement synthetic lineage generation
3. generate one lineage from `resnet2b`
4. select 24 properties from the existing VNNLIB list
5. implement benchmark runner with baseline + alpha-only
6. run one benchmark pass
7. analyze whether alpha helps on the `m -> m'` setting

This should be completed before adding:

- fine-tuning-generated children
- multiple lineages
- branch reuse variants
- `resnet4b`

## Decision Gates

After the first full pass, use these gates:

### Gate 1

If alpha-only reuse shows no benefit even on tiny drift children and hard
properties, alpha should be downgraded and branching or intermediate-bound reuse
should move up.

### Gate 2

If the benchmark machinery works but synthetic-noise children look unrealistic,
add fine-tuning lineages before drawing conclusions.

### Gate 3

If one lineage shows strong effect, add at least one more lineage before making
any strong claim.

## Recommended Immediate Next Implementation

The next coding task after these planning docs should be:

- create the benchmark directory
- add the template YAML
- implement `generate_lineage.py` in synthetic-noise mode only

That produces the first benchmark artifact with minimal dependency on new
training code.
