# M -> M' Benchmark Plan

## Goal

Build a benchmark that measures whether verification artifacts from model `m`
can accelerate verification of a nearby model `m'` on the same verification
property.

The benchmark must answer a narrower question than the current instance-cache
experiments:

- Current instance-cache question: can artifacts from one verification instance
  help another instance on the same model?
- New benchmark question: can artifacts from verifying `(m, p)` help verify
  `(m', p)` when `m'` is a nearby checkpoint and `p` is unchanged?

## Core Principle

The benchmark unit is:

- parent model `m`
- child model `m'`
- fixed property `p`

Every result should compare:

- baseline verification of `(m', p)` with no reuse
- reuse-enabled verification of `(m', p)` seeded from `(m, p)`

The benchmark must keep `p` fixed between `m` and `m'`. Otherwise model drift
and property drift are confounded.

## Scope Lock for V1

V1 should stay intentionally small:

- one model family: `resnet2b`
- one property family: existing CIFAR-10 ResNet-2B VNNLIB properties
- one lineage generation mode required for MVP: synthetic weight drift
- one optional lineage generation mode after MVP: lightweight fine-tuning
- one benchmark slice emphasized in reporting: properties that actually enter
  branch-and-bound
- one reuse target required for MVP: `alpha` warm-start

This is enough to get real signal without overbuilding the benchmark platform.

## Why One Lineage Is Enough to Start

One linear lineage from a single base model is enough for:

- building the benchmark machinery
- validating the manifest format
- testing parent-to-child reuse end-to-end
- getting an initial answer on whether `alpha` warm-start is promising for
  `m -> m'`

One lineage is not enough for:

- claiming the effect generalizes
- separating a real reuse effect from lineage-specific noise
- deciding that a reuse strategy is thesis-worthy

Therefore the benchmark should be staged:

- Stage 1: 1 base model, 1 lineage, 4-5 child checkpoints
- Stage 2: same base model, 2-3 independent lineages
- Stage 3: add `resnet4b`

## Assets Already Present in the Repo

The benchmark should reuse existing verified-model assets first:

- base config:
  `alpha-beta-CROWN/complete_verifier/exp_configs/tutorial_examples/cifar_resnet_2b.yaml`
- batch VNNLIB config pattern:
  `alpha-beta-CROWN/complete_verifier/exp_configs/tutorial_examples/pytorch_model_with_batch_vnnlib.yaml`
- property list:
  `alpha-beta-CROWN/complete_verifier/exp_configs/tutorial_examples/cifar10_resnet2b_instances.csv`
- base checkpoint:
  `alpha-beta-CROWN/complete_verifier/models/cifar10_resnet/resnet2b.pth`

These make `resnet2b` the best V1 benchmark target.

## Benchmark Dataset Structure

The benchmark should be organized as a lineage dataset rather than a flat list
of instances.

### Model Side

For each lineage:

- `m0`: base checkpoint
- `m1`, `m2`, `m3`, `m4`: increasingly drifted child checkpoints

Each child checkpoint should carry metadata:

- `model_id`
- `parent_id`
- `checkpoint_path`
- `lineage_id`
- `generation_mode`
- `seed`
- `delta_w_rel_l2`
- `delta_w_rel_linf`
- `label_agreement_parent`
- `logit_mse_parent`

### Property Side

Each property record should carry:

- `property_id`
- `vnnlib_path`
- `difficulty_bin`
- `baseline_status_root`
- `baseline_time_root_sec`
- `domains_root`

## Child Model Generation

V1 should support two generation modes.

### Required for MVP: Synthetic Weight Drift

Generate child checkpoints by perturbing the parent weights.

Advantages:

- easy to implement
- deterministic
- fast to generate
- enough to test whether reuse is sensitive to nearby parameter changes

Disadvantages:

- less realistic than continued training
- may create function drift that does not resemble normal model evolution

### Optional After MVP: Lightweight Fine-Tuning

Continue training from the base checkpoint with low learning rate and save
checkpoints at increasing drift targets.

Advantages:

- closer to the real `m -> m'` story
- better external validity

Disadvantages:

- requires more engineering and more assumptions about training setup

## Drift Buckets

Child checkpoints should be selected by measured drift rather than raw training
steps.

Suggested buckets:

- tiny: `1e-4`
- small: `5e-4`
- moderate: `1e-3`
- larger: `5e-3`

Each checkpoint should be the first one that reaches or exceeds the target
bucket.

## Property Selection

The existing ResNet-2B VNNLIB list contains enough candidate properties, but V1
should not use all of them blindly.

Instead:

1. run root-model baseline on all available properties
2. stratify them into:
   - easy
   - medium
   - hard
3. select a balanced subset for repeated experiments

Suggested V1 subset size:

- 24 total properties
- 8 easy
- 8 medium
- 8 hard

Headline reporting should emphasize medium and hard properties, because trivial
properties can hide whether reuse is actually helping.

## Benchmark Variants

V1 should support these reuse variants:

- `baseline`
- `alpha_only_parent_to_child`
- `branch_only_parent_to_child`
- `alpha_plus_branch_parent_to_child`
- `alpha_only_root_to_child`

Only `baseline` and `alpha_only_parent_to_child` are required for the first
usable version.

## Evaluation Axes

Every benchmark result should be sliceable by:

- lineage
- child drift bucket
- property difficulty bin
- reuse variant
- reuse source:
  - parent-to-child
  - root-to-child

This will allow us to distinguish:

- whether reuse helps only for tiny drifts
- whether reuse helps only on BaB-heavy properties
- whether immediate-parent artifacts are better than older artifacts

## Metrics

Minimum per-run metrics:

- `source_model_id`
- `target_model_id`
- `lineage_id`
- `property_id`
- `reuse_variant`
- `reuse_source`
- `status`
- `return_code`
- `wall_time_sec`
- `final_status_time_sec`
- `alpha_beta_opt_time_sum_sec`
- `alpha_beta_opt_calls`
- `domains_visited_last`
- `domains_visited_max`
- `entered_bab`
- `timeout_marker_count`

Additional lineage-aware metrics:

- `delta_w_rel_l2`
- `delta_w_rel_linf`
- `label_agreement_parent`
- `logit_mse_parent`
- `difficulty_bin`

Reuse-specific metrics when available:

- `alpha_loaded`
- `alpha_reset`
- `branch_hint_hits`
- `cache_status_last`
- `fallback_count`

## Correctness Rule

No speedup should be counted unless:

- `status_reuse == status_baseline`

If statuses differ:

- mark the run as unusable for performance comparison
- investigate whether the difference is a bug, a timeout instability, or a real
  unsoundness issue

## Directory Layout

The benchmark should live under:

- `alpha-beta-CROWN/complete_verifier/m2m_bench/`

Suggested structure:

- `README.md`
- `configs/`
- `manifests/`
- `lineages/`
- `scripts/`
- `outputs/`

Suggested files:

- `configs/resnet2b_vnnlib_template.yaml`
- `scripts/generate_lineage.py`
- `scripts/select_properties.py`
- `scripts/run_m2m_bench.py`
- `scripts/analyze_m2m_bench.py`

## Output Artifacts

Each benchmark run should produce:

- per-run raw log files
- `runs.csv`
- `summary.json`
- one immutable copy of the lineage manifest used
- one immutable copy of the property manifest used

This is required for reproducibility.

## Stage Plan

### Stage 1

Build the minimum viable benchmark:

- `resnet2b`
- one lineage
- 24 properties
- baseline + alpha-only

### Stage 2

Strengthen internal validity:

- 2-3 independent lineages
- same model family
- baseline + alpha-only + branch-only + combined

### Stage 3

Check portability:

- add `resnet4b`
- compare whether the reuse effect survives a larger architecture in the same
  family

## Success Criteria

The benchmark is successful when:

- it can generate a lineage manifest reproducibly
- it can select a stable property subset
- it can run baseline and reuse experiments on fixed `(m, m', p)` triples
- it writes lineage-aware CSV outputs
- it separates easy, medium, and hard property slices
- it can answer whether alpha reuse is promising specifically for `m -> m'`

## Non-Goals for V1

V1 should not try to do all of the following:

- multiple verifier backends
- multiple datasets
- multiple property languages
- realistic CI/CD semantics changes
- training-time benchmark generation for many architectures
- paper-ready dashboarding

If V1 can answer the `resnet2b` question cleanly, it is already useful.
