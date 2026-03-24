# M -> M' Benchmark

This directory contains the lineage-oriented benchmark for evaluating whether
verification artifacts from model `m` accelerate verification of a nearby model
`m'` on the same property.

The benchmark unit is:

- source model `m`
- target model `m'`
- fixed VNNLIB property `p`

Every comparison keeps `p` fixed and changes only the model checkpoint plus the
reuse mode.

## Directory Layout

- `configs/resnet2b_vnnlib_template.yaml`: template config used to materialize
  per-run verifier configs.
- `scripts/generate_lineage.py`: create one synthetic checkpoint lineage and
  write `lineage.json`.
- `scripts/select_properties.py`: score candidate VNNLIB properties on the root
  model and write `properties.csv`.
- `scripts/run_m2m_bench.py`: run baseline plus reuse variants across
  `(source, target, property)` triples.
- `scripts/analyze_m2m_bench.py`: summarize benchmark output from `runs.csv`.

## Manifest Schemas

### `lineage.json`

Top-level fields:

- `schema_version`
- `lineage_id`
- `model_name`
- `generation_mode`
- `base_checkpoint`
- `drift_targets_rel_l2`
- `models`

Each record inside `models` contains:

- `model_id`
- `parent_id`
- `checkpoint_path`
- `lineage_id`
- `generation_mode`
- `seed`
- `drift_bucket`
- `target_drift_rel_l2`
- `delta_w_rel_l2`
- `delta_w_rel_linf`
- `delta_w_rel_l2_root`
- `delta_w_rel_linf_root`
- `delta_w_rel_l2_parent`
- `delta_w_rel_linf_parent`
- `label_agreement_parent`
- `logit_mse_parent`

`delta_w_rel_l2` and `delta_w_rel_linf` are root-relative for sliceing by drift
bucket. Parent-relative drift is stored separately.

### `properties.csv`

Columns:

- `property_id`
- `property_index`
- `vnnlib_path`
- `difficulty_bin`
- `baseline_status_root`
- `baseline_time_root_sec`
- `domains_root`
- `entered_bab_root`
- `timeout_marker_count_root`

## Reproducibility

- Generated YAMLs, single-property CSVs, logs, and summaries are written into
  the benchmark output directory.
- Benchmark scripts use the verifier CLI (`abcrown.py`) rather than a separate
  API path so that benchmark runs match normal verifier usage.
- Cross-model reuse is isolated behind `--instance_cache_model_group`. This is
  for controlled experiments only; it is not meant to imply that different
  checkpoints are interchangeable.

## Minimal Command Sequence

Assuming you are inside `alpha-beta-CROWN/complete_verifier/` and using the
environment that already runs `abcrown.py`:

```bash
python m2m_bench/scripts/generate_lineage.py \
  --base-checkpoint models/cifar10_resnet/resnet2b.pth \
  --output-dir m2m_bench/outputs/lineage_v1

python m2m_bench/scripts/select_properties.py \
  --output-dir m2m_bench/outputs/property_scan_v1

python m2m_bench/scripts/run_m2m_bench.py \
  --lineage-manifest m2m_bench/outputs/lineage_v1/lineage.json \
  --properties-csv m2m_bench/outputs/property_scan_v1/properties.csv \
  --output-dir m2m_bench/outputs/run_v1

python m2m_bench/scripts/analyze_m2m_bench.py \
  --runs-csv m2m_bench/outputs/run_v1/runs.csv
```
