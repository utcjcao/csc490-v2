# Incremental Study

This folder contains a standalone experimental harness for studying intrinsic
verification similarity between an original model `M` and a perturbed model
`M'`.

It is intentionally separate from:
- the existing `alpha-beta-CROWN` pipeline
- the existing `IVAN` pipeline

The harness uses a vendored local copy of `auto_LiRPA` through a thin adapter,
and does not modify or route through the existing abcrown or IVAN experiment
drivers.

## Scope

V1 focuses on independent root-stage robustness verification:
- load `M`
- create `M'`
- run root-stage verification on `M` from scratch
- run root-stage verification on `M'` from scratch
- log comparable internal artifacts for both runs

This is not proof transfer.

The `M` run never influences the `M'` run.

## Installation

This standalone study does **not** require:
- installing Gurobi
- using the IVAN experiment harness
- using the abcrown experiment harness

The `auto_LiRPA` backend and the model architecture code are already vendored
inside this folder:
- [vendor/auto_LiRPA/](/Users/chris/Developer/school/csc490-v2/incremental_study/vendor/auto_LiRPA)
- [models/](/Users/chris/Developer/school/csc490-v2/incremental_study/models)

So the setup work is mainly:
1. create a Python environment
2. install runtime Python dependencies
3. download model weight files
4. run the paired study from the repo root

### 1. Create and activate an environment

Example with `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install Python dependencies

Install the core dependencies used by the standalone harness and vendored
`auto_LiRPA` backend:

```bash
pip install -r incremental_study/requirements.txt
```

If you want to load `.onnx` models, also install:

```bash
pip install "onnx>=1.8" "onnx2pytorch>=0.4.1"
```

Notes:
- `torch` should be installed with the build appropriate for your machine, especially on a remote GPU box.
- You can install the correct CUDA-enabled PyTorch build first from the official PyTorch install instructions, then run the remaining `pip install ...` command above.
- No separate `pip install` step is required for the vendored `auto_LiRPA` copy because this harness imports it from the local repository.
- The pinned runtime dependency list lives in [requirements.txt](/Users/chris/Developer/school/csc490-v2/incremental_study/requirements.txt).

### 3. Download model weights

The model architecture code is already included locally, but the actual model
weights are not.

If you want to use the model assets referenced by IVAN:
- download the checkpoint files from the Google Drive link in [IVAN/README.md](/Users/chris/Developer/school/csc490-v2/IVAN/README.md#L8)
- place them somewhere local, for example:
  - `incremental_study/assets/models/`
  - or any other path on the machine

Example:

```text
incremental_study/assets/models/
  mnist-net_256x2.onnx
  mnist_conv_small_nat.pth
  cifar_base_kw.pth
```

No separate installation step is required for the vendored model definitions in
[models/](/Users/chris/Developer/school/csc490-v2/incremental_study/models). They are imported directly by the harness.

### 4. Run from the repository root

Run commands from the repository root:

```bash
cd /path/to/csc490-v2
python -m incremental_study.paired_runner ...
```

Running from the repo root ensures Python resolves the local
`incremental_study` package correctly.

## What Gets Logged

Per property and per run, the harness writes structured JSON including:
- run metadata
- center-input behavior
- root lower/upper bound summaries
- unstable ReLU counts per layer
- top root candidate split neurons based on LiRPA ReLU bound-improvement scores
- per-layer bound summaries
- final verification status and runtime

For the perturbed model, the pair manifest also includes:
- actual layerwise weight delta summaries
- overall relative weight drift
- fraction of changed weights
- fraction of zeroed weights
- perturbation-specific metadata

## Current Limitation

V1 uses plain `auto_LiRPA`, not a separate branch-and-bound loop. That means:
- `split_trace` is explicitly marked unavailable
- `search_summary` is explicitly marked unavailable

The harness records root-stage candidate split information only. It does not
fake a search trace.

## Model Loading

The loader supports:
- PyTorch checkpoints (`.pt`, `.pth`) using a vendored local copy of the IVAN model registry
- ONNX models (`.onnx`) if `onnx` and `onnx2pytorch` are installed

Expected use with IVAN assets:
- download the model files from the Google Drive link referenced in [IVAN/README.md](/Users/chris/Developer/school/csc490-v2/IVAN/README.md#L8)
- point `--model-path` at one of those local files

For `.pth` / `.pt` checkpoints, pass `--model-arch` when the checkpoint stem
does not match the vendored architecture registry key.

## Property Definition

V1 uses dataset-indexed local robustness properties:
- dataset image at index `i`
- true label from the dataset
- Linf perturbation radius `eps`

The same set of dataset indices is used for both `M` and `M'`.

The dataset loader will download MNIST or CIFAR-10 automatically into the
specified `--data-root` directory if the data is not already present.

## Choosing a Model Path

### PyTorch checkpoints: `.pt` / `.pth`

For PyTorch checkpoints, the harness reconstructs the model architecture from
the vendored local registry in [models/__init__.py](/Users/chris/Developer/school/csc490-v2/incremental_study/models/__init__.py).

If the checkpoint filename stem already matches the registry key, you can often
omit `--model-arch`.

If it does not, pass `--model-arch` explicitly.

Examples:
- `--model-arch mnist_conv_small_nat`
- `--model-arch mnist_6_100_nat`
- `--model-arch vnncomp_resnet2b`

### ONNX models: `.onnx`

For ONNX models, the harness converts the model to PyTorch using
`onnx2pytorch`. In this case, `--model-arch` is not needed.

## Example

```bash
python -m incremental_study.paired_runner \
  --model-path /path/to/ivan_model.onnx \
  --dataset mnist \
  --eps 0.03 \
  --count 5 \
  --start-index 0 \
  --device cuda \
  --method CROWN-Optimized \
  --perturbation random_noise \
  --random-std 1e-3 \
  --random-relative \
  --output-dir incremental_study_runs
```

Quantization example:

```bash
python -m incremental_study.paired_runner \
  --model-path /path/to/ivan_model.pth \
  --model-arch mnist_conv_small_nat \
  --dataset mnist \
  --eps 0.03 \
  --indices 0,1,2 \
  --device cuda \
  --method CROWN-Optimized \
  --perturbation quantize \
  --quant-bits 8 \
  --output-dir incremental_study_runs
```

Pruning example:

```bash
python -m incremental_study.paired_runner \
  --model-path incremental_study/assets/models/cifar_base_kw.pth \
  --model-arch cifar_base_kw \
  --dataset cifar10 \
  --eps 0.00784313725 \
  --count 3 \
  --device cuda \
  --method CROWN-Optimized \
  --perturbation prune \
  --prune-fraction 0.05 \
  --output-dir incremental_study_runs
```

## Common CLI Arguments

Required:
- `--model-path`: checkpoint or ONNX file
- `--dataset`: `mnist` or `cifar10`
- `--eps`: Linf perturbation radius

Common optional:
- `--model-arch`: architecture key for `.pt` / `.pth` checkpoints
- `--count`: number of properties to run if using contiguous indices
- `--start-index`: first dataset index for contiguous selection
- `--indices`: explicit comma-separated dataset indices
- `--device`: `cpu` or `cuda`
- `--method`: LiRPA method such as `CROWN-Optimized`
- `--perturbation`: `random_noise`, `quantize`, or `prune`
- `--output-dir`: where logs are written
- `--top-k`: how many root candidates to retain in the logs

Perturbation-specific:
- `random_noise`:
  - `--random-std`
  - `--random-relative`
- `quantize`:
  - `--quant-bits`
- `prune`:
  - `--prune-fraction`

## Output Layout

Each run creates a new pair directory:

```text
incremental_study_runs/
  <pair_id>/
    pair_manifest.json
    original/
      <property_id>.json
    perturbed/
      <property_id>.json
```

## Notes

- The statuses are:
  - `VERIFIED`: all LiRPA margin lower bounds are positive
  - `MISS_CLASSIFIED`: the center input is already misclassified
  - `UNKNOWN`: root-stage LiRPA could not certify robustness
  - `ERROR`: runtime/import failure inside the harness
- No proof transfer, template store, caching, or cross-run reuse is implemented here.
- The harness currently logs root-stage artifacts only. It does not produce a BaB search trace.

## Quick Start

If you just want the shortest working setup:

1. Create a Python environment.
2. Run `pip install -r incremental_study/requirements.txt`.
3. Optionally install `onnx` and `onnx2pytorch` if your model is ONNX.
4. Download one IVAN model checkpoint locally.
5. Run:

```bash
python -m incremental_study.paired_runner \
  --model-path /path/to/model \
  --dataset mnist \
  --eps 0.03 \
  --count 5 \
  --device cuda \
  --method CROWN-Optimized \
  --perturbation random_noise \
  --random-std 1e-3 \
  --random-relative
```
