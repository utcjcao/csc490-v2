# Incremental Verification for Machine Learning Models

## Core idea

Incremental verification means we do not redo an entire verification proof every time a model changes a little.

In neural network verification, we usually want to prove a statement like:

"For every input in some allowed range, the model will behave safely."

For example:

- a classifier should keep the same label under a small input change
- a controller should avoid unsafe outputs for a range of sensor values

These proofs can be expensive. A verifier may spend a lot of time computing bounds, splitting cases, and ruling out bad behaviors.

If the model is updated slightly, the usual approach is to verify the new model from scratch. Incremental verification asks:

"Can we reuse part of the old verification work for the new model?"

## Intuition

A good analogy is incremental compilation.

If you change one file in a program, you usually do not rebuild everything from zero. You reuse earlier work and only recompute what is affected.

Incremental verification tries to do the same thing for proofs about neural networks.

If model version B is only a small change from model version A, then some earlier verification artifacts may still be useful, such as:

- neuron bounds
- search decisions
- abstractions
- certificates
- known counterexamples

Instead of fully re-verifying every property, the verifier tries to:

- keep the parts that are still valid
- cheaply update the parts that changed
- only rerun expensive verification where necessary

## Why this matters

Real machine learning systems change constantly. Teams often:

- fine-tune weights
- retrain on new data
- export models to ONNX
- quantize models for deployment
- change runtime providers or compiler settings

If verification only works when we restart from zero every time, then it does not fit real deployment workflows.

Incremental verification could make verification much more practical in CI/CD pipelines and model update cycles.

## Why it is hard

Small model changes do not automatically mean the old proof is still valid.

Even a small weight update can break a safety property. So the challenge is not just speed. The challenge is to reuse old verification work in a way that is still sound.

That means the system must know:

- what changed
- which old proof artifacts are still trustworthy
- which parts must be recomputed

## Thesis version of the idea

A strong thesis project would study incremental verification across the machine learning deployment pipeline, not just across small training updates.

For example:

1. verify a base model and store useful proof artifacts
2. update the model or export it to a new format
3. reuse as much earlier verification work as possible
4. re-verify only the affected parts
5. measure both speedup and correctness

## Example research question

Can we safely reuse verification artifacts across model updates and deployment transformations to make neural network verification significantly faster without losing soundness?

## Possible scope

A practical scope would be:

- start with one verifier
- use one model family
- use one property type, such as local robustness
- evaluate across checkpoint updates, ONNX export changes, or quantization steps

## Why this is a good thesis topic

This idea is appealing because it is:

- practically useful
- technically nontrivial
- connected to real deployment workflows
- less saturated than another small improvement to a standard robustness verifier

It also gives clear evaluation criteria:

- how much verification time is saved
- when reuse is safe
- when reuse fails
- whether the final guarantees remain sound
