# Verifying Neural Networks in 2026: A Critical Research Synthesis and Gap Analysis

## Executive summary

Neural-network verification has matured into a technically sophisticated ecosystem of _problem formulations_ (mostly universal-quantification over an input set), _solver backends_ (SMT/MILP/SAT-based and optimization-based), and _approximation stacks_ (abstract domains, convex relaxations, bound propagation) that can deliver strong guarantees—**but only for carefully constrained semantics and property classes**. A widely used formalization is: given a network \(f:X\to Y\), an input region \(D\subseteq X\), and a (typically linear) output property \(P\subseteq Y\), prove \(\forall x\in D,\ f(x)\in P\). citeturn25view0turn8view7

### What the field currently does well

The strongest “wins” are in **verification of feed-forward, mostly piecewise-linear networks** (especially ReLU) against **local robustness properties** (e.g., \(\ell\_\infty\) balls around a given input) when networks are in the low–to–medium size regime and the semantics are close to real arithmetic. This is the regime where _complete_ methods (sound + complete) can sometimes succeed and where _incomplete but sound_ certification is often practical. citeturn25view0turn24view0turn21view0

A second area of genuine maturity is **competition-driven standardization**: VNN-COMP has pushed a de facto tool interface around **ONNX** model exchange and **VNN-LIB** property files, along with an automated benchmarking pipeline and consistent scoring/penalty schemes. This has made cross-tool comparison far more realistic than the earlier fragmentation era. citeturn8view1turn8view0turn8view7turn6search3turn10view0turn22search24

A third area that is becoming operational (but remains narrower than the hype) is **verification frameworks** and “glue” that reduce format friction—e.g., DNNV’s property DSL and backend dispatch, and recent work compiling higher-level specs down into VNN-LIB query trees. These efforts directly address the _specification/interface bottleneck_. citeturn22search2turn22search3turn22search1turn27view0turn22search24turn8view1

Finally, the applied frontier is expanding beyond “toy MNIST robustness” into **more diverse benchmarks**: vision transformers, generative models, object detection components, system-oriented models, and closed-loop verification attempts appear in recent VNN-COMP benchmark rosters and follow-on papers. citeturn8view7turn16search3turn2search3turn2search15

### Where the real bottlenecks are

The bottlenecks are **less about solver cleverness per se** and more about _semantic scope_, _specification realism_, and _stack integration_:

**Semantic mismatch (the “real arithmetic fantasy”)** is now widely recognized as central. Many verifiers prove properties under idealized semantics (exact reals; deterministic evaluation), while deployment executes **floating-point** with compiler graph rewrites, fused kernels, mixed precision, nondeterministic GPU primitives, and runtime-dependent batching/tactic selection. “Verification is NP-complete” is the _theoretical_ limit, but “verification is not even about the artifact you ship” is increasingly the _practical_ limit. citeturn24view0turn1search27turn12search11turn19search1turn19search7turn19search2turn8view1turn1search19

**Specification poverty** is another deep bottleneck: VNN-LIB is a satisfiability-query format over network I/O variables, which is great for solver interoperability but poor for expressing many system-level requirements (temporal, relational/hyperproperties, multi-network composition, distributional goals). Even high-level DSLs often collapse back into solver limitations (templates, restricted logic fragments), and the common workaround—merging multiple network copies into one ONNX graph—creates its own semantic/engineering pathologies. citeturn22search24turn27view0turn22search2turn8view7turn22search8

**Scalability and “architecture reality”**: complete methods still struggle as you approach modern-scale CNNs/Transformers/OD pipelines, especially once you include operations like softmax, normalization, complex post-processing (NMS/IoU), and quantization effects. There is progress (e.g., object detection robustness verification frameworks, ViT benchmarks), but it is not yet routine and remains benchmark-fragmented. citeturn8view7turn16search3turn6search24

**Engineering-heavy gaps** (format/tooling/runtime integration) dominate real deployments: export fidelity, opset/versioning, custom operators, precision loss in serialized attributes, inference runtime graph partitioning, and serving-system nondeterminism. These are under-researched relative to solver algorithmics because they look “unsexy,” but they are where guarantees most often fail. citeturn1search21turn19search34turn19search2turn19search18turn8view1turn1search27

### Oversaturated vs underexplored problem types

**Oversaturated (high publication volume, diminishing marginal returns):**
Local robustness verification/certification for ReLU classifiers under \(\ell*p\) balls—especially \(\ell*\infty\)—on a relatively small set of benchmark families. This subfield is methodologically rich and still valuable, but it is also where “+3% cactus plot” papers proliferate, often via incremental bounding heuristics tuned to benchmark distributions. citeturn21view0turn25view0turn6search3turn8view7

**Underexplored (high leverage, hard, but not yet overcrowded):**

1. end-to-end _deployment semantics_ (floating-point + compiler/runtime/hardware) with **machine-checkable guarantees**;
2. specification languages and compilation pipelines that scale beyond VNN-LIB’s expressivity without sacrificing solver interoperability;
3. verification of quantized/mixed-precision models and their _exact_ integer/bit-level behavior;
4. compositional verification across pipelines/serving systems;
5. verification of non-classification workloads (generation, detection, ranking, control-in-the-loop) with meaningful specs and benchmarks. citeturn1search27turn12search11turn20search0turn20search3turn8view7turn27view0turn2search3turn6search14

### A blunt bottom line

If you define “verification” as _sound guarantees about what is actually deployed_, the field is **not there yet** for most real systems. The strongest guarantees today usually apply to an _idealized network model_ (often piecewise-linear), checked against _simple input-set properties_. That is still scientifically valuable—but the research frontier that matters in 2026 is increasingly about **closing gaps between mathematical verification objects and deployed ML artifacts**. citeturn1search27turn12search11turn8view1turn19search7turn19search1turn19search2

## Landscape map

This map is organized by what is being verified and where (model-only vs system-in-loop vs deployed artifact), and for each category it distinguishes _formal verification_ from _testing/falsification_ and _runtime monitoring_.

### Constraint solving for feed-forward networks

What is being verified: universal-input properties of a fixed network (typically classification robustness or simple safety constraints) under a mathematical semantics. Canonical formulation: prove \(f(x)\in P\) for all \(x\in D\). citeturn25view0

Assumptions: static graph; known activations/layers; semantics usually real arithmetic (or “as-if real”); solver sees a precise encoding of all relevant ops. citeturn25view0turn24view0turn1search27

Methods/tools: SMT-style (Reluplex lineage; Marabou), MILP/MIP encodings, and DPLL(T)-inspired hybrids (NeuralSAT), as well as BaB + convex relaxations/bound propagation (α/β-CROWN style). citeturn24view0turn22search0turn17search5turn17search31turn25view0turn6search3

Benchmarks: ACAS Xu (historically), MNIST/CIFAR robustness families, and increasingly VNN-COMP suites. citeturn24view0turn8view7turn6search3turn10view0

Main limitations: NP-completeness and combinatorics; heavy dependence on piecewise-linear structure; weak support for non-ReLU ops unless relaxed; and (increasingly critical) mismatch to deployed float/runtime semantics. citeturn24view0turn1search27turn12search11

### Abstract interpretation and certified (incomplete) robustness

What is being verified: sound _over-approximations_ of reachable output ranges for an input set; this yields sound certificates (no false “verified”), but may return “unknown” due to over-approximation looseness. citeturn25view0turn17search10turn17search11turn2search0

Assumptions: depends on abstract domain; typically assumes deterministic arithmetic; can support more activations via abstract transformers but often at the cost of precision/performance. citeturn17search10turn17search11turn25view0turn21view0

Methods/tools: DeepPoly-style abstract domains; ERAN-style analyzers; GPU-accelerated polyhedral verification/relaxation pipelines; and LiRPA as a generalized bound propagation framework used both in verification and training. citeturn17search10turn17search11turn2search9turn2search13

Benchmarks: adversarial robustness benchmarks (MNIST/CIFAR), competition suites, and some emerging non-standard tasks. citeturn17search11turn8view7turn6search3

Main limitations: looseness can dominate; sound but uninformative results are common on hard instances; the research culture sometimes optimizes for benchmark-tightness rather than spec realism. citeturn21view0turn25view0turn2search0

### Training-time “certified training” vs post hoc verification

What is being verified: during training, optimize a differentiable upper bound on worst-case loss over an input region to obtain models that are easier to certify later (or come with training-time certificates). citeturn2search4turn2search0

Assumptions: the certificate is only as good as the bounding method (IBP/LiRPA variants); usually still assumes ideal arithmetic; tends to target robustness norms rather than broader specs. citeturn2search0turn2search9turn12search11

Methods/tools: IBP-based training; LiRPA-based certified defenses; randomized smoothing for distributional (probabilistic) robustness under Gaussian noise. citeturn2search0turn2search9turn2search2

Benchmarks: robustness-certified accuracy on CIFAR/ImageNet (for smoothing); and VNN-COMP style properties for deterministic verifiers. citeturn2search2turn8view7

Main limitations: certificate tightness vs accuracy tradeoffs; robustness norms dominate the objective; integrating richer specs is nontrivial; and the deployed-semantics gap still applies. citeturn2search0turn12search11turn1search27

### Closed-loop systems and neural controllers

What is being verified: properties of the _system in the loop_ (plant + neural controller), typically safety (“avoid unsafe set”), stability, or reachability over time horizons. citeturn2search3turn2search11turn2search15turn2search19

Assumptions: plant model fidelity; discretization; set representations at tool boundaries; controller architecture restrictions (e.g., activations). citeturn2search19turn2search3turn2search11

Methods/tools: reachability analysis (Taylor models, zonotopes, star sets), hybrid system reasoning, and interfacing plant reachability with network reachability. citeturn2search3turn2search19turn17search3

Benchmarks: specialized CPS benchmarks; some VNN-COMP categories include controller-oriented tasks or systems-adjacent benchmarks. citeturn8view7turn2search3

Main limitations: model mismatch between plant reality and analysis; computational blowup over time; specification is often “avoid unsafe region” rather than mission-level requirements; and deployment semantics (hardware, timing) are often not in the formal model. citeturn2search11turn2search3turn23search8

### Quantized and finite-precision models

What is being verified: robustness/safety of models where arithmetic is fixed-point/low-bit quantized, and properties may fail post-quantization even when they held for the float model. citeturn20search4turn20search0turn20search28turn12search11

Assumptions: bit-precise semantics of shifts/rounding/saturation (often absent in mainstream verifiers); and manageable scale (many methods focus on smaller QNNs). citeturn20search0turn20search28turn20search33

Methods/tools: SMT-based software model checking for QNN implementations (QNNVerifier); ILP/MILP encodings capturing fixed-point primitives; quantization-error differential analysis (QEBVerif). citeturn20search0turn20search28turn20search11turn20search4

Benchmarks: emerging; some research benchmarks; parts of VNN ecosystems are adding support, but this is still far from mainstream. citeturn20search0turn22search32

Main limitations: complexity can worsen (quantization introduces piecewise structure of much higher complexity); solver scaling and tooling; fragile alignment with actual hardware kernels. citeturn20search7turn20search25turn1search27

### Assurance methods that are not full formal verification

Testing/falsification: adversarial attacks, fuzzing, and falsifiers can find counterexamples but do not prove absence of failures; frameworks like DNNF explicitly target enabling falsification at scale by reducing properties. citeturn23search39turn21view0

Runtime monitoring / runtime assurance: architectures like Simplex/RTA wrap an unverified controller with monitors and fallback controllers to provide operational safety, explicitly “bypassing” verifying the high-performance AI controller at design time. citeturn23search8turn23search16turn23search28turn23search0

Certification / standards: safety standards and AI risk frameworks generally require safety cases and evidence, not solver-level proofs of network properties. citeturn18search0turn18search7turn18search15turn18search2

## Current specification methods

### What is used in practice

The dominant “verification-facing” specification layer today is **VNN-LIB**, which aims to standardize satisfiability queries over neural networks, including a formal syntax/semantics and a solver command-line interface. citeturn22search24turn8view1turn22search13

In the VNN-COMP ecosystem, specifications are typically _preconditions over inputs_ (often box constraints / linear constraints) combined with _postconditions over outputs_ (often linear inequalities encoding classification margins), with support for conjunctions/disjunctions to model families of regions or alternative outcomes. VNN-COMP 2024 describes specs as disjunctions over conjunctions of pre/post conditions and uses an inference runtime to interpret outputs for counterexamples. citeturn8view1turn8view0turn6search3

For higher-level authoring, **DNNV** provides a Python-embedded property DSL (DNNP) and a compilation/reduction pipeline to run multiple verifiers on standardized ONNX + property inputs. It was explicitly motivated by format fragmentation and the burden of choosing verifier-specific formats. citeturn22search2turn22search3turn22search1

Research is actively pushing “spec compiler” ideas: the 2024 work on compiling high-level first-order/tensor specs into _trees of VNN-LIB queries_ argues that many high-level languages fail to abstract solver limitations and that common workarounds (multi-network merging) have serious drawbacks. citeturn27view0turn22search24

### Where the specification layer is too low-level or fragmented

VNN-LIB’s satisfiability-query nature and its focus on network variables creates several structural limitations:

- **Hyperproperties / relational specs** (monotonicity, fairness, consistency across two inputs) do not fit naturally because the format is centered on “exists an input violating” style queries and single-network applications. citeturn27view0turn23search2
- **Quantification is constrained**: high-level compilation work notes non-alternating quantifier restrictions arising from current solver technology and VNN-LIB’s satisfiability-query structure. citeturn27view0turn22search24
- **System-level temporal behavior** is largely outside VNN-LIB; CPS verification uses different spec idioms (unsafe sets, reachability) and different tooling stacks. citeturn2search3turn2search11turn23search8
- **Deployed artifact mismatch**: specifications usually talk about the abstract network function, not the compiled kernel, runtime provider behavior, batching effects, or nondeterminism. citeturn1search27turn19search7turn19search1turn19search2turn8view1

A key synthesis: the spec layer is simultaneously (i) too low-level for system engineers to author meaningful requirements, and (ii) too solver-shaped to express many properties engineers actually want. Spec compilation research treats this as a programming-languages problem: build expressive source languages with disciplined compilation to solver-fragments, but that immediately runs into quantification, multi-network, and non-linear constraint support limits. citeturn27view0turn22search14

## Verification targets

A core reason results fail to “transfer” is that verification often targets a different artifact than deployment. The main targets, and common guarantee-loss points, are:

### Model artifacts

This is the training-framework representation (e.g., a PyTorch module). Problems: dynamic control flow, custom ops, nondeterministic kernels, and framework-level semantics that are not preserved under export unless constrained. PyTorch explicitly documents nondeterminism and provides deterministic modes that can degrade performance. citeturn19search1turn19search5

Guarantees lost: if verification assumes deterministic, side-effect-free computation graphs but the original model uses nondeterministic ops or data-dependent control flow, you are verifying a _different program_. citeturn19search1turn19search5turn1search27

### Exported graphs

VNN-COMP standardized on **ONNX** for networks. ONNX defines an IR with operator semantics and versioning via opsets, but real deployments may use custom operators or rely on runtime/provider-specific behavior. citeturn1search21turn19search34turn8view1turn6search3

Guarantees lost: export can change numerics or even precision; an ONNX discussion notes that floating-point attribute precision can be harmed by ONNX helper tooling, degrading accuracy. citeturn19search18

### Quantized models

Quantization can invalidate properties that held for float models; quantization error bound verification (QEBVerif) is explicitly motivated by this mismatch. citeturn20search4turn20search11turn20search29

Guarantees lost: many “verified robustness” results are for float models; deploying int8/int4 kernels changes the function computed, sometimes dramatically. Tools like QNNVerifier aim to reason about finite word-length explicitly. citeturn20search0turn12search11turn1search27

### Compiled kernels and graph-lowered execution

Modern deployment stacks (graph optimizers, fusion, tensor compilers like TVM, provider-specific kernels like TensorRT) perform graph-level rewrites and operator fusion for speed. ONNX Runtime documents graph optimizations; TensorRT documents runtime optimizations and conditions under which determinism can be compromised (e.g., batching neighbors). citeturn19search2turn19search7turn19search4turn19search34

Guarantees lost: even “semantics-preserving” rewrites can shift floating-point rounding error due to reassociation; on GPUs, nondeterminism can arise from atomic operations and parallel reduction order. citeturn19search1turn19search22turn19search7turn1search27turn12search11

### Runtime executions (as observed in deployment)

VNN-COMP explicitly evaluates counterexamples by running inference (using onnxruntime) and, in at least one year’s report, discards tool-provided outputs when they mismatch runtime-computed outputs. This is a strong signal that “what counts” is aligned to a runtime, not purely symbolic semantics. citeturn8view1

Guarantees lost: if your verifier’s semantics differ from the runtime’s floating-point behavior, you can certify properties that do not hold in deployment (“semantic gap”). This is the central thesis of recent work on deployed verification soundness and follow-up work adapting certification to floating-point execution. citeturn1search27turn12search11turn1search19

### End-to-end ML systems and pipelines

This includes preprocessing/postprocessing code, ensembling, retrieval, thresholding, control logic, and serving systems. ONNX Runtime partitions graphs across execution providers, which makes end-to-end behavior provider-dependent. citeturn19search34turn19search27

Guarantees lost: most NN verifiers assume the _network is the system_. In practice, safety violations are often pipeline-level (e.g., preprocessing drift, postprocessing thresholds, batching effects, distributed nondeterminism). This area has far more runtime-monitoring/safety-case work than formal proof. citeturn23search37turn23search8turn18search15turn18search0

## Methods comparison

This section compares methods by _what they can truly prove_, _where they approximate_, and _what deployment scenarios they fit_. A recurring theme is that “sound and complete in theory” is not the same as “sound for the system you deploy.” citeturn25view0turn1search27turn12search11

### SMT

Strengths: expressive constraint languages; natural fit for piecewise-linear activations via case splitting; supports counterexample generation; integrates with DPLL(T) style reasoning. Reluplex explicitly targets SMT over linear arithmetic with ReLU constraints and states verification for DNNs is NP-complete. citeturn24view0

Weaknesses: scaling is hard; heavy branching; non-linear ops require relaxations or specialized theory solvers; practical soundness can be undermined by floating-point and implementation shortcuts. citeturn24view0turn1search27

Scalability limits: strong on medium-size ReLU networks; struggles on large transformers/OD; tends to need domain-specific heuristics. citeturn21view0turn8view7turn16search3

Best-fit use cases: exact reasoning for safety-critical subcomponents; high-assurance debugging; smaller networks where counterexamples matter. citeturn24view0turn21view0

Handles poorly: rich temporal/system specs; distributional guarantees; large-scale mixed-precision compiled artifacts. citeturn27view0turn1search27turn19search7

### MILP / MIP

Strengths: strong formulations for piecewise-linear nets; leverages highly optimized industrial solvers; can compute exact adversarial distortions and certificates for certain networks. citeturn17search5turn17search9turn24view0

Weaknesses: binary-variable explosion; numerical conditioning; and again, deployment-semantics mismatch (MILP solvers operate with floating-point internally; modeling “reals” is idealized). citeturn17search5turn20search31turn1search27

Scalability limits: often better than SMT for some CNN robustness instances, but still limited for modern architectures unless tightly engineered. citeturn17search5turn21view0

Best-fit use cases: exact robustness bounds on moderate networks; instances where solver presolve/structure is favorable. citeturn17search5

Handles poorly: non-linear layers without linear encodings; large OD postprocessing; full quantized semantics unless extended to bit-precise encodings. citeturn20search28turn16search3

### Abstract interpretation

Strengths: sound over-approximation; often fast; extensible via abstract transformers (DeepPoly includes transformers for affine/ReLU/sigmoid/tanh/maxpool). citeturn17search10turn17search11

Weaknesses: precision loss drives “unknown”; research often fights for tightness; might miss counterexamples because it over-approximates rather than searches. citeturn25view0turn21view0

Scalability limits: best among “sound-but-incomplete” approaches; still challenged by deep nets with complex ops. citeturn17search10turn2search9

Best-fit use cases: fast safety envelopes; as bounding components inside complete BaB; training-time certified robustness. citeturn25view0turn2search0turn2search9

Handles poorly: high-precision guarantees on hard instances; relational/hyperproperties unless specialized. citeturn27view0turn23search2

### Branch-and-bound and bound propagation (BaB + LiRPA/relaxations)

Strengths: currently the dominant paradigm for “scalable completeness” on piecewise-linear nets: use fast bounding (LP/dual/LiRPA) and split on ReLU phases to eventually get complete results. Ferrari et al. describe this standard BaB framing and explicitly define soundness/completeness. citeturn25view0

Weaknesses: exponential worst-case; performance dominated by branching heuristics and bound tightness; easy to overfit to benchmarks; completeness often restricted to piecewise-linear activations. citeturn25view0turn21view0

Best-fit use cases: ReLU-heavy verification tasks, especially robustness, where GPU-accelerated bounding can prune aggressively. citeturn25view0turn6search3turn8view7

Handles poorly: complex non-linearities (softmax/exp), postprocessing pipelines, and deployed float/mixed-precision semantics. citeturn16search3turn1search27turn12search11

### Theorem proving

Strengths: highest-assurance end of the spectrum; can formalize semantics and prove meta-theorems; promising for “semantics-first” infrastructure that addresses the deployed-semantics gap. Recent work on floating-point-aware robustness certification formalizes results in a theorem prover, emphasizing that real-arithmetic assumptions can be semantically unsound for deployed floats. citeturn12search11turn12search12

Weaknesses: major engineering and proof burden; scaling to modern models is hard; connecting proofs to real compilers/runtimes is nascent. citeturn12search11turn1search27

Best-fit use cases: verifying verification infrastructure, numeric error bounds, certified kernels/compilers, and high-criticality components. citeturn12search11turn27view0

### Symbolic execution / “program analysis” approaches

Strengths: treats NNs as programs; can reason about implementations (including quantized arithmetic), not just abstract math functions. citeturn20search0turn20search31

Weaknesses: scalability; modeling ML libraries/frameworks; high dependence on accurate semantics models. citeturn20search0turn1search27

Best-fit use cases: quantized models, embedded inference code, security-critical “what does this binary do?” verification. citeturn20search0turn20search28

### Model checking and reachability (CPS focus)

Strengths: natural for temporal safety/stability in closed-loop systems; can incorporate plant dynamics; controls-oriented specs are meaningful. citeturn2search3turn2search11turn2search19

Weaknesses: interface friction between NN analysis and plant reachability; blowup over time/uncertainty; plant mismatch. citeturn2search19turn2search3

Best-fit use cases: autonomous CPS controllers where system-level safety is the requirement. citeturn2search3turn23search8

### Testing/falsification hybrids

Strengths: scalable bug-finding; can search for counterexamples even when proof is hard; integrates into development workflows; DNNF explicitly targets making verification tasks amenable to falsification. citeturn23search39turn21view0

Weaknesses: no proof of absence; sensitive to attack/fuzzer strength; can distort research agendas by rewarding “findable” failures. citeturn23search39turn21view0

### Proof-carrying / certificate-based approaches

Where it exists: some tools are moving toward proof production (e.g., Marabou 2.0 lists proof production as a feature), and specialized domains (e.g., binarized networks) are developing proof generation + checking pipelines for trustworthiness. citeturn22search4turn20search2turn20search9

Why it matters: certificates decouple trust from solver implementation, which is critical when deployed-semantics and floating arithmetic are already making “soundness” fragile. citeturn1search27turn12search11turn20search9

## Research gaps

This section prioritizes gaps that are scientifically interesting, practically important, and not already overcrowded. Each gap includes (i) why it matters, (ii) why it remains unsolved, and (iii) a plausible research project shape.

### Deployment semantics: floating point, nondeterminism, and “verifying the actually executed system”

Why it matters: multiple sources argue that soundness under idealized real arithmetic does not imply soundness under deployed floating-point execution; recent work shows concrete robustness counterexamples can exist even when a verifier certifies robustness, and proposes float-aware adaptations. citeturn1search27turn12search11turn20search31turn1search19

Why unsolved: (a) modeling floating-point faithfully is hard; (b) GPU kernels introduce nondeterminism and hardware-dependent behavior; (c) compilers/runtimes reorder and fuse ops; (d) “correctness” may depend on batching, neighbors, and tactic selection. citeturn19search1turn19search22turn19search7turn19search2turn19search4turn1search27

Plausible project: **semantics-aligned verification contracts** for a restricted deployment pipeline (e.g., ONNX → ONNX Runtime with fixed providers + deterministic settings), producing (1) a formal semantics model, (2) a verifier that reasons about that model (or bounds deviations), and (3) a validation harness that detects semantic drift across provider/compiler versions. Use VNN-COMP’s explicit reliance on runtime evaluation of counterexamples as an anchor for “what semantics counts.” citeturn8view1turn19search2turn19search34turn12search11turn1search27

### Specification language and compilation beyond VNN-LIB’s solver-shaped fragment

Why it matters: current formats are powerful for interoperability but poor for expressing many properties engineers care about (hyperproperties, multi-network composition, quantification). The 2024 compilation work argues existing high-level languages often expose solver limitations rather than abstracting them, and that merging networks into one ONNX file is a problematic workaround. citeturn27view0turn22search24turn22search2

Why unsolved: solver technology is still largely satisfiability-query oriented; alternation and richer logics break current backends; designing a language that is both user-friendly and compilation-friendly is a PL research problem with hard semantic constraints. citeturn27view0turn22search14

Plausible project: a **spec IR + certified compilation pipeline** that (1) supports relational specs (fairness, monotonicity, two-run properties), (2) compiles to query trees with maximal sharing (avoiding ONNX duplication), and (3) emits certificates/witnesses that can be checked against a reference semantics. Evaluate on fairness verification/repair settings and on VNN-COMP benchmarks where relational specs would be natural. citeturn27view0turn23search2turn22search8turn22search24

### Verification of quantized and mixed-precision deployment artifacts

Why it matters: quantization is standard in deployment, and properties verified on float models may not hold after quantization; QEBVerif and QNNVerifier directly target this gap by reasoning about quantized counterparts or implementations. citeturn20search4turn20search0turn20search29turn12search11

Why unsolved: bit-precise semantics and fixed-point primitives (shift/rounding) are awkward for classic MILP/SMT encodings; scalability is hard; alignment with hardware kernels is even harder. citeturn20search28turn20search25turn1search27

Plausible project: **hardware-aware QNN verification** for a targeted family (e.g., int8 convnets with bounded-range activations), integrating (a) bit-precise arithmetic semantics, (b) compiler lowering constraints, and (c) proof-carrying certificates for critical subproperties (e.g., overflow absence + robustness margin degradation bounds). Use recent float-aware certification work as a template for “semantics deviation accounting.” citeturn20search0turn20search28turn12search11

### Benchmarks and evaluation: correcting benchmark-culture distortion

Why it matters: practitioner-oriented analysis finds no single verifier dominates; performance complementarities suggest portfolios; and evaluation protocols vary widely. Competition benchmarks can also bias methods toward tuned heuristics rather than generality. citeturn21view0turn6search3turn8view7

Why unsolved: incentives (public leaderboards), reproducibility costs, and lack of agreed-upon deployed-semantics targets.

Plausible project: a **“semantic realism benchmark suite”**: for each benchmark instance, provide (1) the idealized spec, (2) the deployed execution spec (float/provider), (3) differential tests that expose mismatch, and (4) scoring that rewards correctness under the deployed semantics. Base it on known mismatch issues and onnxruntime-based evaluation patterns. citeturn8view1turn1search27turn19search7turn12search11

### Compositional and incremental verification across evolving systems

Why it matters: deployed models are updated frequently; re-verifying from scratch is inefficient. Incremental verification work proposes theory/data structures for reusing proof effort across updated models and reports speedups on standard benchmarks. citeturn26view0turn24view0

Why unsolved: solver internals are complex; proving safe reuse is nontrivial; updates can invalidate bounds and branch decisions; and deployment pipelines change more than just weights.

Plausible project: **proof/bound transfer across compilation stages**: extend incremental verification beyond “weight tweaks” to include export/optimization changes (opset changes, fusions, quantization). Integrate with a compiler/runtime pipeline so “incremental” aligns with real CI/CD changes. citeturn26view0turn19search2turn19search34turn19search7

### Verification for modern tasks and architectures with meaningful specs

Why it matters: recent benchmarks include ViTs, generative models, NN-for-systems, and object detectors; dedicated OD verification work argues realistic OD remains hard due to non-linear coordinate transforms and IoU metrics, and proposes specialized relaxations and transformations to scale beyond toy models. citeturn8view7turn16search3

Why unsolved: postprocessing complexity; non-linearities; scale; and spec ambiguity (what is “correct” detection?). citeturn16search3turn27view0

Plausible project: **specification + verification co-design** for a targeted pipeline (e.g., single-object detector with constrained postprocessing): define an OD correctness spec that matches deployment metrics, then build verifiers with architecture-aware relaxations, and finally validate against deployed float kernels. citeturn16search3turn19search7turn12search11

### Security-oriented gaps: verified models as attack surfaces

Why it matters: recent work explicitly frames “verification of deployed neural networks” as vulnerable when attackers exploit semantic gaps; there is also prior work on exploiting verified NNs via floating-point numerical error. citeturn1search27turn20search31

Why unsolved: threat models differ; verifying against an adaptive attacker who can manipulate deployment conditions (precision, batching, provider choice) crosses into systems security and PL.

Plausible project: **adversarial deployment semantics**: formalize attacker capabilities over runtime/hardware knobs and prove robustness guarantees that hold under those knobs (or produce monitors/guards that enforce the assumptions required by verifier soundness). citeturn1search27turn19search7turn19search1turn19search34

## Opportunity assessment

### Good for a publishable paper

Work that produces a crisp technical contribution without requiring an entire end-to-end stack rebuild:

- **Float-aware or nondeterminism-aware verification/certification adjustments** for a clear method class (e.g., Lipschitz/global certification, LiRPA-based local certification), with formal guarantees and a reference implementation. This is timely and anchored by recent “deployed soundness” critiques and float-aware certification work. citeturn1search27turn12search11turn1search19
- **Quantized-model verification with bit-precise semantics** in a scoped setting (e.g., int8 ReLU convnets), building on QNNVerifier / MILP encodings and showing real deployment alignment. citeturn20search0turn20search28turn20search33
- **Spec compilation / solver abstraction** that demonstrably expands expressivity (relational specs, multi-network composition) while preserving solver interoperability, building on the 2024 compilation-to-VNN-LIB line. citeturn27view0turn22search24

### Good for a thesis

Thesis-worthy directions are those that require building a coherent stack (semantics → IR → tools → evaluation) and can accumulate multiple publishable units:

- **Semantics-first verification across export/runtime**: formalize a subset of ONNX + runtime provider semantics, prove equivalence (or bounded deviation) between verifier semantics and deployed execution, and integrate into a usable workflow. citeturn1search21turn19search34turn19search2turn1search27turn12search11
- **Compositional verification for neuro-symbolic or pipeline systems** (NN + symbolic code + pre/post): build verified interfaces and assume-guarantee style decomposition. citeturn6search14turn27view0turn23search8
- **Verification for modern non-classification tasks** (OD, generation) by co-designing specs and verifiers and aligning with deployment metrics and kernels. citeturn16search3turn8view7
- **Incremental verification for real ML lifecycle**: extend incremental proof reuse to model updates + compilation changes + quantization + provider changes. citeturn26view0turn19search2turn19search7

### Good for a startup or systems tool

The market pull is strongest where verification meets deployment constraints:

- **Verification CI/CD tooling** that detects semantic drift (export/runtime/provider version changes) and reuses proofs/bounds incrementally. This is “boring engineering,” but it addresses a real adoption barrier. citeturn26view0turn19search34turn19search2turn1search27
- **Quantization-verification toolchain** integrated with mobile/edge deployment flows, with explicit guarantees on post-quantization property preservation or quantified degradation bounds. citeturn20search4turn20search0turn20search29
- **Portfolio-based verifier orchestration** (auto-selection/configuration) grounded in evidence that no single method dominates and complementarities are strong. citeturn21view0

### Likely dead ends or “looks good but won’t land”

- “End-to-end verification of an entire frontier LLM agent with tools and environment” as a single project: too broad, spec-unclear, and dominated by semantics ambiguity and distributional issues rather than solver technique. (This can become a thesis only if aggressively scoped to verifiable subcomponents/spec fragments.) citeturn18search8turn23search37
- Another incremental bound-tightening heuristic paper solely to climb a VNN-COMP leaderboard, unless it unlocks new semantics or new property classes; this space is crowded and benchmark-shaped. citeturn6search3turn21view0turn8view7
- “Prove fairness for deep nets in full generality” without committing to a realistic fairness spec and deployment semantics; fairness verification exists, but scaling and spec agreement remain open, and many projects stall on definitions. citeturn23search2turn27view0

## Reading list

Must-read foundations and tool papers:

- Reluplex: SMT solver for ReLU networks; explicitly positions NN verification as NP-complete and evaluates on ACAS Xu. citeturn24view0
- MILP verification of robustness (Tjeng et al.). citeturn17search5turn17search9
- DeepPoly abstract domain / abstract interpretation for NN certification. citeturn17search10
- BaB verification framing and definitions (soundness/completeness) in modern GPU-based verifiers (Ferrari et al.). citeturn25view0
- Critically assessing verifier performance and evaluation culture (König et al., JMLR 2024). citeturn21view0
- VNN-COMP 2024 report (benchmarks, formats, evaluation rules). citeturn6search3turn8view7turn8view1
- VNN-LIB specification (semantics + CLI goals). citeturn22search24
- DNNV framework and DNNP property DSL. citeturn22search2turn22search3turn22search1
- Marabou 2.0 system description (architecture, analyses, proof production). citeturn22search0turn22search4
- NeuralSAT’s DPLL(T) approach and “engineering matters” tool paper. citeturn17search31turn22search19

Must-read on the deployment-semantics gap:

- “No Soundness in the Real World” and related deployed-verification critiques. citeturn1search27turn1search23
- Float-aware robustness certification formalized in a theorem prover (Murray 2026). citeturn12search11turn16search1
- Formal verification of deployed neural networks via error-bounding / nondeterminism-aware schemes. citeturn1search19turn1search18
- Framework-level nondeterminism documentation (PyTorch determinism/reproducibility) and runtime determinism notes (TensorRT). citeturn19search1turn19search5turn19search7

Quantization and finite-precision verification:

- QNNVerifier (SMT-based model checking for quantized NN implementations). citeturn20search0turn20search23
- MILP encoding for QNNs with bit-precise primitives (EMSOFT’22 style). citeturn20search28
- QEBVerif (quantization error bounds + MILP fallback). citeturn20search4turn20search11
- Quantization-aware certified training (QA-IBP / related). citeturn20academia41turn2search36

Closed-loop and runtime assurance:

- Verisig / Verisig 2.0 for NN controllers in closed-loop systems. citeturn2search3turn2search11
- NNV tool papers for set-based verification and NN control systems. citeturn2search15turn22search32
- Runtime assurance (Simplex/RTA) and monitoring architectures (distinguish from design-time verification). citeturn23search8turn23search16turn23search37turn23search28

Optional but strategically useful (for gaps/projects):

- Incremental verification for deployed model updates (IVAN). citeturn26view0
- High-level spec compilation into VNN-LIB query trees (2024). citeturn27view0
- Object detection robustness verification (IoUCert) as a case study in spec+architecture co-design. citeturn16search3
- Verification-guided shielding / safe RL (runtime enforcement rather than pure NN verification). citeturn23search1turn23search17
- entity["organization","National Institute of Standards and Technology","us standards agency"] AI RMF and the generative AI profile as examples of system-level assurance frameworks that are _not_ solver proofs. citeturn18search0turn18search8turn18search4
- entity["organization","International Organization for Standardization","standards body"] SOTIF / safety standards as context for what “certification” expects vs what NN verification supplies. citeturn18search2turn18search5turn18search1

Top 10 most important unsolved problems

1. **Practical soundness for deployed artifacts**: formally relating verifier semantics to floating-point execution under compilers/runtimes/GPUs, including nondeterminism and batching/provider effects. citeturn1search27turn12search11turn19search7turn19search1turn19search2
2. **A specification layer that scales** from VNN-LIB interoperability to real system requirements (hyperproperties, composition, quantification, temporal/pipeline specs) with principled compilation and reusable proof artifacts. citeturn27view0turn22search24turn22search2
3. **Verification-preserving compilation**: exporting, lowering, and optimizing graphs so that verified properties survive (or degrade in quantified ways) across IR transformations and kernel fusion. citeturn19search2turn1search21turn19search4turn1search27
4. **Bit-precise verification for quantized/mixed-precision models at realistic scale**, aligned with actual hardware primitives and kernels. citeturn20search0turn20search28turn20search4turn20search7
5. **Compositional verification across ML pipelines** (pre/postprocessing, ensembling, control logic, retrieval), not just isolated networks. citeturn19search34turn23search37turn6search14
6. **Verification for modern architectures and tasks** (transformers/ViTs, object detection, generative models) with meaningful, deployment-aligned specs. citeturn8view7turn16search3turn2search9
7. **Portfolio and auto-configuration of verifiers with guarantees**: turning the observed complementarity into reliable toolchains rather than ad hoc benchmarking results. citeturn21view0turn6search3
8. **Closed-loop system guarantees under uncertainty** (plant mismatch, timing, sensing), integrating NN verification and hybrid-system reachability without brittle interfaces. citeturn2search19turn2search3turn23search8
9. **Incremental verification for the ML lifecycle**, extending beyond weight updates to include export/runtime/provider/quantization changes. citeturn26view0turn19search2turn20search4
10. **Security-aware verification threat models** where attackers can exploit semantic gaps (precision, runtime knobs, compilation differences) to invalidate certified claims. citeturn1search27turn19search7turn20search31

Top 5 most promising thesis directions

1. **Semantics-aligned verification for a real deployment pipeline** (ONNX + a fixed runtime/provider set), including float/nondeterminism modeling and regression detection across versions. citeturn1search21turn19search34turn19search2turn12search11turn1search27
2. **A next-generation specification/compiler stack**: expressive source specs (relational + quantified + multi-network) compiled to solver queries with sharing, certificates, and minimal “ONNX duplication” hacks. citeturn27view0turn22search24turn22search2
3. **Quantized deployment verification**: bit-precise semantics + scalable verification + quantified relationship to float-model properties (preservation or bounded degradation). citeturn20search0turn20search28turn20search4turn12search11
4. **Verification and repair under realistic objectives**: counterexample-guided repair for properties beyond robustness (e.g., fairness or control safety), with correctness arguments and deployment semantics awareness. citeturn23search2turn23search6turn23search10turn1search27
5. **System-level compositional verification for learning-enabled CPS**: integrate reachability + NN verification with proof-carrying interfaces and runtime assurance fallbacks where full proof is impossible. citeturn2search3turn23search8turn2search19turn23search28

Top 3 ideas likely overrated or saturated

1. **Yet another \(\ell\_\infty\) local robustness verifier improvement** tuned to a benchmark subset (especially if the main contribution is heuristic tightening without expanding semantics/spec scope). citeturn21view0turn6search3turn25view0
2. **“Certification” claims that ignore deployed semantics** (real-arithmetic proofs presented as deployment guarantees) without a concrete float/runtime alignment story. citeturn1search27turn12search11turn19search7
3. **Grand “end-to-end” verification of agentic systems** without scoping to a precise, formal semantics and property class; these projects usually collapse into monitoring/testing rather than true verification. citeturn18search8turn23search37turn23search8
