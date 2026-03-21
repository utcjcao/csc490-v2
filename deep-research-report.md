# Incremental Neural Network Verification in the CROWN Lineage

## Foundations of CROWN-style verification and why linear bounds dominate

Modern “CROWN-lineage” verifiers are built around **linear relaxation based perturbation analysis (LiRPA)**: replace each nonlinearity by **sound linear upper and lower bounds**, then **propagate** these bounds through the network to obtain a **global affine bound** on the verification objective (typically a logit margin or a safety predicate). citeturn26view0turn12view0

Two technical details from the CROWN lineage are essential for understanding efficiency—and also for designing an incremental verifier.

**Affine bound representation.** A common canonical form is: for all perturbed inputs \(x\in\mathcal{S}\), the objective \(f(x)\) is bounded by
\[
\underline{f}(x)=\underline{a}^\top x+\underline{b}\;\le\; f(x)\;\le\;\overline{a}^\top x+\overline{b}=\overline{f}(x),
\]
with \((\underline{a},\underline{b}),(\overline{a},\overline{b})\) computed by bound propagation. auto_LiRPA generalizes this idea to *general computational graphs* and explicitly frames LiRPA as computing such linear functions for output neurons. citeturn26view0turn26view1

**Closed-form concretization over norm-balls.** Once an affine bound in \(x\) is available, the worst case over common perturbation sets often becomes *closed form*. For instance, for an \(\ell_p\)-ball, CROWN derives closed-form global bounds using norm duality (e.g., Hölder): the min/max of \(a^\top x+b\) over the ball reduces to \(a^\top x_0 \pm \epsilon\|a\|_q + b\). citeturn12view1turn26view2  
This “propagate affine → concretize cheaply” pattern is the core reason LiRPA scales and maps well to GPUs: most work becomes matrix/tensor ops plus reductions.

**Where “incrementality” can hook in.** In nearly every CROWN extension, the expensive part is not (only) the final concretization; it is the repeated construction and reuse of intermediate affine forms, relaxation parameters, and constraint sets while exploring subproblems (BaB) or optimizing relaxation tightness. These internal artifacts are the natural candidates for reuse across multiple verification runs.

## CROWN to α/β-CROWN and auto_LiRPA: what is computed and how bounds are made tight cheaply

### CROWN

The original CROWN framework (introduced in entity["organization","NeurIPS","machine learning conference"] 2018) computes **explicit linear bounds** for each output neuron by bounding each activation with two linear functions and then “unwrapping” the network layer-by-layer. citeturn12view0turn12view1

**Quantities computed.** For each neuron at layer \(k\) with pre-activation bounds \([l_r^{(k)},u_r^{(k)}]\), CROWN defines **linear lower/upper surrogates** for the activation:
\[
h^{(k)}_{L,r}(y)=\alpha^{(k)}_{L,r}(y+\beta^{(k)}_{L,r}),\quad
h^{(k)}_{U,r}(y)=\alpha^{(k)}_{U,r}(y+\beta^{(k)}_{U,r}),
\]
such that \(h_{L,r}^{(k)}(y)\le\sigma(y)\le h_{U,r}^{(k)}(y)\) on the interval. citeturn12view0turn12view1  
CROWN then builds layerwise “equivalent” linear weights and biases (captured in matrices/vectors such as \(\Lambda^{(k)},\Omega^{(k)},\Delta^{(k)},\Theta^{(k)}\) in the paper’s notation) to express output bounds as explicit affine functions of the input. citeturn12view1

**Propagation direction.** The derivation is essentially **backward**: starting from the output layer, propagate coefficient vectors/matrices backward through linear layers and through activation relaxations, using sign-dependent case splits so the correct upper/lower line is used. The paper explicitly describes repeating a procedure to “back-propagate” to the input layer to obtain \(f^U_j(x)\) and \(f^L_j(x)\). citeturn12view1

**Bottlenecks.** The dominant cost is forming and multiplying large coefficient matrices/vectors as they are propagated (particularly at the last layers and for many outputs), plus repeatedly computing intermediate bounds \([l^{(k)},u^{(k)}]\) that these relaxations depend on. CROWN provides an explicit polynomial complexity discussion for the bound-computation pipeline. citeturn12view2

**Key efficiency ideas.**
- **Closed-form global bounds** over \(\ell_p\) balls (avoiding LP/MIP for the final step). citeturn12view1
- **Adaptive choice of linear surrogates** per neuron to reduce approximation error (a precursor to later “optimize relaxation parameters” approaches). citeturn12view0turn12view1

**Intermediate artifacts produced.** Pre-activation bounds \([l^{(k)},u^{(k)}]\); per-neuron relaxation parameters \(\alpha_{U/L}^{(k)},\beta_{U/L}^{(k)}\); and per-layer propagated coefficient matrices/vectors (e.g., \(\Lambda,\Omega,\Delta,\Theta\)) forming explicit affine bounds. citeturn12view0turn12view1turn12view2

---

### α-CROWN (optimized LiRPA) via “Fast and Complete”

The paper “Fast and Complete” (entity["organization","ICLR","machine learning conference"] 2021) reframes the LiRPA/CROWN bound as a differentiable function of **free relaxation parameters** and then performs **gradient-based tightening** instead of solving LPs inside branch-and-bound. citeturn15view1turn33view0

**Quantities computed.**
- **Free slope variables \(\alpha\)** for unstable ReLUs: for an unstable ReLU, the lower bound slope \(\alpha^{(i)}_j\in[0,1]\) does not affect soundness as long as it stays within the admissible range. citeturn15view0turn15view1
- These \(\alpha\) values are embedded into **diagonal relaxation matrices** \(D_\alpha\) (and corresponding bias terms), yielding a global affine bound
  \[
  L(x,\alpha)\le f(x)\le U(x,\alpha),
  \]
  where \(L,U\) are linear in \(x\) for fixed \(\alpha\). citeturn15view0turn15view1

**Propagation direction.** The core computation is still **backward bound propagation**: propagate the affine relationship from the output backward through layers (linear layers + relaxed ReLUs) until reaching the input, producing \(L(x,\alpha),U(x,\alpha)\). citeturn15view0turn15view1

**Bottlenecks.**
- Optimizing \(\alpha\) adds an iterative inner loop, and each iteration requires evaluating bound functions and gradients (which, in practice, means multiple bound-propagation passes and substantial memory traffic for intermediate tensors). citeturn15view1turn27view0
- In complete verification, the bigger bottleneck becomes **exploring many BaB subdomains** with tight-enough bounds fast enough to prune. citeturn33view0

**Key efficiency ideas.**
- **Closed-form inner minimization/maximization over \(x\)**: since \(L,U\) are linear in \(x\) for fixed \(\alpha\), one can optimize only \(\alpha\) (outer loop), using closed-form solutions (again via norm duality/Hölder). citeturn15view1
- **Projected gradient descent on \(\alpha\)** (coordinatewise projection to \([0,1]\)), leveraging autodiff and GPUs. citeturn15view1
- **Parallel BaB (“batch splits”)**: rather than split one node at a time, process a batch of subdomains and compute their bounds in one GPU batch to improve hardware utilization. citeturn33view0
- **Minimal LP fallback for completeness**: LiRPA alone may miss infeasible split combinations; the framework uses LP sparingly to check feasibility / guarantee completeness. citeturn15view2turn33view0

**Intermediate artifacts produced.** Optimized \(\alpha\) values (and optionally “best-so-far” snapshots); per-layer \(D_\alpha\) and propagated affine coefficients used to compute \(L,U\); a BaB pool/queue of subdomains; and (when used) LP feasibility results at select nodes. citeturn15view0turn15view1turn33view0  
auto_LiRPA’s implementation interface explicitly supports retaining best \(\alpha\)/\(\beta\) (“keep_best”), initializing \(\alpha\) from a CROWN pass (“init_alpha”), and sharing \(\alpha\) across a layer to trade memory for tightness—these are direct “artifact management” surfaces. citeturn27view0

---

### β-CROWN

β-CROWN extends the optimized bound-propagation idea to **encode BaB split constraints** (which LP solvers traditionally handle) using **optimizable Lagrangian multipliers \(\beta\)**, while keeping the computation “CROWN-like” and GPU-parallel. citeturn18view0turn18view1

**Quantities computed.**
- A key object is an **affine relationship matrix** \(A^{(i)}\) representing the linear relationship between the objective \(f(x)\) and an intermediate pre-activation \(\hat z^{(i)}\). The paper explicitly constructs \(A^{(L-2)}\) from \(A^{(L-1)}\) and shows how a term involving \(\beta\) enters the recursion. citeturn18view0
- **Split constraints multipliers \(\beta\ge 0\)**: they appear as Lagrange multipliers multiplying the split constraints; their propagation yields an objective of the form (paper’s notation) \(\max_{\beta\ge 0}\min_{x\in\mathcal{C}} (a+P\beta)^\top x + q^\top\beta + c\). citeturn18view0turn18view1
- β-CROWN also uses **\(\alpha\)** (as in α-CROWN) and introduces additional sets \(\alpha',\beta'\) when jointly optimizing intermediate-layer bounds. citeturn18view1turn18view2

**Propagation direction.** Still fundamentally **backward propagation of affine forms**, but now each relaxed ReLU layer contributes an extra \(\beta^\top S\)-type term (in their derivation) so that split constraints influence the propagated coefficients. citeturn18view0turn18view1

**Bottlenecks.**
- Optimization over \(\beta\) adds an iterative concave maximization layer; joint optimization of intermediate bounds introduces many variables (\(\alpha',\beta'\)) and can become large. citeturn18view2
- In BaB, the primary bottleneck becomes the number of explored nodes; tighter bounds reduce the tree size but cost more per node. β-CROWN aims to keep per-node computation close to CROWN. citeturn18view1turn18view2

**Key efficiency ideas.**
- **Soundness without convergence:** the paper stresses that *any* \(\beta\ge 0\) yields a valid lower bound; optimization is only for tightness. This is a crucial “warm-start friendly” property. citeturn18view1
- **Projected (super)gradient ascent** with autodiff to optimize \(\beta\) efficiently. citeturn18view0turn18view1
- **Dual interpretation:** β corresponds to dual variables of split constraints in an LP with split constraints; with optimal \(\alpha,\beta\) (and fixed intermediate bounds), β-CROWN can match the LP optimum in principle. citeturn18view1
- **Batch splits in BaB** to fill the GPU, building on the “Fast and Complete” design philosophy. citeturn18view2turn33view0

**Intermediate artifacts produced.** Propagated affine coefficient objects (e.g., \(A^{(i)}\), plus \(P,q,a,c\) in their theorem form); \(\beta\) multipliers for each split constraint; optionally separate \(\alpha',\beta'\) variable sets for intermediate-bound optimization; and a BaB tree/pool. citeturn18view0turn18view1turn18view2

---

### auto_LiRPA

auto_LiRPA is best understood as the **systems abstraction layer** that makes the CROWN lineage extensible and efficient on modern architectures: it lifts LiRPA/CROWN from feed-forward derivations to *DAG computational graphs* (e.g., Transformer blocks, matmul-heavy models), and it exposes bound propagation and bound optimization as differentiable, composable operations. citeturn26view0turn26view1

**Quantities computed.**
- It formalizes bounds as **linear functions of the concatenated perturbed independent nodes** \(X\): \(\underline{W}_o X+\underline{b}_o \le h_o(X)\le \overline{W}_o X+\overline{b}_o\). citeturn26view1
- The notations table and algorithmic sections introduce explicit coefficient artifacts (e.g., linear coefficients and bias terms accumulated during propagation) that mirror the “A/b matrices” in older CROWN papers. citeturn26view1

**Propagation direction and modes.**
- Supports forward mode, backward mode, and hybrid schemes; the paper calls out **IBP+Backward** (generalizing CROWN-IBP) as particularly efficient for certified training. citeturn26view2turn26view3

**Bottlenecks and key optimizations.**
- A recurring bottleneck is scaling bound computation to large label spaces (cost proportional to \(K\), the number of logits) and large graphs. auto_LiRPA introduces **loss fusion**: treat the loss as part of the computational graph and bound it directly, reducing the time complexity dependence on label count and making LiRPA training scale to large \(K\). citeturn26view3turn25view0
- The public documentation shows explicit engineering knobs for optimization-style bounds: choosing optimizer, learning rates for \(\alpha\) and \(\beta\), saving the best parameters (“keep_best”), sharing \(\alpha\) to save memory, and choosing whether to fix intermediate bounds. citeturn27view0

**Intermediate artifacts produced.** Graph-level caches of node bounds; affine coefficient objects per node/operator; optimized relaxation parameters (\(\alpha\), \(\beta\)) and best snapshots; and (with loss fusion) a transformed graph whose output is the loss rather than logits. citeturn26view1turn26view3turn27view0

## Tightening constraints with cutting planes: GCP-CROWN and what it teaches about reuse

### GCP-CROWN

GCP-CROWN generalizes the bound propagation formulation to incorporate **arbitrary linear cutting planes**—including those involving relaxed integer indicator variables—without giving up GPU-friendly bound propagation. citeturn20view0turn20view1

**Quantities computed.**
- It introduces **general cut constraints** that can couple variables across any layers, including pre-activations \(x^{(i)}\), post-activations \(\hat x^{(i)}\), and relaxed ReLU indicators \(z^{(i)}\). The paper writes a single cut as \(\sum_i (h^{(i)\top}x^{(i)}+g^{(i)\top}\hat x^{(i)}+q^{(i)\top}z^{(i)})\le d\), and a set of \(N\) cuts in matrix form with \(H^{(i)},G^{(i)},Q^{(i)}\). citeturn20view0turn20view1
- In the dual, these cuts introduce additional dual variables (denoted \(\beta\) in the GCP-CROWN theorem; conceptually “cut multipliers”), and the bound propagation rule becomes a recursion on variables like \(\nu^{(i)}\) that includes terms from \(H^{(i)},G^{(i)},Q^{(i)}\) weighted by the cut multipliers. citeturn20view1

**Propagation direction.** Derived from the **dual of the LP relaxation with cuts** and then implemented as a layer-by-layer bound propagation procedure (so it stays “CROWN-like” operationally). citeturn20view0turn20view1

**Bottlenecks.**
- The core challenge is not just applying cuts, but **finding high-quality cuts** that tighten relaxations meaningfully. citeturn20view2
- Generic cut generation via MIP solvers is CPU-heavy and scales poorly with network size; GCP-CROWN addresses this by decoupling cut generation and bound propagation. citeturn20view2

**Key efficiency ideas.**
- **Asynchronous CPU–GPU parallelism:** run a MIP solver on CPUs *only* to generate cuts (branching disabled), while the GPU verifier continues BaB with current cuts; newly generated cuts are incorporated on-the-fly. citeturn20view2
- **Global sharing of root cuts:** cuts are generated at (and valid for) the root formulation and therefore remain sound when restricting to subdomains created by BaB split constraints. citeturn20view2
- The verifier structure explicitly emphasizes that if cut generation is slow or returns nothing, the GPU bound-propagation BaB proceeds without being blocked. citeturn20view2

**Intermediate artifacts produced.** A persistent **cut pool** \(\{H^{(i)},G^{(i)},Q^{(i)},d\}\); cut dual variables/multipliers; propagated dual/affine recursion state (e.g., \(\nu^{(i)}\)); and a BaB tree whose nodes consume a shared cut set. citeturn20view0turn20view1turn20view2

**Why this matters for incremental verification.** GCP-CROWN is already an *intra-run incremental system*: it continuously accumulates tightening constraints and applies them across many subdomains. The architectural lesson is that “global reusable artifacts” (cuts) pay off when they are:
1) shared across many subproblems, and  
2) cheap to apply inside the bound-propagation kernel (no per-node LP). citeturn20view2  
An incremental verifier across *multiple* runs can mimic this by persisting a cut pool when it remains sound (more on soundness later).

## Branch-and-bound advancements: BICCOS and GenBaB as explicit reuse mechanisms

### BICCOS

BICCOS (“Branch-and-bound Inferred Cuts with Constraint Strengthening”) was designed precisely to make cut generation **scalable** by exploiting structure discovered during BaB, rather than relying on external MIP solvers. citeturn21view0turn31view1

**Quantities computed.**
- **BaB-inferred cuts from UNSAT subproblems:** when a BaB node is verified/UNSAT under a set of split decisions, BICCOS constructs a cut that excludes the corresponding combination of ReLU states globally. The paper states a general cut formulation based on the sets of neurons fixed to positive vs negative regimes in that UNSAT path. citeturn21view0
- **Dual variables tied to fixed indicators:** BICCOS explicitly leverages the dual variables associated with fixed ReLU indicators to reason about which constraints are influential and to strengthen cuts. citeturn31view0
- **Influence scores** per neuron (heuristic): computed from recorded improvements in lower bounds attributable to introducing a neuron’s split; used to decide which split constraints can be dropped while preserving UNSAT. citeturn31view0turn31view1

**Propagation and constraint handling.**
- A key technical point is that BaB fixes some ReLU indicator variables to 0/1, which complicates applying indicator-based cuts naively. BICCOS therefore **extends the GCP-CROWN formulation** to handle branching on indicator variables while keeping cuts shared (avoiding expensive per-subdomain cut rewriting). citeturn21view0

**Bottlenecks.**
- Naively inferred cuts can be ineffective if they are already implied by the branch context of remaining nodes, motivating “constraint strengthening.” citeturn21view0turn31view0
- Searching for good UNSAT explanations is itself costly if done deep in the tree; BICCOS targets shallow UNSAT nodes to make cuts more general. citeturn31view1

**Key efficiency ideas.**
- **Constraint strengthening via neuron elimination:** iteratively drop low-influence split constraints (guided by dual variables and influence scores), re-verify UNSAT, and produce a cut with fewer variables—provably stronger in their corollary sense. citeturn31view0turn31view1
- **Multi-tree search (presolving):** run multiple shallow BaB trees with different branching decisions; cuts found in one tree are globally valid and applied to other trees, amplifying pruning. citeturn31view1
- **Breadth-first search preference** (for cut generation): prioritize nodes closest to the root with few constraints to increase cut generality. citeturn31view1
- **Compatibility with other cut sources:** BICCOS notes its cuts can coexist with GCP-CROWN MIP cuts. citeturn21view0

**Intermediate artifacts produced.** A globally shared pool of inferred cuts; per-node metadata about split sets; dual variable values for fixed indicators; influence score tables; and potentially multiple BaB trees during the presolve phase. citeturn31view0turn31view1

**Direct reuse lesson.** BICCOS is, again, an explicit “reuse engine”: it transfers information from verified subproblems to accelerate the rest of the search. Designing an incremental verifier across runs can treat prior runs as a source of globally reusable constraints—*if* they can be validated under the new run’s conditions.

---

### GenBaB

GenBaB generalizes BaB to **non-piecewise-linear nonlinearities and general computational graphs**, and—crucially for incrementality—introduces a reusable **branch-point lookup table** and a reusable **shortcut-based branching score**. citeturn22view0turn23view2

**Quantities computed.**
- **General branching points:** for non-ReLU nonlinearities, branching may occur at points other than 0, splitting an intermediate bound interval at a chosen point \(c\). citeturn22view0turn23view3
- **Pre-optimized branching points:** GenBaB enumerates possible (lower, upper) bound pairs and chooses branching points that optimize relaxation tightness, storing them into a lookup table used during verification. citeturn23view1turn23view3
- **Stored linear bounds for intermediate nodes:** GenBaB records linear bound parameters (their Eq. (5) style “linear bounds propagated to input”) during initial verification and reuses them to score branch candidates cheaply. citeturn23view2

**Propagation and branching heuristic.**
- Bounding within each BaB subdomain still uses linear bound propagation; GenBaB’s novelty is the **branching** step.
- The proposed heuristic **BBPS (Bound Propagation with Shortcuts)** computes a branch score by using saved linear bounds for an intermediate node and directly concretizing them at the input as a shortcut, rather than re-propagating the entire network for each candidate. citeturn23view2turn23view0

**Bottlenecks and efficiency ideas.**
- Branching heuristics can dominate runtime if each candidate requires recomputing tight verified bounds; GenBaB avoids this by computing scores from cached linear bounds plus cheap concretization. citeturn23view2
- Pre-optimizing branching points is amortized: the paper emphasizes the lookup table is computed once per model and reused across many instances, making pre-optimization cost negligible in aggregate. citeturn23view1

**Intermediate artifacts produced.** (1) a per-model branching-point lookup table; (2) caches of saved linear bounds for candidate intermediate nodes; (3) per-run branching decisions and BaB state. citeturn23view1turn23view2  
GenBaB is therefore an explicit example of cross-instance incrementality built into the verifier design.

## Reusable computation taxonomy, stability under change, and soundness constraints for reuse

This section synthesizes reusable artifacts across CROWN, α/β-CROWN, auto_LiRPA, GCP-CROWN, BICCOS, GenBaB, and SDP-CROWN, and then evaluates their reuse under three change types: small parameter updates, architecture changes, and input-domain changes.

### Linear bound representations

**What they are.** These are the **affine forms** propagated through the network: coefficient vectors/matrices plus biases mapping input (or graph-independent variables) to objective bounds. In CROWN these appear as explicit coefficient matrices/vectors like \(\Lambda,\Omega,\Delta,\Theta\); in auto_LiRPA they appear as \(W,b\)-style linear coefficients on the graph’s independent nodes. citeturn12view1turn26view1

**Reuse potential.**
- **Across small input-domain changes:** high. The propagated affine coefficients for a *fixed relaxation* are reusable, and only the final concretization changes when the set \(\mathcal{S}\) changes; however, many relaxations themselves depend on intermediate pre-activation bounds, which in turn depend on the domain. citeturn12view0turn15view2
- **Across small weight updates:** moderate for warm-start, low for direct reuse. Because the affine coefficients are products/compositions of layer weights and relaxation matrices (e.g., \(W^{(L)}D_\alpha\cdots W^{(1)}\) in optimized LiRPA), a weight update changes them globally. citeturn15view0turn15view1
- **Across architecture edits:** generally low unless the edit is localized and you track dependency boundaries (e.g., adding a head or inserting a layer breaks the coefficient chain).

**Stability vs sensitivity.**
- These artifacts are **continuous** in weights within a fixed relaxation regime, but they are **regime-sensitive**: if a neuron’s pre-activation interval crosses a boundary that changes its relaxation case (e.g., stable vs unstable ReLU), the structure of the bound propagation changes. This sensitivity is explicit in LiRPA’s reliance on pre-activation bounds to decide relaxation cases. citeturn15view0turn15view2turn12view0

**Soundness invariant for reuse.** Reused affine forms must correspond to a relaxation that is still valid for the new run’s intermediate bounds and operator semantics; otherwise they may cease to upper/lower bound the true network. This is why CROWN and its successors tie relaxation parameters to \([l,u]\) intervals. citeturn12view0turn20view0

**Detecting invalidity.** A practical check is to recompute (or cheaply update) the key intermediate bounds that define relaxation regimes and verify that all neurons remain in the same relaxation cases; if not, invalidate or partially recompute.

### Relaxation parameters and dual variables

This includes CROWN’s per-neuron line parameters (\(\alpha,\beta\) in the activation bounds), α-CROWN’s optimizable \(\alpha\in[0,1]\), β-CROWN’s split multipliers \(\beta\ge 0\), GCP-CROWN’s cut multipliers, SDP-CROWN’s \(\lambda\) parameters, and BICCOS’s dual variables tied to fixed indicators. citeturn12view0turn18view1turn20view1turn30view2turn31view0

**Reuse potential.**
- **Warm-start is broadly sound and often cheap.** A central property in β-CROWN is that any \(\beta\ge 0\) yields a valid lower bound; similarly α-CROWN bounds are sound for any \(\alpha\in[0,1]\) (within the parameterization). citeturn18view1turn15view1  
  This makes relaxation parameters excellent warm-start candidates across runs: reuse them as initial values, then re-optimize.
- **Direct reuse without re-optimization:** correctness can remain (often) sound if parameters remain in their admissible set, but tightness can degrade. Fast-and-Complete explicitly motivates optimizing \(\alpha\) because heuristic choices can be very loose. citeturn15view1turn15view2

**Stability vs sensitivity.**
- Optimized parameters can be stable under small changes when the optimization landscape changes smoothly, but they can be brittle when the set of unstable neurons changes (which changes the dimension/meaning of \(\alpha\) and \(\beta\) variables). citeturn15view0turn18view1
- BICCOS explicitly warns that a zero dual variable indicates no contribution *in the current optimization context*—not globally—highlighting context sensitivity. citeturn31view0

**Soundness invariants.**
- Parameter domain constraints must hold (\(\alpha\in[0,1]\), \(\beta\ge 0\), \(\lambda\ge 0\)). citeturn15view1turn18view1turn30view2
- The parameterization must still correspond to the same relaxation template (e.g., the same mapping from \(\alpha\) to a ReLU lower line). citeturn15view0turn12view2

**Detecting invalidity.** Project parameters to the admissible set, re-evaluate bounds, and verify that the relaxation construction remains well-defined for the new \([l,u]\) intervals (e.g., no degenerate cases that change the formula).

### Constraints: cuts, domain restrictions, and neuron splits

**What they are.**
- **Neuron split constraints** in BaB: fixing a ReLU to active/inactive region, producing subdomains. citeturn18view1turn33view0  
- **General cutting planes**: linear constraints involving variables across layers and, potentially, relaxed indicator variables. citeturn20view0turn20view1  
- **BaB-inferred cuts** (BICCOS) derived from UNSAT branches and strengthened by removing unnecessary variables. citeturn21view0turn31view0

**Reuse potential (and the main danger).**
- **Across different subdomains within the same run:** high, and this is already exploited. GCP-CROWN shares root cuts across all BaB subdomains. citeturn20view2  
- **Across multiple verification runs:** only safe under strong conditions:
  - If the *new verification problem is a restriction* of the old one (e.g., smaller input region, stronger constraints), then **valid cuts remain valid** in principle because they were valid for a superset feasible set. This monotonic reuse pattern matches why root cuts are sound for BaB children (children add constraints). citeturn20view2
  - If the new run expands the domain or changes model parameters, previously-valid cuts may become invalid and can break soundness if reused blindly. BICCOS’s need to carefully extend GCP-CROWN to handle fixed indicator variables also illustrates how easy it is to mishandle constraints under changing variable sets. citeturn21view0

**Stability vs sensitivity.**
- **Split constraints** are extremely sensitive to model/domain changes because feasibility of a branch combination can flip.
- **MIP-derived cuts** and **BaB-inferred cuts** are generally *instance-specific*: they rely on the exact LP/MIP relaxation of a particular network + input domain + specification. citeturn20view2turn21view0

**Soundness invariants for reuse.**
- A reused cut must still be a valid inequality for the new problem’s feasible set (in MIP terms, it must not eliminate any feasible integer solutions; in LP-relaxation terms, it must be implied or valid for the intended relaxation). citeturn20view0turn20view1  
- For BICCOS-style inferred cuts, validity depends on the derivation from UNSAT branches; if the underlying UNSAT fact no longer holds, the cut becomes unsound. citeturn21view0

**Detecting invalidity.** In practice, you need a validation step:
- For monotone updates (domain shrinking), validation can be structural.
- For weight changes or domain expansions, you need a re-check—typically by re-solving (or re-bounding) a cut-validity subproblem.

### Search structures: BaB trees, node bounds, and pruning proofs

**What they are.** BaB maintains a tree (or pool) of subdomains formed by split constraints, each with computed bounds and a verification status (SAT/UNSAT/unknown). Fast-and-Complete makes this explicit with a pool \(P\) of unverified subdomains and batch splitting. citeturn33view0turn15view2  
BICCOS extends this to multiple trees in parallel for presolving. citeturn31view1

**Reuse potential.**
- **Across runs with small input-domain changes:** moderate. The *split structure* and the “hardness landscape” can remain similar, so reusing the tree skeleton and branching order can provide value.
- **Across weight updates:** low for direct pruning reuse. A node pruned because its lower bound exceeded 0 under model \(f\) might not be prunable under \(f'\). Therefore, you cannot reuse “pruned = safe” labels without recomputing bounds.

**Stability vs sensitivity.**
- Tree topology is sensitive to any change that alters which neurons are unstable (since branching choices are made on unstable neurons).
- Bound values at nodes can change significantly if tighter relaxations (cuts, α/β optimization) are used differently in the new run. citeturn15view1turn18view1turn20view2

**Soundness invariants.**
- Any pruning decision must be justified by bounds valid for the *current* run. Thus, tree reuse must be treated as a reuse of *structure and warm-starts*, not reuse of final pruning conclusions.

**Detecting invalidity.** Maintain versioning: if the model hash / domain descriptor differs, mark all leaf-node bounds “stale” and recompute before declaring verified.

### Heuristics: branching scores, node priorities, and cut-strengthening rules

**What they are.**
- Fast-and-Complete relies on a BaB framework with batch selection and branching heuristics in parallel. citeturn33view0turn15view2  
- GenBaB introduces BBPS (shortcut-based scoring) and explicitly contrasts it with more aggressive approximations in earlier heuristics. citeturn23view2turn23view0  
- BICCOS defines influence-score-based heuristics for dropping constraints during cut strengthening. citeturn31view0turn31view1

**Reuse potential.** High: heuristics are not themselves soundness-critical (bounds and logical checks are), so reusing learned/hand-tuned heuristic state across runs is usually safe as long as the verifier never treats heuristic output as proof.

**Detecting invalidity.** Monitor performance regressions (e.g., exploding node counts) and fall back to default heuristic configurations; this is safe because it affects only speed, not correctness.

## Concrete strategies for an incremental verifier informed by the CROWN lineage

The following strategies are designed to maximize reuse while preserving the soundness guarantees that CROWN-lineage verifiers rely on. Each strategy is paired with the invariants needed for correctness and checks for invalidity.

### Reusing and incrementally updating linear bounds \(A,b\)

**Idea.** Persist **intermediate affine forms** (per layer or per node in the computational graph) and reuse them as cached “symbolic bound objects,” recomputing only the segments affected by changes.

**How it is informed by CROWN.**
- CROWN explicitly constructs and propagates intermediate coefficient objects (\(\Lambda,\Omega,\Delta,\Theta\)) to build output bounds. citeturn12view1  
- GenBaB’s BBPS stores linear bounds for intermediate nodes and reuses them as shortcuts for branching estimation—demonstrating that storing intermediate affine forms is practical and beneficial. citeturn23view2

**Incremental update rules (sound).**
- **Input-domain change only:** keep the same affine coefficients and redo only concretization (provided the relaxation template and intermediate bounds used to build it remain valid). If intermediate bounds depend on the domain, recompute those bounds for layers where domain change materially affects \([l,u]\). citeturn12view0turn15view2
- **Small weight changes:** treat cached affine forms as warm-start metadata, but recompute coefficients because they are explicit functions of weights (e.g., products \(W^{(L)}D_\alpha\cdots W^{(1)}\)). citeturn15view0turn15view1
- **Architecture edits:** reuse only subgraph-local caches when the computational graph mapping remains unchanged; auto_LiRPA’s graph formalization suggests the right unit of caching is a DAG node/operator boundary. citeturn26view1

**Invalidity detection.** Recompute the pre-activation bounds for a small sentinel set of neurons/layers and check if their stability categories (stable active/inactive vs unstable) changed; if changed, invalidate caches upstream of that boundary. citeturn15view0turn15view2

### Warm-starting optimization of relaxation parameters \(\alpha,\beta,\lambda\)

**Idea.** Persist optimized parameters from a previous run and reuse them as initialization in the next run’s optimization loop.

**Why it should work in this lineage.**
- α-CROWN explicitly optimizes \(\alpha\) via projected gradient descent with \(\alpha\in[0,1]\). citeturn15view1turn15view2
- β-CROWN emphasizes soundness for any \(\beta\ge 0\), making warm-starting natural and safe. citeturn18view1
- SDP-CROWN introduces \(\lambda\ge 0\) parameters that can be optimized together with standard bound-propagation relaxations. citeturn30view2turn30view2
- auto_LiRPA exposes explicit state-management features (“keep_best”, “init_alpha”, separate learning rates for \(\alpha\)/\(\beta\)). citeturn27view0

**Soundness invariant.** Warm-start values must be projected to admissible sets (\([0,1]\), \(\ge 0\)) and bound construction must be rerun to ensure the resulting relaxation is still valid for the current \([l,u]\) bounds. citeturn15view1turn18view1

**Detecting when reuse is no longer helpful.** Track optimization progress: if the first few iterations fail to improve beyond a threshold relative to a cheap baseline (e.g., heuristic \(\alpha\)), reset to default initialization.

### Reusing branch-and-bound trees and pruned regions safely

**Idea.** Persist the BaB tree skeleton (split decisions, node ordering, and possibly node-level parameter warm-starts), but **do not reuse pruning conclusions** unless they can be re-certified cheaply.

**Why this matches existing designs.**
- Fast-and-Complete maintains a pool of unverified subdomains and processes them in batches; this architecture naturally supports saving and reloading a pool/tree. citeturn33view0
- BICCOS runs multiple trees in parallel as a presolve step and then prunes down to one—illustrating that tree structure can be manipulated independently from the final proofs. citeturn31view1

**Soundness rules.**
- For each reused node, recompute its bound under the current model/domain *before* applying prune rules.
- Use the stored split constraints as inputs to β-CROWN/GCP-CROWN style bounding; since split constraints are encoded as part of the bounding objective via multipliers, they remain compatible with warm-starting. citeturn18view1turn20view2

**Detecting invalidity.** If the set of unstable neurons changes materially at a node (e.g., many splits become irrelevant or new unstable neurons appear), mark the subtree as “structurally stale” and rebuild locally.

### Caching and replaying cutting planes

**Idea.** Persist a cut pool and replay it in later runs—*only when validity is preserved*.

**When reuse is sound (practical conditions).**
- **Monotone restriction:** If the new problem is a restriction (e.g., a smaller input region, additional constraints) of the old one, cuts valid for the old feasible set remain valid. This is aligned with GCP-CROWN’s use of root cuts for all BaB subdomains created by adding constraints. citeturn20view2
- **Same problem family:** For GenBaB, a per-model branching-point lookup table is explicitly reusable across instances; this is a “constraint/decision cache” that is structurally tied to the model’s nonlinearities, not to a specific input. citeturn23view1turn23view3

**High-risk reuse situations.**
- Weight updates, domain expansions, or changed specifications can invalidate cuts (especially BICCOS inferred cuts, which encode UNSAT facts). citeturn21view0turn31view1

**Validation strategies.**
- For each persisted cut, attach a provenance descriptor (model ID, relaxation regime summary, domain descriptor).
- If descriptors mismatch in non-monotone ways, revalidate by running a “cut validity check” subproblem (potentially expensive; best applied only to high-impact cuts).

### Sensitivity-guided recomputation

**Idea.** Use sensitivity information already produced by the verifier to decide what must be recomputed after small changes.

**Evidence from the lineage.**
- BICCOS uses dual variables and influence scores to identify which split constraints materially affect bounds, then drops low-influence ones and re-verifies. That is a concrete “sensitivity-driven recomputation” strategy inside a single run. citeturn31view0turn31view1
- GenBaB’s BBPS relies on cached linear bound parameters to estimate potential improvements, effectively prioritizing recomputation where it matters. citeturn23view2

**Incremental verifier adaptation.**
- Maintain per-layer/per-node “impact scores” (e.g., based on gradients of the bound wrt parameters, or observed bound improvements when tightening a layer).
- After a change, recompute only:
  - layers with high impact scores, and
  - nodes where regime changes (stable↔unstable) are detected.

This mirrors BICCOS’s observation that not all constraints matter equally, and that re-verification after dropping constraints is a practical way to maintain correctness while simplifying. citeturn31view0

## System design for an incremental CROWN-lineage verifier and the core tradeoffs

### Persistent state the verifier should maintain

A sound incremental verifier needs persistent state at three levels:

**Model-level state.**
- Computational graph structure and operator metadata (aligned with auto_LiRPA’s DAG formalization). citeturn26view1
- Optional per-model artifacts that are *explicitly reusable*: GenBaB’s branching-point lookup table is a concrete example. citeturn23view1turn23view3

**Bound-propagation state.**
- Cached pre-activation bounds \([l^{(i)},u^{(i)}]\) per node/layer (because relaxation regimes depend on them). citeturn12view0turn15view2
- Cached affine coefficient objects (layerwise or operatorwise), in a representation compatible with GPUs (tensorized rather than dense matrices, where possible).
- Best-known relaxation parameters \(\alpha,\beta,\lambda\) and their optimizer state (momentum terms), leveraging auto_LiRPA’s “keep_best” and optimizer hooks. citeturn27view0turn15view1turn18view1turn30view2

**Search and constraint state.**
- BaB tree/pool representation including split decisions per node (Fast-and-Complete’s pool-based view). citeturn33view0
- Cut pool + provenance metadata (GCP-CROWN, BICCOS). citeturn20view2turn31view1
- Heuristic statistics: branching scores, influence scores, node expansion statistics (BICCOS influence scoring; GenBaB BBPS). citeturn31view0turn23view2

### How updates should be applied

**Input region update.**
1) Recompute only the minimal set of intermediate bounds needed to certify that all relaxation regimes remain valid. citeturn12view0turn15view2  
2) Reuse affine coefficient caches where valid; redo concretization.  
3) Warm-start \(\alpha/\beta/\lambda\) optimization if running optimized bounds. citeturn15view1turn18view1turn30view2  
4) Reuse BaB tree skeleton but recompute node bounds before pruning.

**Small parameter update (training/fine-tuning).**
1) Assume coefficient caches are stale (because coefficients are explicit functions of weights). citeturn15view0turn15view1  
2) Warm-start relaxation optimization parameters because admissibility constraints still apply and β-CROWN guarantees soundness for any \(\beta\ge0\). citeturn18view1  
3) Reuse heuristics and tree skeleton only as scheduling hints; re-certify bounds.

**Architecture update.**
- Use graph-diff to identify unchanged subgraphs and reuse their cached artifacts; otherwise fall back to recomputation guided by node/operator boundaries (consistent with auto_LiRPA’s general-graph bound propagation). citeturn26view1

### Preserving correctness and detecting reuse failure

**Core invariants for soundness.**
- Every reused bound must be reconstructible from a relaxation that is valid for the current \([l,u]\) and current model operators. citeturn12view0turn26view1  
- Every reused pruning decision must be justified by recomputed current bounds (tree reuse cannot skip this). citeturn33view0  
- Every reused cut must remain valid for the new feasible set; if not provably monotone, it must be revalidated. citeturn20view0turn21view0

**Practical invalidity detectors.**
- **Regime change detector:** detect neurons whose stability class changed (e.g., stable↔unstable). This is a high-signal trigger because it changes which relaxations apply. citeturn15view0turn12view2
- **Optimization stagnation detector:** if warm-started optimization fails to recover improvements quickly, reset. citeturn27view0turn15view2
- **Cut effectiveness monitor:** similar to BICCOS’s philosophy—if cuts don’t improve bounds, the algorithm behaves like regular BaB—an incremental verifier should measure marginal benefit and drop stale cuts. citeturn21view0

### Tradeoffs and worst-case behavior

**Reuse overhead vs recomputation cost.**
- Maintaining caches and dependency metadata costs memory and bookkeeping, but can amortize expensive bound propagation and BaB exploration—exactly the kind of amortization GenBaB achieves with its lookup table and BBPS shortcut scoring. citeturn23view1turn23view2

**Memory requirements.**
- Storing full affine forms is expensive; auto_LiRPA explicitly provides memory-saving options like sharing \(\alpha\) within a layer, acknowledging memory as a primary limiter. citeturn27view0  
- Cut pools and multi-tree states (BICCOS) can become large; pruning and merging cuts is an explicit step. citeturn31view1

**Numerical stability and error accumulation.**
- Many bounds are computed through repeated linear algebra transformations; reuse must avoid compounding stale numeric approximations. The safe approach is to treat reuse as warm-start and recompute the final bound numerically each run.

**Worst-case behavior when reuse fails.**
- If caches are frequently invalidated (e.g., training updates flip many neuron regimes), overhead can exceed recomputation and the system should degrade gracefully to the baseline verifier (analogous to BICCOS behaving like regular BaB when it finds no useful cuts). citeturn21view0  
- Similarly, if external cut generation (GCP-CROWN) or inferred cuts (BICCOS) are ineffective, designs that keep the main BaB loop non-blocking ensure the verifier remains correct and progresses. citeturn20view2turn31view1