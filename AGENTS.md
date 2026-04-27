# Math Specification (LLM-Targeted)
## Fraiman, Lin & Olvera-Cravioto (2024) — Opinion Dynamics on Directed Complex Networks

This document is a structured, high-density reference for use as LLM context when developing code, slides, paper drafts, or Q&A material. It omits pedagogical narrative and redundant explanation. All claims below are either proven in the paper or are direct consequences.

---

## META

```
paper_id: arXiv:2209.00969v2
authors: [Fraiman, N., Lin, T-C., Olvera-Cravioto, M.]
year: 2024
venue: arXiv math.PR preprint
code_available: unknown (verify before Phase 3)
central_contribution: linear stochastic recursion for opinion dynamics on directed random graphs with (1) directed topology, (2) vertex attributes, (3) attribute-dependent external signals; proves stationary distribution exists and is universal in local-weak-limit regime.
core_theorems: [Theorem_1, Theorem_2]
core_propositions: [Proposition_3, Proposition_4, Proposition_5, Proposition_6, Theorem_8]
scope_of_simulations: Figures 1-7, all on directed Erdős-Rényi G(n=1000, p=0.03) or (800 regular + 200 bot) variant; MATLAB; no ensemble averaging reported
```

---

## MODEL_DEFINITION

### Core objects

```
R_i^(k): opinion of vertex i at time k ∈ ℕ
    type: real_number
    range: [-1, +1]
    evolution: Markov chain

G(V, E; 𝒜): directed marked graph
    V: vertex set, |V| = n
    E: directed edges, (i,j) ∈ E ⟹ j listens to i
    d_i^-: in-degree of i (number of inbound neighbors)
    d_i^+: out-degree of i
    marks 𝒜 = {x_i : i ∈ V}

x_i: vertex attribute vector
    = (a_i, q_i, s_i, d_i^-, c(i,1)·𝟙(d_i^-≥1), c(i,2)·𝟙(d_i^-≥2), ...)
    components:
        a_i: generic attribute in Polish space S'
        q_i: internal opinion, ∈ [-1, +1]
        s_i: stubborn flag, ∈ {0, 1}
        d_i^-: in-degree
        c(i, r): weight placed on r-th inbound neighbor, ∈ [0, 1]

Model parameters (global):
    c: total neighbor trust budget, ∈ [0, 1)
    d: media trust budget, ∈ (0, 1]
    constraint: c + d ≤ 1
    (1 - c - d): memory/self-inertia coefficient, ≥ 0

Weight constraint (per vertex with d_i^- > 0):
    ∑_{r=1}^{d_i^-} c(i, r) = c
```

### Recursion (Equation 1)

```
R_i^(k+1) = Σ_{r=1}^{d_i^-} c(i,r) · R_{ℓ(i,r)}^(k)    [neighbor term]
          + W_i^(k)                                    [external signal]
          + (1 - c - d) · R_i^(k)                      [self-inertia term]

where:
    ℓ(i, r) = index of i's r-th inbound neighbor
    W_i^(k) = external signal received by i at time k
```

### External signal specification (Equation 2)

```
W_i^(k) = q_i · (c - Σ_r c(i,r)) + d · Z_i^(k)

dichotomy:
    if d_i^- ≥ 1: Σ_r c(i,r) = c ⟹ W_i^(k) = d · Z_i^(k)
    if d_i^- = 0: Σ_r c(i,r) = 0 ⟹ W_i^(k) = c · q_i + d · Z_i^(k)

Z_i^(k): "media signal"
    range: [-1, +1]
    conditional independence: {Z_i^(k) : k ≥ 1, i ∈ V} mutually independent
    {Z_i^(k) : k ≥ 1} conditionally i.i.d. given x_i
    distribution: modeler choice, conditional law ν(x_i)
```

### Regularity assumption on signals (for Theorem 2)

```
Lipschitz condition on ν under Wasserstein-1 metric:
    d_1(ν(x), ν(x̃)) ≤ K · ρ(x, x̃)    for some K < ∞, all x, x̃ ∈ S

where ρ is the product metric on S given in the paper.
```

---

## THEOREM_1

### Statement

```
assumption: d > 0, locally finite directed marked graph G
claim: ∃ random vector R ∈ [-1, +1]^|V| such that:
    (a) distribution of R does not depend on R^(0)
    (b) R^(k) ⟹ R as k → ∞
    (c) convergence is geometric at rate ≤ (1 - d)
```

### Key operator: Δ

```
Δ: L^∞(V) → L^∞(V)
(Δ f)(i) := Σ_{r=1}^{d_i^-} c(i,r) f(ℓ(i,r)) + (1 - c - d) f(i)

operator_norm:
    ‖Δ‖_∞ = sup_{f ∈ L^∞(V)} ‖Δf‖_∞ / ‖f‖_∞ ≤ 1 - d < 1

consequence: Δ is a contraction on L^∞(V) with modulus ≤ (1 - d).
```

### Recursion as affine map

```
R^(k+1) = Δ R^(k) + W^(k)

iterated:
R^(k) = Δ^k R^(0) + Σ_{r=0}^{k-1} Δ^r W^(k-1-r)

time_reversed (equal in distribution, since {W^(r)} i.i.d.):
R^(k) =_d Δ^k R^(0) + B^(k)
where B^(k) := Σ_{r=0}^{k-1} Δ^r W^(r)
```

### Convergence mechanism

```
component_1 (initial condition decay):
    ‖Δ^k R^(0)‖_∞ ≤ (1-d)^k · ‖R^(0)‖_∞ → 0 geometrically

component_2 (noise accumulation):
    B^(k) is monotone on shifted support [0, 2]
    B^(k) ↗ B := Σ_{r=0}^∞ Δ^r W^(r)   a.s.
    ‖B^(k) - B‖_∞ ≤ 2 Σ_{r=k}^∞ (1-d)^r ≤ (2/d)(1-d)^k → 0

combined:
    ‖R^(k) - (B - 𝟏)‖_∞ ≤ (1-d)^k · (2/d + 2)
    limit: R := B - 𝟏
```

### Important consequence for simulation

```
iteration_count_for_epsilon_accuracy:
    k_ε ≈ log(C/ε) / d
    
example: d = 0.45, ε = 10^-6 ⟹ k_ε ≈ 24
example: d = 0.05, ε = 10^-6 ⟹ k_ε ≈ 280
```

---

## THEOREM_2

### Setup

```
random_graph_sequence: {G(V_n, E_n; 𝒜_n) : n ≥ 1}
    V_n = {1, ..., n}
    𝒜_n = {Y_i : i ∈ V_n}, Y_i = (A_i, Q_i, S_i)
    full_mark_after_edge_sampling: X_i = (A_i, Q_i, S_i, D_i^-, C(i,1)𝟙(D_i^-≥1), ...)

I_n: uniformly chosen vertex in V_n

quantity_of_interest: R_{I_n}
    conditional_distribution:
        P(R_{I_n} ∈ A | G_n) = (1/n) Σ_{i=1}^n 𝟙(R_i ∈ A)
    interpretation: simultaneously (1) distribution of typical individual,
                    (2) empirical distribution of opinions across network
                    (the quantity histograms in Fig 1-7 estimate)
```

### Strong coupling assumption

```
definition_informal: ∃ joint construction of G_n and a random rooted marked
    directed graph G(𝒳) such that the k-hop in-neighborhood of I_n in G_n
    matches the k-hop in-neighborhood of root ∅ in G_∅(𝒳) with high probability,
    including vertex marks up to ε distance.

examples_with_strong_couplings:
    - Erdős-Rényi directed
    - Chung-Lu (expected degree model)
    - Norros-Reittu (Poissonian random graph)
    - generalized random graph
    - configuration model
    - stochastic block model (multi-type GW tree)
    - preferential attachment

local_weak_limit_by_graph_type:
    ER(n, p=λ/n) ⟹ Poisson-GW tree, offspring distribution Poisson(λ)
    CM with degree dist. p_k ⟹ size-biased GW tree
    SBM ⟹ multi-type marked GW tree
    PA ⟹ non-standard limit (not simple GW, but locally tree-like)
```

### Claim

```
Theorem 2 (paraphrased):
    Under strong coupling + Lipschitz assumption on ν:

    Part 1 (finite-time convergence):
        ∃ sequence {(R^(r), B^(r)) : 0 ≤ r ≤ k} s.t.
        sup_{0≤r≤k} 𝔼[|R_{I_n}^(r) - R^(r)| | 𝒜_n] →_P 0
        (1/n) Σ_i f(R_i^(k)) →_P 𝔼[f(R^(k))]
        for bounded continuous f

    Part 2 (stationary convergence):
        ∃ R* (stationary opinion of root of G(𝒳)) s.t.
        𝔼[|R_{I_n} - R*| | 𝒜_n] →_P 0
        (1/n) Σ_i f(R_i) →_P 𝔼[f(R*)]
        R^(k) ⟹ R* as k → ∞ at geometric rate
```

### Proof technique

```
method: second moment method
key_lemma (Theorem 7 in paper):
    For independent uniformly chosen I_n, J_n ∈ V_n:
    𝔼[|(f(R_{I_n}) - f(R^(t)))(h(R_{J_n}) - h(R̂^(t)))| | 𝒜_n] →_P 0
    
    where R^(t), R̂^(t) are i.i.d. copies of the tree opinion at time t.

covariance_decay_mechanism:
    In sparse graphs, k-hop in-neighborhoods of two random vertices
    are asymptotically disjoint ⟹ opinions decouple ⟹ variance of
    empirical mean converges to 0 ⟹ convergence in probability.
```

---

## PROPOSITION_3

### Explicit tree representation

```
setup: local weak limit G(𝒳) is a tree 𝒯(𝒳)

result:
R* = Σ_{s=0}^∞ Σ_{l=0}^s Σ_{|𝐣|=l} Π_𝐣 · a_{l,s} · W_𝐣^(s)

where:
    𝐣: tree index (ancestry tuple), 𝐣 ∈ ⋃_{k≥0} ℕ_+^k
    |𝐣|: generation/depth of node 𝐣
    Π_𝐣: product of weights on ancestry path
        recursion: Π_{(𝐢, j)} = Π_𝐢 · C(𝐢, j), Π_∅ = 1
    a_{l, s} = C(s, l) · (1 - c - d)^{s - l}  [binomial memory coefficient]
    W_𝐣^(s): external signal at node 𝐣 time s
```

### No-memory simplification (c + d = 1)

```
R* = Σ_{j=1}^{N_∅} C_j R_j + W_∅^(0)

where each R_j is i.i.d. copy of R solving the branching fixed-point equation:
R =_d Σ_{j=1}^{N_1} C(1,j) R_j + W_1^(1)

classification: R is the "special endogenous solution" of the smoothing transform.
```

---

## PROPOSITION_4 (unconditional moments, no-memory)

### Assumptions

```
- assumptions of Theorem 2 hold
- 𝒯(𝒳) is (delayed) marked Galton-Watson tree
- P(N_∅ > 0) = P(N_1 > 0) = 1
- c + d = 1
```

### Results

```
mean:
    𝔼[R*] = d · 𝔼[Z_∅] + c · 𝔼[Z_1]

variance:
    Var(R*) = d² · Var(Z_∅) + (ρ_2* · d²) / (1 - ρ_2) · Var(Z_1)

auxiliary_moments:
    ρ_2*  = 𝔼[Σ_{i=1}^{N_∅} C_i²]
    ρ_2   = 𝔼[Σ_{i=1}^{N_1} C(1, i)²]

    note: ρ_2 < 1 is required for geometric series convergence;
          automatically satisfied when d > 0 and weights respect constraint.
```

---

## PROPOSITION_5 (conditional moments, no-memory)

### Results

```
conditional_mean:
    𝔼[R* | 𝒳_∅] = d · 𝔼[Z_∅ | 𝒳_∅] + c · 𝔼[Z_1]

conditional_variance:
    Var(R* | 𝒳_∅) = d² · Var(Z_∅ | 𝒳_∅)
                  + (d² / (1 - ρ_2)) · Σ_{i=1}^{N_∅} C_i² · Var(Z_1)
```

### Polarization criterion (key result)

```
identity (law of total variance):
    𝔼[Var(R*) - Var(R* | 𝒳_∅)] = d² · Var(𝔼[Z_∅ | 𝒳_∅])

polarization_condition:
    polarization occurs ⟺
        (a) d not too small, AND
        (b) Var(𝔼[Z_∅ | 𝒳_∅]) large
             ⟺ media conditional mean varies across attribute values
             ⟺ selective exposure is strong

consensus_conditions:
    consensus occurs when:
        (a) d small (neighbors dominate, averaging over tree)
        OR
        (b) Var(Z_1) small (everybody hears similar media)
```

### Bimodality diagnostic for Fig 4/5 replication

```
simulate paper's Fig 4 setup: ER(1000, 0.03), c+d∈{1, <1}, selective exposure Beta(8,1)/Beta(1,8)

expected values per paper:
    Var(R*) ≈ 0.1484
    𝔼[R* | Q_∅ = -1] ≈ -0.3684
    𝔼[R* | Q_∅ = +1] ≈ +0.3684
    Var(R* | Q_∅ = ±1) ≈ 0.0095

your replication MUST reproduce these to within simulation error (~1%) to claim fidelity.
```

---

## PROPOSITION_6 (memory reduces variance)

### Comparison

```
general_recursion (M*):
    M_𝐢^(k+1) = Σ_j C(𝐢, j) M_{(𝐢, j)}^(k) + W_𝐢^(k) + (1-c-d) M_𝐢^(k)
    parameters: c, d > 0, c + d < 1

no_memory_recursion (R*, with rescaled weights to preserve proportions):
    R_𝐢^(k+1) = Σ_j (C(𝐢, j)/(c+d)) R_{(𝐢, j)}^(k) + W_𝐢^(k)/(c+d)

result:
    Var(M*) ≤ Var(R*)

interpretation:
    memory term (1 - c - d) smooths individual trajectories over time,
    yielding stationary distribution with smaller variance than the proportionally-
    equivalent no-memory case.

visual_confirmation:
    Fig 6: memory histogram is tighter
    Fig 7: memory trajectory is smoother (less jagged)
```

---

## THEOREM_8 (general moments, appendix)

### Full moment formulas (with memory, general in-degree)

```
notation:
    ρ_1*  = 𝔼[Σ_{i=1}^{N_∅} C_i]
    ρ_1   = 𝔼[Σ_{i=1}^{N_1} C(1, i)]
    ρ_2*  = 𝔼[Σ_{i=1}^{N_∅} C_i²]
    ρ_2   = 𝔼[Σ_{i=1}^{N_1} C(1, i)²]
    Y_𝐢   = 𝔼[W_𝐢^(s) | 𝒳_𝐢]
    V_𝐢   = Var(W_𝐢^(s) | 𝒳_𝐢)
    T     = Geometric random variable on {0,1,2,...} with success prob 1 - ρ_2
    p_T   = Σ_{s=0}^∞ C(s+T, s)² · (1-c-d)^{2s} · (c+d)^{2(T+1)}

mean:
    𝔼[R*] = (1/(c+d)) · 𝔼[W_∅] + ρ_1* · 𝔼[W_1] / ((c+d-ρ_1)(c+d))

variance of R* (full general case): see eq. in Theorem 8 of paper,
    multi-term expression involving Var(Σ_C), Var(Y), Cov(Σ_C, Y), 𝔼[V], etc.
    NOT needed for Figure replication; only needed if variation study uses c+d<1
    and you want theoretical comparison points.
```

### Implementation note

```
for Figure 4/5 replication, Proposition 4/5 formulas suffice (c+d=1 mode).
for Figure 6/7, both modes are needed; general formulas from Theorem 8 give
theoretical predictions for the memory case.
```

---

## CRITIQUE_TARGETS

### Tightly grounded critique points

```
critique_1 (STRONGEST):
    claim_of_paper: "Theorem 2 holds for a broad class of random graphs
                    including ER, CM, Chung-Lu, SBM, PA."
    evidence_in_paper: All simulations (Fig 1-7) use ER only.
    our_test: Variation 1 (re-run Fig 4 on power-law directed graph via
              configuration model with degree dist. P(D=k) ~ k^{-α}).
    expected_outcome: polarization behavior differs materially;
                      Var(R*) shifts because ρ_2 and ρ_2* change;
                      possibly different bimodality character.
    what_would_kill_critique: if power-law graphs give same behavior,
                               critique collapses → report that honestly.

critique_2:
    claim_of_paper: "Selective exposure produces polarization (Fig 4)."
    evidence_in_paper: single parameter point Beta(8,1)/Beta(1,8).
    our_test: Variation 2 (scan Beta asymmetry from (8,1) to (1.5, 1)).
    expected_outcome: identify polarization phase transition threshold;
                      polarization likely degrades continuously, not suddenly.

critique_3:
    claim_of_paper: "Stubborn agents (bots) can polarize opinions (Fig 5)."
    evidence_in_paper: Fig 5 shows a SHIFTED distribution (unimodal, off-center),
                       not a bimodal polarization.
    our_observation: paper conflates "distribution shift by unbalanced bots"
                     with "polarization via echo chamber".
    defense_needed: be careful — paper section 5.3 acknowledges this distinction
                    ("bots cannot target individuals based on their internal
                    opinions, which reduces their effectiveness for polarizing").
                    Critique is valid but needs precise framing.

critique_4:
    claim_of_paper: "this work is relevant for modeling real social networks."
    evidence_in_paper: uses ER graphs throughout.
    our_observation: real social networks are scale-free with heavy-tailed
                     in-degree distributions; also have clustering, which
                     violates tree-like-limit assumption.
    connection_to_critique_1: deeper version of same issue.

critique_5 (reproducibility):
    issue: paper does not report iteration count, random seed, number of
           ensemble replicates, or code repository.
    documentation: our Phase 2 blueprint documents these ambiguities
                   (see fraiman_2024_replication_blueprint.md section E).
    standing: legitimate, minor critique; becomes strong if you actually
              demonstrate divergent outcomes depending on these choices.
```

### Critiques to avoid (insufficient grounding)

```
"the linear model is too simple"
    → rejected: paper acknowledges nonlinear alternatives in Section 6
      (bounded confidence, biased-assimilation); linearity is a deliberate
      modeling choice, not an oversight.

"why not use continuous time?"
    → rejected: paper is discrete-time by construction; continuous time is
      a different modeling paradigm, not a flaw.

"the paper doesn't consider multi-topic dynamics"
    → rejected: paper explicitly states "multi-topic model will be considered
      in future work" (Section 2). Known limitation, already disclosed.

"the paper doesn't prove existence of solutions to the smoothing transform"
    → rejected: paper cites Alsmeyer, Biggins & Meiners (2012) and Aldous &
      Bandyopadhyay (2005) for this; existence is well-established in
      the referenced literature.
```

---

## IMPLEMENTATION_CONTRACTS

### For code development

```
MODULE: graph_construction

function directed_er(n: int, p: float, seed: int) → DirectedGraph
    for each ordered pair (i, j), i ≠ j:
        include edge (i, j) with prob p, independently
    return adjacency_list_by_inbound

function directed_er_with_bots(n_regular: int, n_bots: int, p: float, seed: int)
    regular_subgraph = directed_er(n_regular, p, seed)
    for each bot b:
        for each regular r:
            include edge (b, r) with prob p  # bots have OUT-degree only
        bot has zero in-degree
    return combined_graph

function directed_cm_powerlaw(n: int, alpha: float, seed: int) → DirectedGraph
    # for Variation 1
    sample in-degree and out-degree sequences from power-law distribution
    match total edges; pair via configuration model
    return graph

invariants_after_construction:
    - for each vertex i: len(inbound_neighbors(i)) = d_i^-
    - graph is directed (asymmetric adjacency)


MODULE: attributes

function sample_attributes(graph, scenario: str, seed: int) → dict[vertex → attributes]
    scenario ∈ {"fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7"}
    for each vertex i:
        Q_i distribution by scenario (per paper §5.1-5.4):
            fig1, fig2, fig3 : Q_i ~ Uniform(-1, 1)      continuous
            fig4..fig7       : Q_i ~ Uniform({-1, +1})   discrete binary
            Fig 5 bots       : Q_i = +1                   deterministic
        S_i = 1 if i is bot, 0 otherwise
        weights: c(i, r) = c / d_i^- if d_i^- > 0, else undefined
    return attributes

MODULE: signals

function media_signal(Q_i: float, S_i: int, scenario: str) → sampler_function
    scenario_cases:
        "fig1": Z ~ Uniform(-0.03, 0.03) [independent of Q]
        "fig2": Z ~ Uniform({-1, +1}) [independent of Q]
        "fig3": Z ~ -1 + 2·Beta(1, 8) [independent of Q]
        "fig4", "fig5" (regular): if Q=+1, Z ~ -1 + 2·Beta(8, 1)
                                   if Q=-1, Z ~ -1 + 2·Beta(1, 8)
        "fig5" (bots, S=1): Z = +1 (constant)
        "fig6", "fig7": Z ~ Uniform(-1, 1) [independent of Q]
    return callable that samples Z^(k) at each time step

function external_signal(Q_i, S_i, c, d, c_i_weights_sum, Z_i_k) → W_i_k
    return Q_i * (c - c_i_weights_sum) + d * Z_i_k
    # reduces to d*Z_i_k if vertex has neighbors, c*Q_i + d*Z_i_k if no neighbors

MODULE: dynamics

function one_step_update(R_k: ndarray, graph, attributes, signals_k, c, d) → R_k_plus_1
    R_k_plus_1 = zeros_like(R_k)
    for i in vertices:
        neighbor_term = sum(c(i,r) * R_k[ell(i,r)] for r in range(d_i^-))
        media_term = signals_k[i]  # already W_i_k
        memory_term = (1 - c - d) * R_k[i]
        R_k_plus_1[i] = neighbor_term + media_term + memory_term
    return R_k_plus_1

function run_to_stationarity(graph, attributes, c, d, scenario, n_iter, seed) → R_final
    R_0 ~ Uniform({-1, +1}) on vertices   (paper p.12; stationary is R_0-independent by Thm 1)
    R_k = R_0
    for k in range(n_iter):
        signals_k = sample media signals at time k for all vertices
        R_k = one_step_update(R_k, graph, attributes, signals_k, c, d)
    return R_k

iteration_count_calibration:
    target_precision: 10^-6
    k_iter ≥ ceil(log(10^6) / d)
    fig_4: d=0.45 → k_iter ≥ 31; USE 200 for safety margin
    fig_6: d=0.2  → k_iter ≥ 69; USE 300

MODULE: validation

function validate_theorem1_convergence(graph, attributes, c, d)
    # verify ‖R^(k+1) - R^(k)‖ decays at rate ≤ (1-d) per step

function validate_proposition_4(R_final, attributes, c, d)
    # compute empirical mean/var of R_final
    # compare to 𝔼[R*] = d·𝔼[Z_∅] + c·𝔼[Z_1]
    # compare to Var(R*) per closed form

function validate_proposition_5(R_final, attributes, c, d)
    # partition by Q_i ∈ {-1, +1}
    # compute conditional means/vars
    # compare to paper's reported values (for Fig 4 scenario)
```

### For plotting

```
MODULE: plot

function plot_opinion_histogram(R, attributes, scenario, save_path, paper_params_overlay=True)
    # split by Q ∈ {-1, +1} and plot overlaid histograms
    # overlay: dashed line for underlying media signal distribution
    # axis: [-1, 1] x opinion; y = probability density (normalized)
    # bin count: 50 (matches visual style of paper)

function plot_trajectory(R_over_time, vertex_ids, scenario, save_path)
    # for Fig 7 replication
    # show k on x-axis, R_i^(k) on y-axis for selected vertices

style_rules:
    match paper's colormap: navy/blue for Q=-1, pink/red for Q=+1
    include both memory and no-memory panels side-by-side
    label axes explicitly
```

---

## EXPECTED_RESULTS_TABLE

```
scenario | c       | d         | memory | Var(R*) | E[R*|Q=-1] | E[R*|Q=+1]
---------+---------+-----------+--------+---------+------------+-----------
fig1_mem | 0.001   | 0.3       | yes    | ~0      | 0          | 0
fig1_nom | 0.0033  | 0.9967    | no     | ~0      | 0          | 0
fig2_mem | 0.5     | 0.001     | yes    | small   | 0          | 0          (consensus at 0)
fig2_nom | 0.998   | 0.002     | no     | small   | 0          | 0
fig3_mem | 0.5     | 0.001     | yes    | small   | -0.78      | -0.78      (consensus at mean of Beta(1,8))
fig3_nom | 0.998   | 0.002     | no     | small   | -0.78      | -0.78
fig4_mem | 0.5     | 0.45      | yes    | 0.1484  | -0.3684    | +0.3684
fig4_nom | 0.5263  | 0.4737    | no     | 0.1484  | -0.3684    | +0.3684    (paper: ~same)
fig5_mem | 0.5     | 0.45      | yes    | shifted | shifted    | shifted    (bots push toward +1)
fig6_mem | 0.3     | 0.2       | yes    | moderate| 0          | 0
fig6_nom | 0.6     | 0.4       | no     | larger  | 0          | 0          (Var(M*) ≤ Var(R*))
```

---

## PRESENTATION_SLIDE_MAP

```
slide_M1: "Three forces update each opinion" (2.5 min)
    content:
        - equation (1) color-coded
        - c, d, 1-c-d interpretation
        - equation (2) with dichotomy explanation
    must_say: "neighbor weights sum to c, not 1"

slide_M2: "Theorem 1: opinions settle" (3 min)
    content:
        - Δ operator definition
        - ‖Δ‖_∞ ≤ 1 - d calculation
        - R^(k) = Δ^k R^(0) + accumulated noise
        - geometric decay rate
    must_say: "d > 0 is essential; d = 0 recovers deterministic DeGroot"

slide_M3: "Theorem 2: universality across graphs" (3 min)
    content:
        - local weak limit concept with visual (ER → GW tree)
        - R_{I_n} ⟹ R*
        - list of graph classes this applies to
        - point out: paper's proofs cover many graphs, simulations cover only ER
    must_say: "this is the theoretical achievement, but also where our critique lands"

slide_M4: "Variance decomposition and polarization" (2 min)
    content:
        - Var(R*) formula
        - law-of-total-variance decomposition
        - polarization iff d sizeable AND Var(E[Z|X]) large
    must_say: "polarization has a precise mathematical signature, testable in our variation study"

slide_M5: "From theory to experiment" (0.5 min)
    transition to replication section
```

---

## TIMELINE_INTEGRATION

```
Apr 21-22: Read paper. Fill in this doc's uncertainty points (code_available, etc.)
Apr 23-24: Build code modules per implementation_contracts. Verify Theorem 1 convergence numerically.
Apr 24-25: Replicate Fig 4 → must match expected_results_table (Var(R*) ≈ 0.1484).
           If mismatch: debug before moving on. Do not build variations on broken baseline.
Apr 25: Replicate Fig 5, 6, 7.
Apr 26: Variation 1 (topology). Critical for main critique.
Apr 27: Variations 2, 3, 4.
Apr 28-29: Write paper. Use proposition_X formulas as method-section math.
Apr 30: Repo cleanup + submit.
May 1-3: Build slides following slide_M1 through slide_M5. Dry run.
May 4-8: Present.
```

---

## END_OF_SPEC
