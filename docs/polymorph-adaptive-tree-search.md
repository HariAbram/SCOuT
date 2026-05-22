# polyMorph Monte Carlo Tree Search

## Purpose

The `polymorph` mode searches over source-to-source loop transformations exposed by Tadashi. The search space is not a fixed vector of independent parameters. A transformation changes the schedule tree, and therefore changes the legality and identity of subsequent transformation sites. The search procedure is consequently formulated as a sequential decision problem over transformation prefixes.

Let a transformation candidate be represented as:

```text
c = (scop, node, transform, arguments)
```

and let a transformation sequence be:

```text
S = [c_1, c_2, ..., c_k]
```

The objective is to find a legal sequence `S` that improves the selected metric, usually parsed SYCL runtime, relative to the baseline program.

## Search State

The adaptive tree search maintains statistics for prefixes of transformation sequences. A prefix is any leading subsequence:

```text
P_i = [c_1, ..., c_i], 0 <= i <= k
```

For each prefix, the search stores:

- number of visits
- accumulated reward
- mean reward
- best reward
- number of pruned evaluations
- number of early-stopped evaluations
- number of failed evaluations

The reward is derived from the primary objective. For a minimization objective such as runtime:

```text
reward = baseline_runtime / candidate_runtime
```

A reward greater than `1.0` indicates an improvement over the baseline.

## Candidate Generation

At every tree step, candidates are enumerated from the current schedule tree, not from the original schedule tree. This is necessary because earlier transformations may rewrite the tree and invalidate earlier node identifiers.

The enumeration procedure is:

1. inspect every SCoP discovered by Tadashi
2. keep only SCoPs whose exported JScop/function identifier looks SYCL-kernel related
3. inspect every node in the current schedule tree for those SCoPs
4. collect transformations currently reported as available for that node
5. infer legal argument lists for each transformation
6. discard candidates rejected by static pruning, analytical constraints, or measured performance bias

This dynamic enumeration avoids applying stale transformation coordinates.

The SYCL SCoP filter is controlled by `search.constraints.sycl_kernel_scop_filter`, enabled by default. It matches markers such as `sycl`, `hipsycl`, `acpp`, `__sscp`, `nd_item`, `handler`, `parallel_for`, `kernel`, `queue`, and `runtest` in Tadashi's exported JScop names. Plain host/helper SCoPs, for example a `main___...jscop` with no kernel marker, are not considered for transformation. If a compiler/runtime uses different outlined-kernel names, override the marker list with `search.constraints.sycl_kernel_scop_markers`. If no SCoP matches at all, the default behavior is to keep all SCoPs and print identifiers so the search does not silently collapse to zero candidates; set `search.constraints.sycl_kernel_scop_filter_fallback_all` to `false` for strict filtering.

## Tree Policy

The tree search uses an upper-confidence style score to rank possible children of the current prefix:

```text
score(P + c) =
    mean_reward(P + c)
  + exploration * sqrt(log(visits(P) + 1) / visits(P + c))
  + prior(c)
  - failure_penalty * failure_rate(P + c)
```

For unvisited children, the score uses a novelty bonus plus the static prior. The prior is obtained from analytical scoring, learned history, pairwise interaction statistics, and measured per-kernel performance feedback when those features are enabled.

The policy balances:

- exploitation of prefixes that previously produced good speedups
- exploration of unvisited transformation sites
- avoidance of prefixes that repeatedly fail, prune, or early-stop

Before deeper sequences are explored, the search performs a short single-transform screening phase. During this phase each trial contains one legal transformation. The resulting measurements provide low-cost estimates of individual transformation quality and prevent longer sequences from being built around transformations that are already poor in isolation.

The search also includes a `STOP` action. This action records the no-transformation baseline as a valid tree outcome, allowing the search to represent the case where no Tadashi transformation is preferable for a kernel.

## Tadashi Baseline and Codegen Fairness

Tadashi's Polly backend generates JScops from LLVM bitcode, rewrites JScop schedules, imports the modified JScops back through Polly, and emits an object file. During transformed code generation, Tadashi currently runs `opt -O3` and `llc -O3` internally.

To avoid comparing a normal source build against a Tadashi-generated `-O3` object, MCTS mode measures the baseline through the same no-op Tadashi path. The baseline has its original schedules, but it is still emitted through Tadashi's JScop import and object-generation pipeline. Candidate trials use the same path after schedule transformation. This makes measured speedups compare schedule changes rather than normal compiler source builds against Tadashi object builds.

The configured `polyMorph.flags` are still passed into Tadashi's initial source-to-bitcode step and into the benchmark build command, but the final Tadashi object emission uses Tadashi's internal `opt -O3` and `llc -O3`. See `docs/polymorph-config-knobs.md` for the full step-by-step command pipeline.

## Prefix Pruning

A prefix is considered unpromising when it has enough observations and no sufficiently good result. The default decision rule is:

```text
if visits(P) >= tree_prune_min_visits
and best_reward(P) < tree_prune_best_speedup_below:
    prune subtree rooted at P
```

A second rule handles repeated early stopping:

```text
if early_stop_count(P) >= tree_prune_after_early_stops
and best_reward(P) < 1.0:
    prune subtree rooted at P
```

These rules prevent the search from repeatedly extending a transformation prefix that is already known to be slower than the baseline.

Individual candidates are also blacklisted when repeated pruned, failed, or early-stopped observations indicate that the same `(scop, node, transform, args)` is consistently unproductive. The default blacklist threshold is controlled by `candidate_blacklist_failures`.

## Sequence Construction

For each trial, the search constructs a sequence incrementally:

```text
S = []
while len(S) < max_transforms_per_trial:
    C = enumerate_live_candidates(current_schedule_tree)
    C = filter_invalid_or_pruned_children(C, S)
    if C is empty:
        break

    c = argmax tree_score(S + c) over C
    apply c to the schedule tree
    append c to S

    if S should not be expanded:
        break
```

The expansion test prevents deeper sequences from being formed until the current prefix has demonstrated sufficient promise.

## Evaluation

After a sequence is selected, the transformed program is generated, built, and measured. The normal evaluation path is:

1. generate transformed source
2. build transformed binary
3. run a low-fidelity measurement if multi-fidelity search is enabled
4. early-stop if the candidate is clearly worse than baseline
5. otherwise run the full measurement repeat count
6. compute per-kernel timing deltas against the baseline
7. classify the regression or improvement pattern
8. update prefix and pairwise interaction statistics
9. append history, cache, and CSV records

The low-fidelity early-stop test uses warmup runs before measuring, so a cold AdaptiveCpp JIT run is not directly compared with the baseline.

## Backend Control

Objective measurements can be pinned to a backend through:

```json
"target_backend": "omp"
```

When this field is present, baseline and transformed objective measurements run with:

```text
ACPP_VISIBILITY_MASK=<target_backend>
```

The backend-sensitivity check may run additional masks, for example:

```json
"backend_sensitivity_masks": ["omp", "ocl"]
```

Those runs compare measured objective values across backends. They are diagnostic measurements and are not used as the primary objective unless the selected measurement backend is explicitly configured to do so.

By default, backend sensitivity is measured once for the baseline and stored in the result JSON. To repeat the backend-sensitivity test for every successful transformed trial, explicitly enable:

```json
"backend_sensitivity_per_trial": true
```

## Persistent Records

The search writes several classes of records:

- trial CSV: trial state, objective, speedup, transforms, metrics, feedback summary
- history JSONL: complete and pruned transformation records
- cache JSONL: reusable evaluation results keyed by source, sequence, and configuration
- result JSON: baseline metrics, backend-sensitivity analysis, best result, and backend-specific summaries

History and cache records are used to seed future prefix statistics. Thus previous failures and improvements affect later search runs over similar kernels. Cache keys include the configured `target_backend`, so measurements from different backend masks are not reused interchangeably.

Setting `replay_top_k` replays the top completed trial sequences after search and stores them under `replay_results` in the result JSON.

## Important Parameters

The following constraint keys control the adaptive tree behavior:

```json
"tree_prune_min_visits": 2,
"tree_prune_best_speedup_below": 0.95,
"tree_prune_after_early_stops": 3,
"tree_min_visits_before_expand": 1,
"tree_expand_min_best_speedup": 0.90,
"tree_exploration": 0.8,
"tree_unvisited_bonus": 1.0,
"tree_failure_penalty": 0.25,
"tree_max_transforms_per_scop": 1,
"candidate_blacklist_failures": 2,
"mcts_include_stop_action": true
```

These parameters are optional. If omitted, the implementation uses internal defaults.

## Interpretation

The adaptive tree search differs from fixed beam search in one central respect: it evaluates and learns from prefixes. If a transformation prefix is consistently worse than the baseline, the subtree rooted at that prefix is suppressed. The search therefore avoids spending many trials on small variants of a poor sequence.

This is important for polyhedral transformation search because transformations are not independent. A profitable transformation in isolation may become harmful when composed with another transformation, and an unprofitable first step should usually not be expanded unless there is evidence that deeper compositions can recover performance.
